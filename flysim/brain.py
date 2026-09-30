"""Whole-brain leaky integrate-and-fire model on the FlyWire connectome.

Follows Shiu et al., Nature 2024 ("A Drosophila computational brain model reveals
sensorimotor processing"): every neuron is a LIF unit with an exponentially
decaying synaptic current; each synapse adds (synapse count x sign x W_SYN) to
the postsynaptic current after a fixed delay. Stimulated neurons receive Poisson
input strong enough to make them fire at roughly the input rate.

Differences from their Brian2 code, for speed:
- Event-driven propagation (CSR by presynaptic neuron) and an active-neuron
  list: cost scales with activity, not with the size of the brain.
- dt = 0.5 ms instead of 0.1 ms (both integrate the linear dynamics exactly;
  delay and refractory period are rounded to whole steps).

Validation (scripts/validate_brain.py, Shiu's sugar experiment on FlyWire v630,
against a fresh run of their Brian2 model): at dt = 0.1 ms per-neuron rates match
with r = 0.999 and slope 1.01; at dt = 0.5 ms r = 0.998 with rates ~10% high.
"""

import math

import numba
import numpy as np

# Model constants (Shiu et al. 2024).
V_REST = -52.0  # mV, also reset potential
V_THRESHOLD = -45.0  # mV
TAU_MEMBRANE = 20.0  # ms
TAU_SYNAPSE = 5.0  # ms
REFRACTORY = 2.2  # ms
DELAY = 1.8  # ms
W_SYN = 0.275  # mV per synapse
POISSON_WEIGHT = 250 * W_SYN  # mV; each input spike reliably triggers a spike

DEFAULT_DT = 0.5  # ms
# Membrane/current values below this (mV) are treated as rest. Far below the
# 7 mV spike threshold, but lets quiet neurons leave the active list quickly.
# Rates vs Brian2 are unchanged from 0.01 to 0.2 (r = 0.997-0.998).
REST_EPS = 0.1


@numba.njit(nogil=True, cache=True)
def _run(
    n_steps, u, x, refractory, ref_steps, ring, ring_len, ring_head,
    active, is_active, n_active, rest_eps,
    indptr, targets, weights,
    stim_idx, stim_prob, rng_state,
    a_m, a_s, c_ms, u_th, delay_steps, poisson_w,
    spike_counts,
):
    """Advance the network n_steps. u = v - V_REST, x = synaptic current g.

    As in the Brian2 model: a spike resets both u and x to 0; during the
    refractory period neither is integrated and synaptic input is discarded;
    Poisson input kicks u directly. Spikes emitted at step t are stored in ring
    slot (t + delay) and delivered then.

    Only neurons in the active list are integrated: a neuron joins when it gets
    input and leaves once it is back at rest, so the cost of a step scales with
    recent activity rather than with the size of the brain.
    Returns (new ring head, new active count).
    """
    n_slots = ring_len.shape[0]
    np.random.seed(rng_state)
    for _ in range(n_steps):
        # 1. Deliver spikes whose delay has elapsed. Refractory neurons ignore
        #    input, as they do in the Brian2 model.
        slot = ring_head
        for k in range(ring_len[slot]):
            pre = ring[slot, k]
            for s in range(indptr[pre], indptr[pre + 1]):
                t = targets[s]
                if refractory[t] == 0:
                    x[t] += weights[s]
                    if not is_active[t]:
                        is_active[t] = True
                        active[n_active] = t
                        n_active += 1
        ring_len[slot] = 0

        # 2. External Poisson drive, straight onto the membrane.
        for k in range(stim_idx.shape[0]):
            if np.random.random() < stim_prob[k]:
                i = stim_idx[k]
                u[i] += poisson_w
                if not is_active[i]:
                    is_active[i] = True
                    active[n_active] = i
                    n_active += 1

        # 3. Integrate active membranes; queue new spikes `delay_steps` ahead;
        #    compact the active list in place, dropping neurons back at rest.
        out_slot = (ring_head + delay_steps) % n_slots
        kept = 0
        for k in range(n_active):
            i = active[k]
            if refractory[i] > 0:
                refractory[i] -= 1
                active[kept] = i
                kept += 1
                continue
            ui = u[i] * a_m + x[i] * c_ms
            xi = x[i] * a_s
            if ui > u_th:
                ui = 0.0
                xi = 0.0
                refractory[i] = ref_steps[i]
                spike_counts[i] += 1
                m = ring_len[out_slot]
                if m < ring.shape[1]:
                    ring[out_slot, m] = i
                    ring_len[out_slot] = m + 1
            else:
                if -rest_eps < ui < rest_eps:
                    ui = 0.0
                if -rest_eps < xi < rest_eps:
                    xi = 0.0
            u[i] = ui
            x[i] = xi
            if ui != 0.0 or xi != 0.0 or refractory[i] > 0:
                active[kept] = i
                kept += 1
            else:
                is_active[i] = False
        n_active = kept
        ring_head = (ring_head + 1) % n_slots
    return ring_head, n_active


class Brain:
    """LIF network over a signed, weighted connectome.

    Args:
        n_neurons: number of neurons.
        pre, post: synapse endpoints (neuron indices), one entry per connection.
        signed_counts: synapse count x (+1 excitatory / -1 inhibitory).
    """

    MAX_SPIKES_PER_STEP = 20000

    def __init__(
        self, n_neurons: int, pre: np.ndarray, post: np.ndarray, signed_counts: np.ndarray,
        dt: float = DEFAULT_DT, rest_eps: float = REST_EPS,
    ):
        self.n = n_neurons
        self.dt = dt
        self.rest_eps = rest_eps
        order = np.argsort(pre, kind="stable")
        self._targets = post[order].astype(np.int32)
        self._weights = (signed_counts[order] * W_SYN).astype(np.float64)
        self._indptr = np.zeros(n_neurons + 1, dtype=np.int64)
        np.cumsum(np.bincount(pre, minlength=n_neurons), out=self._indptr[1:])

        self._a_m = math.exp(-dt / TAU_MEMBRANE)
        self._a_s = math.exp(-dt / TAU_SYNAPSE)
        # Exact solution of du/dt = (x - u)/tau_m with x decaying at tau_s.
        self._c_ms = TAU_SYNAPSE / (TAU_SYNAPSE - TAU_MEMBRANE) * (self._a_s - self._a_m)
        self._u_th = V_THRESHOLD - V_REST
        # Brian2 counts a neuron refractory while fewer than REFRACTORY/dt steps
        # have elapsed since its spike, i.e. for the REFRACTORY/dt - 1 steps after it.
        self._default_ref_steps = max(0, int(math.floor(REFRACTORY / dt + 1e-9)) - 1)
        self._ref_steps = np.full(n_neurons, self._default_ref_steps, dtype=np.int32)
        self._delay_steps = max(1, int(round(DELAY / dt)))

        n_slots = self._delay_steps + 1
        self._ring = np.zeros((n_slots, self.MAX_SPIKES_PER_STEP), dtype=np.int32)
        self._ring_len = np.zeros(n_slots, dtype=np.int64)
        self._rng_seed = 0
        self.reset()
        self.set_stimulus({})

    def reset(self):
        self.u = np.zeros(self.n)
        self.x = np.zeros(self.n)
        self._refractory = np.zeros(self.n, dtype=np.int32)
        self._ring_len[:] = 0
        self._ring_head = 0
        self._active = np.zeros(self.n, dtype=np.int32)
        self._is_active = np.zeros(self.n, dtype=np.bool_)
        self._n_active = 0
        self.time_ms = 0.0

    @property
    def n_active(self) -> int:
        """Neurons currently away from rest (the ones the kernel integrates)."""
        return self._n_active

    def set_stimulus(self, rates_hz: dict[int, float]):
        """Poisson drive: {neuron index: rate in Hz}. Replaces the previous drive.
        As in Shiu et al., driven neurons have no refractory period."""
        idx = np.fromiter(rates_hz.keys(), dtype=np.int64, count=len(rates_hz))
        rates = np.fromiter(rates_hz.values(), dtype=np.float64, count=len(rates_hz))
        self._ref_steps[:] = self._default_ref_steps
        self._ref_steps[idx] = 0
        self._stim_idx = idx
        self._stim_prob = np.clip(rates * self.dt / 1000.0, 0.0, 1.0)

    def run(self, duration_ms: float) -> np.ndarray:
        """Advance the network; returns the spike count of every neuron."""
        n_steps = int(round(duration_ms / self.dt))
        counts = np.zeros(self.n, dtype=np.int32)
        self._rng_seed = (self._rng_seed + 1) % (2**31)
        self._ring_head, self._n_active = _run(
            n_steps, self.u, self.x, self._refractory, self._ref_steps,
            self._ring, self._ring_len, self._ring_head,
            self._active, self._is_active, self._n_active, self.rest_eps,
            self._indptr, self._targets, self._weights,
            self._stim_idx, self._stim_prob, self._rng_seed,
            self._a_m, self._a_s, self._c_ms, self._u_th,
            self._delay_steps, POISSON_WEIGHT,
            counts,
        )
        self.time_ms += n_steps * self.dt
        return counts

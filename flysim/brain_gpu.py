"""The connectome brain on an NVIDIA GPU (CuPy).

Same model as flysim/brain.py (Shiu et al. LIF, Brian2 semantics: reset of u
and g on a spike, refractory neurons ignore input, Poisson input kicks the
membrane), but all flies' brains advance together as rows of the same arrays
and share one copy of the wiring in GPU memory. Every neuron of every brain is
updated each step (no active list needed on a GPU), so the cost barely depends
on how busy the brains are -- which is what an always-active, autonomous brain
needs.

Each 0.5 ms step is two kernels:
- deliver: spikes whose axonal delay has elapsed add their weights to their
  targets' synaptic current (atomic adds; refractory targets ignore input);
- integrate: exact update of every membrane, Poisson input, threshold, reset,
  and queuing of new spikes `delay` steps ahead.

`GpuBrainEngine` owns the arrays and a thread that runs chunks for every slot;
`GpuBrainLink` is one fly's handle with the same interface as
brain_link.BrainLink.
"""

from __future__ import annotations

import collections
import threading
import time

import numpy as np

from flysim.brain import POISSON_WEIGHT
from flysim.brain_link import CHUNK_MS, MAX_LAG_MS, RATE_TAU_MS, READOUTS, SPONTANEOUS_HZ, BrainModel

MAX_SPIKES_PER_STEP = 20000  # per brain

_KERNELS = r"""
extern "C" {

__device__ __forceinline__ float uniform(unsigned int a, unsigned int b, unsigned int c) {
    // Counter-based hash (PCG-style mixing) -> [0, 1).
    unsigned int h = a * 747796405u + b * 2891336453u + c * 1181783497u + 0x9E3779B9u;
    h ^= h >> 16; h *= 0x7feb352du;
    h ^= h >> 15; h *= 0x846ca68bu;
    h ^= h >> 16;
    return (h >> 8) * (1.0f / 16777216.0f);
}

// One block row per brain (blockIdx.y); blocks stride over that brain's spikes
// in the ring slot, threads stride over each spiking neuron's synapses.
__global__ void deliver(
    const int n, const int n_slots, const int slot, const int max_spk,
    const int* __restrict__ ring_idx, const int* __restrict__ ring_len,
    const long long* __restrict__ indptr, const int* __restrict__ targets,
    const float* __restrict__ weights,
    const int* __restrict__ refractory, float* __restrict__ x)
{
    const int b = blockIdx.y;
    const int count = min(ring_len[b * n_slots + slot], max_spk);
    const int* spikes = ring_idx + ((long long)b * n_slots + slot) * max_spk;
    const long long base = (long long)b * n;
    for (int k = blockIdx.x; k < count; k += gridDim.x) {
        const int pre = spikes[k];
        for (long long s = indptr[pre] + threadIdx.x; s < indptr[pre + 1]; s += blockDim.x) {
            const long long t = base + targets[s];
            if (refractory[t] == 0) atomicAdd(&x[t], weights[s]);
        }
    }
}

// One thread per (brain, neuron).
__global__ void integrate(
    const int n, const int n_brains, const int n_slots, const int head, const int out_slot,
    const int max_spk, const unsigned int* __restrict__ step_base, const unsigned int k,
    const unsigned int seed,
    float* __restrict__ u, float* __restrict__ x,
    int* __restrict__ refractory, const unsigned char* __restrict__ ref_steps,
    const float* __restrict__ stim_prob, const float poisson_w,
    const float a_m, const float a_s, const float c_ms, const float u_th,
    int* __restrict__ ring_idx, int* __restrict__ ring_len, int* __restrict__ counts)
{
    const long long gid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= (long long)n * n_brains) return;
    const int b = (int)(gid / n);
    const int i = (int)(gid - (long long)b * n);
    if (i == 0) ring_len[b * n_slots + head] = 0;  // delivered by `deliver` already

    float ui = u[gid];
    const float p = stim_prob[gid];
    if (p > 0.0f && uniform((unsigned int)gid, step_base[0] + k, seed) < p) ui += poisson_w;

    const int r = refractory[gid];
    if (r > 0) {
        refractory[gid] = r - 1;
        u[gid] = ui;
        return;
    }
    float xi = x[gid];
    ui = ui * a_m + xi * c_ms;
    xi *= a_s;
    if (ui > u_th) {
        ui = 0.0f;
        xi = 0.0f;
        refractory[gid] = ref_steps[gid];
        counts[gid] += 1;
        const int pos = atomicAdd(&ring_len[b * n_slots + out_slot], 1);
        if (pos < max_spk) ring_idx[((long long)b * n_slots + out_slot) * max_spk + pos] = i;
    }
    u[gid] = ui;
    x[gid] = xi;
}

}
"""


class GpuBrainEngine(threading.Thread):
    """All brains, stepped together on the GPU by one thread."""

    def __init__(self, model: BrainModel, capacity: int):
        super().__init__(daemon=True)
        import cupy as cp

        self.cp = cp
        self.model = model
        tpl = model.template
        self.n = n = tpl.n
        self.capacity = capacity
        self.dt = tpl.dt
        self.steps_per_chunk = int(round(CHUNK_MS / tpl.dt))
        self.delay_steps = tpl._delay_steps
        self.n_slots = self.delay_steps + 1
        self.default_ref = tpl._default_ref_steps
        self.params = dict(a_m=tpl._a_m, a_s=tpl._a_s, c_ms=tpl._c_ms, u_th=tpl._u_th)

        mod = cp.RawModule(code=_KERNELS)
        self._deliver = mod.get_function("deliver")
        self._integrate = mod.get_function("integrate")

        # Wiring, shared by every brain.
        self.indptr = cp.asarray(tpl._indptr.astype(np.int64))
        self.targets = cp.asarray(tpl._targets.astype(np.int32))
        self.weights = cp.asarray(tpl._weights.astype(np.float32))

        S = capacity
        self.u = cp.zeros((S, n), dtype=cp.float32)
        self.x = cp.zeros((S, n), dtype=cp.float32)
        self.refractory = cp.zeros((S, n), dtype=cp.int32)
        self.ref_steps = cp.full((S, n), self.default_ref, dtype=cp.uint8)
        self.stim_prob = cp.zeros((S, n), dtype=cp.float32)
        self.counts = cp.zeros((S, n), dtype=cp.int32)
        self.ring_idx = cp.zeros((S, self.n_slots, MAX_SPIKES_PER_STEP), dtype=cp.int32)
        self.ring_len = cp.zeros((S, self.n_slots), dtype=cp.int32)
        self.head = 0
        self.step = 0
        self.step_dev = cp.zeros(1, dtype=cp.uint32)
        # Every GPU op of the engine goes through this stream (needed to record
        # a chunk as a CUDA graph and keep all ops ordered).
        self.stream = cp.cuda.Stream(non_blocking=True)
        self._graph = None
        self.seed = int(np.random.default_rng().integers(1, 2**31))

        # Neurons read out every chunk, gathered on the GPU in one go.
        self._readout_names = list(READOUTS)
        groups = [model.groups[READOUTS[k]] for k in self._readout_names]
        self._readout_idx = cp.asarray(np.concatenate(groups).astype(np.int64))
        self._readout_split = np.cumsum([len(g) for g in groups])[:-1]

        self.links: list[GpuBrainLink | None] = [None] * S
        self.brain_time_ms = 0.0
        self.load = 0.0
        self._wake = threading.Event()
        # Guards the GPU state: a chunk runs under it, and slot resets / rate
        # changes from other threads wait for the chunk to finish.
        self._lock = threading.RLock()
        # Record the CUDA graph now, while no other thread touches the GPU
        # (stream capture fails or hangs if another thread enqueues work).
        self.run_chunk()
        self._clear_all()

    # ------------------------------------------------------------ slots

    def attach(self, link: GpuBrainLink) -> int:
        with self._lock:
            slot = self.links.index(None)
            self.links[slot] = link
            self._clear(slot)
            return slot

    def detach(self, slot: int):
        with self._lock:
            self.links[slot] = None
            self._clear(slot)

    def _clear(self, slot: int):
        with self._lock, self.stream:
            for a in (self.u, self.x, self.refractory, self.stim_prob, self.counts, self.ring_len):
                a[slot] = 0
            self.ref_steps[slot] = self.default_ref

    def min_target_ms(self) -> float:
        targets = [l.target_time_ms for l in self.links if l is not None]
        return min(targets) if targets else self.brain_time_ms

    def _clear_all(self):
        for s in range(self.capacity):
            self._clear(s)

    def set_rates(self, slot: int, rates_hz: np.ndarray):
        """Per-neuron Poisson rates for one brain (driven neurons: no refractory)."""
        cp = self.cp
        prob = np.clip(rates_hz * self.dt / 1000.0, 0.0, 1.0).astype(np.float32)
        ref = np.where(rates_hz > 0, 0, self.default_ref).astype(np.uint8)
        with self._lock, self.stream:
            self.stim_prob[slot] = cp.asarray(prob)
            self.ref_steps[slot] = cp.asarray(ref)

    # ------------------------------------------------------------ stepping

    def _launch_chunk(self):
        """Queue one chunk's kernels on the engine stream (starts at head 0)."""
        n, S = self.n, self.capacity
        threads = 256
        int_blocks = (n * S + threads - 1) // threads
        p = self.params
        self.counts.fill(0)
        head = 0
        for k in range(self.steps_per_chunk):
            out_slot = (head + self.delay_steps) % self.n_slots
            self._deliver(
                (64, S), (128,),
                (np.int32(n), np.int32(self.n_slots), np.int32(head), np.int32(MAX_SPIKES_PER_STEP),
                 self.ring_idx, self.ring_len, self.indptr, self.targets, self.weights,
                 self.refractory, self.x),
            )
            self._integrate(
                (int_blocks,), (threads,),
                (np.int32(n), np.int32(S), np.int32(self.n_slots), np.int32(head), np.int32(out_slot),
                 np.int32(MAX_SPIKES_PER_STEP), self.step_dev, np.uint32(k), np.uint32(self.seed),
                 self.u, self.x, self.refractory, self.ref_steps, self.stim_prob,
                 np.float32(POISSON_WEIGHT), np.float32(p["a_m"]), np.float32(p["a_s"]),
                 np.float32(p["c_ms"]), np.float32(p["u_th"]),
                 self.ring_idx, self.ring_len, self.counts),
            )
            head = (head + 1) % self.n_slots

    def run_chunk(self):
        """Advance every brain one chunk. The chunk's 40-odd kernels are recorded
        once as a CUDA graph and replayed with a single launch, which keeps the
        CPU (busy with physics) out of the way. A chunk is a whole number of
        ring cycles, so each chunk starts at ring head 0."""
        with self.stream:
            self.step_dev.fill(self.step)
            if self._graph is None and self.steps_per_chunk % self.n_slots == 0:
                try:
                    self.stream.begin_capture()
                    self._launch_chunk()
                    self._graph = self.stream.end_capture()
                except Exception:
                    self._graph = False  # capture unsupported: launch kernels directly
            if self._graph:
                self._graph.launch(self.stream)
            else:
                self._launch_chunk()
            self.step += self.steps_per_chunk
            # Wait for the GPU by sleeping: a blocking copy would spin a whole
            # CPU core meanwhile, stealing it from the flies' physics.
            done = self.stream.record()
            while not done.done:
                time.sleep(0.0005)
            # Readouts for every brain; which neurons spiked (for the brain
            # map) only for watched brains: each list costs a GPU round trip.
            readout = self.counts[:, self._readout_idx].get()
            n_spiking = (self.counts > 0).sum(axis=1).get()
            spiking = [None] * self.capacity
            for s, link in enumerate(self.links):
                if link is not None and link.watched:
                    spiking[s] = self.cp.flatnonzero(self.counts[s]).get().astype(np.uint32)
        return readout, n_spiking, spiking

    def run(self):
        while True:
            with self._lock:
                links = [(s, l) for s, l in enumerate(self.links) if l is not None]
            if not links:
                self._wake.wait(0.05)
                self._wake.clear()
                continue
            target = min(l.target_time_ms for _, l in links)
            if target - self.brain_time_ms < CHUNK_MS:
                self._wake.wait(0.05)
                self._wake.clear()
                continue
            for s, l in links:
                l._apply_drive(s)
            t0 = time.perf_counter()
            with self._lock:
                readout, n_spiking, spiking = self.run_chunk()
            self.load += 0.05 * ((time.perf_counter() - t0) * 1000.0 / CHUNK_MS - self.load)
            self.brain_time_ms += CHUNK_MS
            for s, l in links:
                if self.links[s] is l:
                    rows = np.split(readout[s], self._readout_split)
                    l._consume(rows, int(n_spiking[s]), spiking[s])


class GpuBrainLink:
    """One fly's brain on the shared GPU engine; same interface as BrainLink."""

    def __init__(self, engine: GpuBrainEngine):
        self.engine = engine
        self.model = engine.model
        self.groups = engine.model.groups
        self._spontaneous = np.zeros(engine.n)
        for group, hz in SPONTANEOUS_HZ.items():
            self._spontaneous[self.groups[group]] = hz
        self._drive: dict[str, float] = {}
        self._drive_changed = True
        self._lock = threading.Lock()
        self.rates = {name: 0.0 for name in READOUTS}
        self.n_active = 0
        self.target_time_ms = engine.brain_time_ms
        self._spike_log: collections.deque = collections.deque(maxlen=200)
        self._spike_seq = 0
        self._alpha = 1 - np.exp(-CHUNK_MS / RATE_TAU_MS)
        # Set by the server for the fly shown on the brain map.
        self.watched = False
        self.slot = engine.attach(self)

    # thread-compat no-ops: the engine thread does the work
    def start(self):
        if not self.engine.is_alive():
            self.engine.start()

    def is_alive(self) -> bool:
        return self.engine.is_alive()

    @property
    def load(self) -> float:
        return self.engine.load

    @property
    def brain_time_ms(self) -> float:
        return self.engine.brain_time_ms

    # ------------------------------------------------------------ world side

    def set_drive(self, drive: dict[str, float]):
        drive = {k: v for k, v in drive.items() if v > 0}
        with self._lock:
            if drive != self._drive:
                self._drive = drive
                self._drive_changed = True

    def advance_to(self, sim_time_s: float):
        """Let the brains run up to this time; blocks if they lag. The engine
        steps every brain together, up to the *slowest* target, so wait on that
        -- waiting on this fly's own target while another fly hasn't set its
        target yet would deadlock."""
        self.target_time_ms = sim_time_s * 1000.0
        engine = self.engine
        engine._wake.set()
        while engine.min_target_ms() - engine.brain_time_ms > MAX_LAG_MS and engine.is_alive():
            time.sleep(0.0005)

    def sync_clock(self, sim_time_s: float):
        """Align the shared brain clock with the world's (after a world reset,
        or when a fly joins)."""
        self.target_time_ms = sim_time_s * 1000.0
        self.engine.brain_time_ms = self.target_time_ms

    def reset(self):
        """Fresh neural state for this brain (the shared clock goes on)."""
        with self.engine._lock:
            self.engine._clear(self.slot)
        self.rates = {name: 0.0 for name in READOUTS}
        self._drive_changed = True

    def stop(self):
        self.engine.detach(self.slot)

    def spikes_since(self, seq: int) -> tuple[int, np.ndarray]:
        log = list(self._spike_log)
        chunks = [idx for s, idx in log if s > seq]
        latest = log[-1][0] if log else seq
        if not chunks:
            return latest, np.empty(0, np.uint32)
        return latest, np.unique(np.concatenate(chunks))

    # ------------------------------------------------------------ engine side

    def _apply_drive(self, slot: int):
        with self._lock:
            if not self._drive_changed:
                return
            drive = dict(self._drive)
            self._drive_changed = False
        rates = self._spontaneous.copy()
        for group, hz in drive.items():
            idx = self.groups[group]
            rates[idx] = np.maximum(rates[idx], hz)
        self.engine.set_rates(slot, rates)

    def _consume(self, readout_rows, n_spiking: int, spiked: np.ndarray | None):
        rates = dict(self.rates)
        for name, row in zip(self.engine._readout_names, readout_rows):
            hz = row.mean() * 1000.0 / CHUNK_MS if len(row) else 0.0
            rates[name] += self._alpha * (hz - rates[name])
        self.rates = rates
        self.n_active = n_spiking  # neurons that spiked in the last 10 ms
        if spiked is not None:  # watched: feed the live brain map
            self._spike_seq += 1
            self._spike_log.append((self._spike_seq, spiked))

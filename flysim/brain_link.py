"""Runs the connectome brain alongside the physics and exchanges signals with the body.

The brain advances in fixed chunks of simulated time on its own thread (the numba
kernel releases the GIL, so it runs in parallel with MuJoCo). Before each chunk
it reads the current sensory drive (neuron group -> Poisson rate) set by the
world; after it, it publishes smoothed firing rates of the readout groups. The
physics thread waits if the brain falls more than MAX_LAG_MS behind, so the two
stay in the same (possibly slowed-down) time.
"""

import collections
import gc
import threading
import time

import numpy as np

from flysim import connectome, neurons
from flysim.brain import Brain

CHUNK_MS = 10.0
MAX_LAG_MS = 40.0
# Exponential smoothing of readout rates (time constant, ms).
RATE_TAU_MS = 60.0

# Ongoing activity. A real fly's sensory neurons and visual projection neurons
# fire all the time; the model of Shiu et al. starts from a silent brain, which
# never does anything on its own. With this baseline (scripts/spontaneous.py),
# the walking descending neurons fluctuate by themselves: DNp09 fires in
# bouts (forward walking), DNa02/DNa01 wander left and right (turning), while
# escape, feeding and grooming neurons stay below their thresholds. Sensory
# drive from the world adds on top (the higher rate wins per neuron).
# Taste neurons fire very little without food; at 2 Hz the sugar ones alone
# made the fly "feed" on nothing.
SPONTANEOUS_HZ = {"visual_projection": 2.0, "sensory_nonvisual": 2.0, "gustatory": 0.5}

READOUTS = {
    "giant_fiber_left": "giant_fiber_left",
    "giant_fiber_right": "giant_fiber_right",
    "dna02_left": "turn_dna02_left",
    "dna02_right": "turn_dna02_right",
    "dna01_left": "turn_dna01_left",
    "dna01_right": "turn_dna01_right",
    "dnp09_left": "forward_dnp09_left",
    "dnp09_right": "forward_dnp09_right",
    "mdn": "backward_mdn",
    "proboscis": "feeding_mns",
    "grooming": "grooming_dns",
}


def _grooming_dns(groups: neurons.NeuronGroups) -> np.ndarray:
    """Descending neurons the connectome routes bristle/touch input to (found by
    scripts/probe_behaviors.py: the top DNs for the touch and head-bristle
    stimuli; head grooming DNs in the literature are DNg types too)."""
    types = ["DNg84", "DNg35", "DNg57", "DNg15", "DNge036", "DNg85", "DNg48"]
    return np.flatnonzero(np.isin(groups.cell_type, types))


def _feeding_mns(groups: neurons.NeuronGroups) -> np.ndarray:
    """Proboscis motor neurons that sugar drives (scripts/probe_behaviors.py):
    CB0911 and CB0871. Head-bristle touch drives other proboscis motor neurons
    (CB0789, CB0858), so reading all 24 would mistake grooming for feeding."""
    mns = groups["proboscis_mn"]
    return mns[np.isin(groups.cell_type[mns], ["CB0911", "CB0871"])]


class BrainModel:
    """The wiring, loaded once and shared by every fly's brain: connectome with
    corrections, named neuron groups, and a template Brain to clone."""

    def __init__(self):
        t0 = time.perf_counter()
        self.connectome = connectome.load()
        self.groups = neurons.load(self.connectome)
        neurons.apply_dale_signs(self.connectome)
        neurons.apply_antennal_lobe_fixes(self.connectome)
        self.groups.by_name["grooming_dns"] = _grooming_dns(self.groups)
        self.groups.by_name["feeding_mns"] = _feeding_mns(self.groups)
        c = self.connectome
        self.template = Brain(c.n_neurons, c.pre, c.post, c.signed_counts)
        self.template.run(CHUNK_MS)  # JIT warm-up
        # Loading leaves ~200k long-lived Python objects (annotation strings,
        # lookups). Without this, every garbage collection walks them all and
        # the physics loop, which allocates small arrays each step, runs ~2x slower.
        gc.collect()
        gc.freeze()
        self.load_seconds = time.perf_counter() - t0

    # FlyWire coordinates are voxels of 4 x 4 x 40 nm.
    VOXEL_NM = np.array([4.0, 4.0, 40.0])
    SUPER_CLASSES = [
        "optic", "central", "sensory", "visual_projection", "ascending", "descending",
        "sensory_ascending", "visual_centrifugal", "motor", "endocrine",
    ]

    def map_points(self) -> bytes:
        """[uint32 n][float32 n*3 positions in um, centered][uint8 n super-class
        index, 255 = unknown]. Display axes: x = left-right, y = dorsal-up,
        z = anterior (the brain seen from the front)."""
        p = self.groups.positions * self.VOXEL_NM / 1000.0
        known = ~np.isnan(p).any(axis=1)
        center = np.nanmedian(p, axis=0)
        p = np.where(known[:, None], p - center, 0.0)
        xyz = np.stack([p[:, 0], -p[:, 1], -p[:, 2]], axis=1).astype(np.float32)
        lookup = {name: i for i, name in enumerate(self.SUPER_CLASSES)}
        cls = np.array([lookup.get(sc, 255) for sc in self.groups.super_class], dtype=np.uint8)
        cls[~known] = 255
        n = np.array([len(xyz)], dtype=np.uint32)
        return n.tobytes() + xyz.tobytes() + cls.tobytes()

    def map_meta(self) -> dict:
        key = {name: self.groups[group].tolist() for name, group in READOUTS.items()}
        for sense in ("sugar", "bitter", "looming", "touch", "head_bristle", "odor_vinegar"):
            key[f"in_{sense}"] = self.groups[sense].tolist()
        return {"classes": self.SUPER_CLASSES, "groups": key}


class BrainLink(threading.Thread):
    """One fly's brain: its own state and thread, wiring shared via BrainModel."""

    def __init__(self, model: BrainModel, seed: int = 0):
        super().__init__(daemon=True)
        self.model = model
        self.groups = model.groups
        self.brain = model.template.clone(seed)
        c = model.connectome

        self._spontaneous = np.zeros(c.n_neurons)
        for group, hz in SPONTANEOUS_HZ.items():
            self._spontaneous[self.groups[group]] = hz
        self._lock = threading.Lock()
        self._drive: dict[str, float] = {}
        self._drive_changed = True
        self.rates = {name: 0.0 for name in READOUTS}
        self.spike_counts = np.zeros(c.n_neurons, dtype=np.int32)  # last chunk
        self.n_active = 0
        self.brain_time_ms = 0.0
        self.target_time_ms = 0.0
        # Wall-clock seconds the brain needs per simulated second (EMA).
        self.load = 0.0
        self._wake = threading.Event()
        self._reset_requested = False
        self._stopped = False
        self.watched = False  # the CPU brain always logs its spikes
        # (sequence number, indices of neurons that spiked) per chunk, for the
        # live brain map; each viewer keeps its own read position.
        self._spike_log: collections.deque = collections.deque(maxlen=200)
        self._spike_seq = 0

    # ------------------------------------------------------------ brain map

    def spikes_since(self, seq: int) -> tuple[int, np.ndarray]:
        """Neurons that spiked in chunks after `seq`; returns (new seq, indices)."""
        log = list(self._spike_log)
        chunks = [idx for s, idx in log if s > seq]
        latest = log[-1][0] if log else seq
        if not chunks:
            return latest, np.empty(0, np.uint32)
        return latest, np.unique(np.concatenate(chunks))

    # ------------------------------------------------------------ world side

    def set_drive(self, drive: dict[str, float]):
        """Sensory drive: {neuron group name: Poisson rate in Hz}."""
        drive = {k: v for k, v in drive.items() if v > 0}
        with self._lock:
            if drive != self._drive:
                self._drive = drive
                self._drive_changed = True

    def advance_to(self, sim_time_s: float):
        """Let the brain run up to this simulated time; blocks if it lags."""
        self.target_time_ms = sim_time_s * 1000.0
        self._wake.set()
        while self.target_time_ms - self.brain_time_ms > MAX_LAG_MS and self.is_alive():
            time.sleep(0.001)

    def stop(self):
        """End the brain thread (its fly was removed)."""
        self._stopped = True
        self._wake.set()

    def sync_clock(self, sim_time_s: float):
        """Join a world that has already been running (fresh brain state)."""
        self.brain_time_ms = self.target_time_ms = sim_time_s * 1000.0

    def reset(self):
        self._reset_requested = True
        self.target_time_ms = 0.0
        self._wake.set()
        while self._reset_requested and self.is_alive():
            time.sleep(0.001)

    # ------------------------------------------------------------ brain thread

    def _apply_drive(self):
        with self._lock:
            if not self._drive_changed:
                return
            drive = dict(self._drive)
            self._drive_changed = False
        rates = self._spontaneous.copy()
        for group, hz in drive.items():
            idx = self.groups[group]
            rates[idx] = np.maximum(rates[idx], hz)
        self.brain.set_stimulus_array(rates)

    def run(self):
        alpha = 1 - np.exp(-CHUNK_MS / RATE_TAU_MS)
        while not self._stopped:
            if self._reset_requested:
                self.brain.reset()
                self.brain_time_ms = 0.0
                self.rates = {name: 0.0 for name in READOUTS}
                self._drive_changed = True
                self._reset_requested = False
            if self.target_time_ms - self.brain_time_ms < CHUNK_MS:
                self._wake.wait(0.05)
                self._wake.clear()
                continue
            self._apply_drive()
            t0 = time.perf_counter()
            counts = self.brain.run(CHUNK_MS)
            self.load += 0.05 * ((time.perf_counter() - t0) * 1000.0 / CHUNK_MS - self.load)
            self.spike_counts = counts
            self.n_active = self.brain.n_active
            self._spike_seq += 1
            self._spike_log.append((self._spike_seq, np.flatnonzero(counts).astype(np.uint32)))
            rates = dict(self.rates)
            for name, group in READOUTS.items():
                idx = self.groups[group]
                hz = counts[idx].mean() * 1000.0 / CHUNK_MS if len(idx) else 0.0
                rates[name] += alpha * (hz - rates[name])
            self.rates = rates
            self.brain_time_ms += CHUNK_MS

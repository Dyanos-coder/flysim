"""Check the GPU brain against the CPU brain (itself validated against Brian2),
then benchmark the GPU engine with every slot busy.

Run:  uv run python -m scripts.validate_gpu
"""

import time

import numpy as np

from flysim.brain_gpu import GpuBrainEngine
from flysim.brain_link import SPONTANEOUS_HZ, BrainModel
from flysim.perf import disable_windows_power_throttling

SECONDS = 5.0


def gpu_rates(engine, slot, rates, seconds):
    engine._clear(slot)
    engine.set_rates(slot, rates)
    for _ in range(20):  # settle
        engine.run_chunk()
    total = np.zeros(engine.n)
    n_chunks = int(seconds * 100)
    for _ in range(n_chunks):
        engine.run_chunk()
        total += engine.counts[slot].get()
    return total / seconds


def cpu_rates(brain, rates, seconds):
    brain.reset()
    brain.set_stimulus_array(rates)
    brain.run(200.0)
    return brain.run(seconds * 1000.0) / seconds


def main():
    disable_windows_power_throttling()
    model = BrainModel()
    g = model.groups
    cpu = model.template.clone(seed=3)
    cpu.rest_eps = 0.01  # closest to the exact model
    engine = GpuBrainEngine(model, capacity=6)

    spont = np.zeros(model.template.n)
    for group, hz in SPONTANEOUS_HZ.items():
        spont[g[group]] = hz
    sugar = np.zeros(model.template.n)
    sugar[g["sugar"]] = 150.0

    for label, rates in [("sucre 150 Hz", sugar), ("activité spontanée", spont)]:
        c = cpu_rates(cpu, rates, SECONDS)
        gr = gpu_rates(engine, 0, rates, SECONDS)
        m = (c > 0) | (gr > 0)
        r = np.corrcoef(c[m], gr[m])[0, 1]
        slope = np.polyfit(c[m], gr[m], 1)[0]
        print(f"{label}: CPU {np.count_nonzero(c)} active / {c.sum():.0f} spikes/s, "
              f"GPU {np.count_nonzero(gr)} active / {gr.sum():.0f} spikes/s, r = {r:.3f}, slope {slope:.2f}")
        for name in ("feeding_mns", "forward_dnp09", "turn_dna02", "giant_fiber", "grooming_dns"):
            idx = g[name]
            print(f"    {name:14s} CPU {c[idx].mean():6.1f} Hz   GPU {gr[idx].mean():6.1f} Hz")

    # Benchmark: every slot with ongoing activity.
    for s in range(engine.capacity):
        engine._clear(s)
        engine.set_rates(s, spont)
    for _ in range(20):
        engine.run_chunk()
    n = 200
    t = time.perf_counter()
    for _ in range(n):
        engine.run_chunk()
    engine.cp.cuda.Stream.null.synchronize()
    el = time.perf_counter() - t
    sim = n * 0.01
    print(f"\nGPU, {engine.capacity} brains with ongoing activity: {el / sim:.3f} s per simulated s "
          f"(x{sim / el:.1f} realtime for all {engine.capacity} together)")


if __name__ == "__main__":
    main()

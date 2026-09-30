"""Feasibility test for an autonomous fly (docs/autonomie.md, palier 1).

Does ongoing activity -- spontaneous firing of sensory neurons and/or background
synaptic noise -- make the known walking descending neurons (DNp09 forward,
DNa01/DNa02 turning, MDN backward) fire on their own, in a credible way (not
never, not constantly), without setting off escape, feeding or grooming? And at
what compute cost?

Run:  uv run python -m scripts.spontaneous
"""

import time
from collections import Counter

import numpy as np

from flysim.brain_link import BrainModel
from flysim.perf import disable_windows_power_throttling

SECONDS = 5.0
BIN_MS = 100.0

WATCH = {
    "DNp09 (avancer)": ["forward_dnp09_left", "forward_dnp09_right"],
    "DNa02 (tourner)": ["turn_dna02_left", "turn_dna02_right"],
    "DNa01 (tourner)": ["turn_dna01_left", "turn_dna01_right"],
    "MDN (reculer)": ["backward_mdn_left", "backward_mdn_right"],
    "fibre géante": ["giant_fiber_left", "giant_fiber_right"],
    "MN trompe": ["feeding_mns"],
    "DN toilettage": ["grooming_dns"],
}

CONDITIONS = [
    # name, sensory spontaneous rate (Hz), background rate (Hz/neuron), background weight (mV)
    ("silencieux (Shiu)", 0, 0, 0),
    ("A: sensoriel 2 Hz", 2, 0, 0),
    ("A: sensoriel 5 Hz", 5, 0, 0),
    ("B: bruit 20 Hz x 1.5 mV", 0, 20, 1.5),
    ("B: bruit 20 Hz x 3 mV", 0, 20, 3.0),
    ("B: bruit 50 Hz x 3 mV", 0, 50, 3.0),
    ("A+B: 2 Hz + 20 Hz x 3 mV", 2, 20, 3.0),
]


def main():
    disable_windows_power_throttling()
    model = BrainModel()
    g = model.groups
    sensory = np.flatnonzero(g.super_class == "sensory")
    dn = g["descending"]
    brain = model.template.clone(seed=11)

    for name, sens_hz, bg_hz, bg_mv in CONDITIONS:
        brain.reset()
        brain.set_background(bg_hz, bg_mv)
        brain.set_stimulus({int(i): float(sens_hz) for i in sensory} if sens_hz else {})
        brain.run(300.0)  # settle
        n_bins = int(SECONDS * 1000 / BIN_MS)
        series = {k: [] for k in WATCH}
        total = np.zeros(brain.n)
        t0 = time.perf_counter()
        active_sizes = []
        for _ in range(n_bins):
            c = brain.run(BIN_MS)
            total += c
            active_sizes.append(brain.n_active)
            for k, groups in WATCH.items():
                idx = np.concatenate([g[x] for x in groups])
                series[k].append(c[idx].mean() * 1000 / BIN_MS)
        cost = (time.perf_counter() - t0) / SECONDS
        rates = total / SECONDS
        print(f"\n=== {name}: coût {cost:.2f} s/s (x{1 / cost:.1f} temps réel), "
              f"{np.count_nonzero(rates)} neurones actifs, liste active ~{int(np.mean(active_sizes))}, "
              f"taux moyen {rates.mean():.2f} Hz")
        for k, s in series.items():
            s = np.array(s)
            on = (s > 5).mean() * 100
            print(f"   {k:16s} moyenne {s.mean():6.1f} Hz  max {s.max():6.1f}  actif {on:5.1f} % du temps")
        dn_on = dn[rates[dn] > 5]
        print(f"   DN actifs (>5 Hz): {len(dn_on)}/{len(dn)}; top: " +
              ", ".join(f"{g.cell_type[i]}_{str(g.side[i])[:1]}={rates[i]:.0f}" for i in dn_on[np.argsort(-rates[dn_on])][:8]))
    brain.set_background(0, 0)


if __name__ == "__main__":
    main()

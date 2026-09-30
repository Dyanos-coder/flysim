"""Stimulate each sensory group of the v783 brain and report which output
neurons respond. This tells us which sense -> behavior links the connectome
model actually produces before wiring any of them to the body.

Run:  uv run python -m scripts.probe_behaviors
"""

import time

import numpy as np

from flysim import connectome, neurons
from flysim.brain import Brain
from flysim.perf import disable_windows_power_throttling

RATE_HZ = 150.0
TRIAL_MS = 1000.0
N_TRIALS = 3

STIMULI = [
    "sugar_right", "sugar", "bitter", "water",
    "odor_vinegar_left", "odor_geosmin_left",
    "looming_left", "looming",
    "touch", "head_bristle",
]
READOUTS = [
    "giant_fiber_left", "giant_fiber_right",
    "turn_dna02_left", "turn_dna02_right",
    "turn_dna01_left", "turn_dna01_right",
    "forward_dnp09_left", "forward_dnp09_right",
    "backward_mdn_left", "backward_mdn_right",
]


def main():
    disable_windows_power_throttling()
    c = connectome.load()
    g = neurons.load(c)
    changed = neurons.apply_dale_signs(c)
    print(f"Dale's principle: {changed} of {len(c.pre)} connection signs corrected")
    resigned, silenced = neurons.apply_antennal_lobe_fixes(c)
    print(f"Antennal lobe: {resigned} uncertain LNs -> GABA, {silenced} cholinergic LNs' chemical output removed")
    print(f"v783: {c.n_neurons} neurons, {len(c.pre)} connections")
    for name in STIMULI + READOUTS + ["proboscis_mn", "descending"]:
        print(f"  {name:22s} {len(g[name]):5d} neurons")
    brain = Brain(c.n_neurons, c.pre, c.post, c.signed_counts)
    brain.run(1.0)

    for stim in STIMULI:
        rates = np.zeros(c.n_neurons)
        t0 = time.perf_counter()
        for _ in range(N_TRIALS):
            brain.reset()
            brain.set_stimulus({int(i): RATE_HZ for i in g[stim]})
            rates += brain.run(TRIAL_MS)
        rates /= N_TRIALS * TRIAL_MS / 1000
        dt = (time.perf_counter() - t0) / N_TRIALS
        print(f"\n=== {stim} ({len(g[stim])} neurons @ {RATE_HZ:.0f} Hz) — "
              f"{np.count_nonzero(rates)} active, {dt:.2f} s per simulated s")
        print("   " + "  ".join(f"{r.replace('_left', '_L').replace('_right', '_R')}={rates[g[r]].mean():.0f}"
                                 for r in READOUTS))
        pmn = g["proboscis_mn"]
        top_pmn = pmn[np.argsort(-rates[pmn])[:3]]
        print("   proboscis MNs: " + ", ".join(f"{g.cell_type[i]}={rates[i]:.0f}" for i in top_pmn))
        dn = g["descending"]
        top_dn = dn[np.argsort(-rates[dn])[:8]]
        top_dn = [i for i in top_dn if rates[i] > 0]
        print("   top DNs: " + ", ".join(f"{g.cell_type[i]}_{str(g.side[i])[:1].upper()}={rates[i]:.0f}"
                                          for i in top_dn))


if __name__ == "__main__":
    main()

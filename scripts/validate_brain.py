"""Validate flysim.brain against Shiu et al.'s published Brian2 results.

Reproduces their example experiment (21 right-hemisphere sugar GRNs driven by
Poisson input, 30 trials x 1 s, FlyWire v630) and compares per-neuron firing
rates with their `results/example/sugarR.parquet`.

Run:  uv run python -m scripts.validate_brain
"""

import sys
import time

import numpy as np
import pyarrow.parquet

from flysim import connectome
from flysim.brain import Brain
from flysim.perf import disable_windows_power_throttling

V630 = connectome.DATA_DIR / "v630"
SUGAR_R = [
    720575940624963786, 720575940630233916, 720575940637568838, 720575940638202345,
    720575940617000768, 720575940630797113, 720575940632889389, 720575940621754367,
    720575940621502051, 720575940640649691, 720575940639332736, 720575940616885538,
    720575940639198653, 720575940620900446, 720575940617937543, 720575940632425919,
    720575940633143833, 720575940612670570, 720575940628853239, 720575940629176663,
    720575940611875570,
]
MN9 = 720575940660219265
N_TRIALS = 30


def reference_rates(c: connectome.Connectome) -> np.ndarray:
    t = pyarrow.parquet.read_table(V630 / "sugarR.parquet", columns=["flywire_id", "trial"])
    ids = t.column("flywire_id").to_numpy()
    n_trials = int(t.column("trial").to_numpy().max()) + 1
    uniq, counts = np.unique(ids, return_counts=True)
    rates = np.zeros(c.n_neurons)
    rates[c.index_of(uniq)] = counts / n_trials
    return rates


def main():
    disable_windows_power_throttling()
    c = connectome.load(V630 / "completeness_630.csv", V630 / "connectivity_630.parquet")
    print(f"v630: {c.n_neurons} neurons, {len(c.pre)} connections")
    ref = reference_rates(c)
    sugar = c.index_of(SUGAR_R)
    # Stimulated neurons fire at the Poisson rate: read it off the reference.
    r_poi = ref[sugar].mean()
    print(f"reference Poisson rate ~{r_poi:.0f} Hz, MN9 {ref[c.index_of([MN9])[0]]:.1f} Hz, "
          f"{np.count_nonzero(ref)} active neurons")

    dt = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
    brain = Brain(c.n_neurons, c.pre, c.post, c.signed_counts, dt=dt)
    print(f"dt = {dt} ms")
    brain.run(1.0)  # JIT warm-up
    stim = {int(i): round(r_poi / 50) * 50.0 for i in sugar}
    total = np.zeros(c.n_neurons)
    t0 = time.perf_counter()
    for _ in range(N_TRIALS):
        brain.reset()
        brain.set_stimulus(stim)
        total += brain.run(1000.0)
    elapsed = time.perf_counter() - t0
    ours = total / N_TRIALS
    print(f"ours: {N_TRIALS} x 1 s in {elapsed:.1f} s (x{N_TRIALS / elapsed:.1f} realtime), "
          f"{np.count_nonzero(ours)} active neurons, MN9 {ours[c.index_of([MN9])[0]]:.1f} Hz")

    active = (ref > 0) | (ours > 0)
    downstream = active.copy()
    downstream[sugar] = False
    r = np.corrcoef(ref[downstream], ours[downstream])[0, 1]
    print(f"downstream neurons active in either: {downstream.sum()}, rate correlation r = {r:.3f}")

    top = np.argsort(-ref * downstream)[:15]
    print(f"\n{'root id':>20} {'Brian2':>8} {'ours':>8}")
    for i in top:
        print(f"{c.root_ids[i]:>20} {ref[i]:8.1f} {ours[i]:8.1f}")


if __name__ == "__main__":
    main()

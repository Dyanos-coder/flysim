"""Ground truth: run Shiu et al.'s exact Brian2 model (their model.py equations and
default parameters) on the v630 sugar experiment, and save per-neuron rates.

Needs brian2, which is not a project dependency:
    uv run --with brian2 --with pandas python -m scripts.brian2_reference [n_trials]
"""

import sys
import time
from textwrap import dedent

import numpy as np
from brian2 import Hz, Network, NeuronGroup, PoissonInput, SpikeMonitor, Synapses, mV, ms, prefs

from flysim import connectome
from scripts.validate_brain import SUGAR_R, V630

prefs.codegen.target = "numpy"

PARAMS = {
    "v_0": -52 * mV, "v_rst": -52 * mV, "v_th": -45 * mV, "t_mbr": 20 * ms,
    "tau": 5 * ms, "t_rfc": 2.2 * ms, "t_dly": 1.8 * ms, "w_syn": 0.275 * mV,
    "r_poi": 200 * Hz, "f_poi": 250,
}
EQS = dedent("""
    dv/dt = (v_0 - v + g) / t_mbr : volt (unless refractory)
    dg/dt = -g / tau               : volt (unless refractory)
    rfc                            : second
""")


def main():
    n_trials = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    c = connectome.load(V630 / "completeness_630.csv", V630 / "connectivity_630.parquet")
    sugar = c.index_of(SUGAR_R)
    total = np.zeros(c.n_neurons)
    for trial in range(n_trials):
        t0 = time.perf_counter()
        neu = NeuronGroup(c.n_neurons, EQS, method="linear", threshold="v > v_th",
                          reset="v = v_rst; g = 0 * mV", refractory="rfc", namespace=PARAMS)
        neu.v = PARAMS["v_0"]
        neu.g = 0 * mV
        neu.rfc = PARAMS["t_rfc"]
        syn = Synapses(neu, neu, "w : volt", on_pre="g += w", delay=PARAMS["t_dly"])
        syn.connect(i=c.pre, j=c.post)
        syn.w = c.signed_counts * PARAMS["w_syn"]
        pois = []
        for i in sugar:
            pois.append(PoissonInput(neu[int(i)], "v", 1, PARAMS["r_poi"],
                                     weight=PARAMS["w_syn"] * PARAMS["f_poi"]))
            neu[int(i)].rfc = 0 * ms
        mon = SpikeMonitor(neu)
        Network(neu, syn, mon, *pois).run(1000 * ms)
        total += np.bincount(np.asarray(mon.i), minlength=c.n_neurons)
        print(f"trial {trial}: {time.perf_counter() - t0:.0f} s, {mon.num_spikes} spikes", flush=True)
    np.save(V630 / "brian2_rates.npy", total / n_trials)


if __name__ == "__main__":
    main()

"""Headless end-to-end check of brain-driven behaviors in the physical world.

Run:  uv run python -m scripts.test_behaviors
"""

import time

import numpy as np

from flysim.brain_link import BrainLink
from flysim.perf import disable_windows_power_throttling
from flysim.world import World


def heading_deg(w: World) -> float:
    f = w._thorax_frame()[:, 0]
    return float(np.degrees(np.arctan2(f[1], f[0])))


def run(w: World, seconds: float, every: float = 0.25, label: str = ""):
    t_end = w.data.time + seconds
    next_print = w.data.time
    t0, s0 = time.perf_counter(), w.data.time
    while w.data.time < t_end:
        w.step()
        if w.data.time >= next_print:
            next_print += every
            r = w.brain.rates
            p = w.data.xpos[w.follow_body]
            b = w.behavior
            print(f"  t={w.data.time:5.2f} pos=({p[0]:6.1f},{p[1]:6.1f},{p[2]:5.2f}) head={heading_deg(w):6.1f}° "
                  f"up={w._thorax_frame()[2, 2]:+.2f} | GF {max(r['giant_fiber_left'], r['giant_fiber_right']):4.0f} "
                  f"DNa02 {r['dna02_left']:3.0f}/{r['dna02_right']:3.0f} prob {r['proboscis']:4.0f} groom {r['grooming']:4.0f} "
                  f"| {'FEED ' if b['feeding'] else ''}{'GROOM ' if b['grooming'] else ''}{'JUMP ' if b['escaping'] else ''}"
                  f"turn={b['turn']:+.2f} active={w.brain.n_active}")
    el = time.perf_counter() - t0
    print(f"  [{label}] {w.data.time - s0:.2f} s simulated in {el:.1f} s (x{(w.data.time - s0) / el:.2f}), brain load {w.brain.load:.2f}")


def main():
    disable_windows_power_throttling()
    brain = BrainLink()
    brain.start()
    print(f"brain loaded in {brain.load_seconds:.0f} s")
    w = World(brain=brain)

    print("\n1) sugar drop under the proboscis")
    tip = w.data.geom_xpos[w._haustellum_geom]
    w.add_item("sugar", tip[0], tip[1])
    run(w, 1.0, label="sugar")
    w.clear_items()
    run(w, 0.5, label="sugar removed")

    print("\n2) bitter drop under the proboscis")
    w.reset()
    tip = w.data.geom_xpos[w._haustellum_geom]
    w.add_item("bitter", tip[0], tip[1])
    run(w, 1.0, label="bitter")

    print("\n3) looming threat")
    w.reset()
    w.launch_threat(from_left=True)
    run(w, 1.5, every=0.1, label="threat")

    print("\n4) looming threat from the right")
    w.reset()
    w.launch_threat(from_left=False)
    run(w, 1.2, every=0.1, label="threat right")

    print("\n5) holding the head (touch)")
    w.reset()
    g = w._eye_geom["left"]
    w.set_push(g, np.zeros(3), w.data.geom_xpos[g].copy())
    run(w, 1.0, label="touch head")
    w.clear_push()


if __name__ == "__main__":
    main()

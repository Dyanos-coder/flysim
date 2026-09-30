"""Headless end-to-end check of brain-driven behaviors in the physical world.

Run:  uv run python -m scripts.test_behaviors
"""

import time

import numpy as np

from flysim.brain_link import BrainLink, BrainModel
from flysim.perf import disable_windows_power_throttling
from flysim.world import World


def run(w: World, seconds: float, every: float = 0.25, label: str = ""):
    t_end, next_print = w.time + seconds, w.time
    t0, s0 = time.perf_counter(), w.time
    while w.time < t_end:
        w.step_chunk()
        if w.time >= next_print:
            next_print += every
            parts = []
            for f in w.flies:
                r, b, p = f.brain.rates, f.behavior, f.position
                flags = " ".join(k for k in ("seeking_odor", "feeding", "grooming", "escaping", "flying") if b[k])
                parts.append(f"{f.name} ({p[0]:5.1f},{p[1]:5.1f},{p[2]:4.1f}) GF {max(r['giant_fiber_left'], r['giant_fiber_right']):3.0f} "
                             f"prob {r['proboscis']:3.0f} groom {r['grooming']:3.0f} [{flags}]")
            items = ", ".join(f"{i.kind}:{i.amount:.2f}" for i in w.items)
            print(f"  t={w.time:5.2f} " + " | ".join(parts) + (f"  items[{items}]" if items else ""))
    el = time.perf_counter() - t0
    print(f"  [{label}] {w.time - s0:.2f} s simulated in {el:.1f} s (x{(w.time - s0) / el:.2f}), recoveries {w.recoveries}")


def main():
    disable_windows_power_throttling()
    model = BrainModel()
    print(f"brain loaded in {model.load_seconds:.0f} s")

    brain = BrainLink(model)
    brain.start()
    w = World(n_flies=1, brains=[brain])
    fly = w.flies[0]

    print("\n1) sugar drop under the proboscis")
    tip = fly.proboscis_tip()
    w.add_item("sugar", tip[0], tip[1])
    run(w, 1.0, label="sugar")
    w.clear_items()

    print("\n2) bitter drop under the proboscis")
    w.reset()
    tip = fly.proboscis_tip()
    w.add_item("bitter", tip[0], tip[1])
    run(w, 1.0, label="bitter")
    w.clear_items()

    print("\n3) looming threat -> takeoff")
    w.reset()
    w.launch_threat(fly, from_left=True)
    run(w, 2.0, every=0.2, label="threat")

    print("\n4) holding the head (touch -> grooming)")
    w.reset()
    g = fly._eye_geom["left"]
    fly.set_push(g, np.zeros(3), fly.data.geom_xpos[g].copy())
    run(w, 1.0, label="touch head")
    fly.clear_push()

    print("\n5) three flies, sugar in the middle")
    brains = [brain] + [BrainLink(model, seed=i) for i in (1, 2)]
    for b in brains[1:]:
        b.start()
    w = World(n_flies=3, brains=brains)
    w.add_item("sugar", 0.0, 0.0)
    run(w, 6.0, every=1.0, label="colony")


if __name__ == "__main__":
    main()

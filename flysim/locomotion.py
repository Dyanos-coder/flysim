"""Fast CPG walking controller.

Same model as flygym_demo's HybridTurningController (tripod CPG network driving
preprogrammed single-leg steps recorded from a real fly, modulated by a
two-sided descending signal), but vectorized: the step splines are sampled once
into a lookup table, so one controller tick costs tens of microseconds instead
of milliseconds. The retraction/stumbling reflexes are omitted: they only matter
on rough terrain.
"""

import numpy as np
from flygym.anatomy import LEGS, JointDOF
from flygym_demo.complex_terrain import (
    PreprogrammedSteps,
    dof_spec_to_jointdof,
    make_tripod_cpg_network,
)

N_PHASE_BINS = 1024
# Adhesion stays off a little past the end of swing so the foot can land first.
SWING_EXTENSION = np.pi / 4


class WalkingController:
    def __init__(self, timestep: float, output_dof_order: list[JointDOF], seed: int = 0):
        self.cpg = make_tripod_cpg_network(timestep, seed=seed)
        self._base_freqs = self.cpg.intrinsic_freqs.copy()
        self._seed = seed

        steps = PreprogrammedSteps()
        grid = np.linspace(0, 2 * np.pi, N_PHASE_BINS + 1)
        # (6 legs, 7 dofs, bins) sampled step trajectories and (6, 7) neutral pose.
        self._table = np.stack([steps._psi_funcs[leg](grid) for leg in LEGS])
        self._neutral = np.stack([steps.neutral_pos[leg][:, 0] for leg in LEGS])

        swing = np.array([steps.swing_period[leg] for leg in LEGS])
        self._swing_start = swing[:, 0]
        self._swing_end = swing[:, 1] + SWING_EXTENSION

        lookup = {
            dof_spec_to_jointdof(leg, spec): (leg_i, dof_i)
            for leg_i, leg in enumerate(LEGS)
            for dof_i, spec in enumerate(steps.dofs_per_leg)
        }
        idx = np.array([lookup[dof] for dof in output_dof_order])
        self._out_leg, self._out_dof = idx[:, 0], idx[:, 1]

    def reset(self):
        self.cpg.random_state = np.random.RandomState(self._seed)
        self.cpg.intrinsic_freqs = self._base_freqs.copy()
        self.cpg.reset()

    def step(self, descending: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Advance one tick. `descending` = (left, right) drive; returns
        (joint angle targets in output order, per-leg adhesion on/off)."""
        self.cpg.intrinsic_amps = np.repeat(np.abs(descending), 3)
        self.cpg.intrinsic_freqs = self._base_freqs * np.repeat(np.sign(descending) + (descending == 0), 3)
        self.cpg.step()

        phase = self.cpg.curr_phases % (2 * np.pi)
        pos = phase / (2 * np.pi) * N_PHASE_BINS
        i0 = pos.astype(int)
        frac = (pos - i0)[:, None]
        legs = np.arange(6)
        step_angles = self._table[legs, :, i0] * (1 - frac) + self._table[legs, :, i0 + 1] * frac
        angles = self._neutral + self.cpg.curr_magnitudes[:, None] * (step_angles - self._neutral)

        in_swing = (self._swing_start < phase) & (phase < self._swing_end)
        return angles[self._out_leg, self._out_dof], ~in_swing

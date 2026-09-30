"""One fly in the shared world: its body landmarks, walking controller, senses,
brain link and behavior.

Every SENSE period the fly:
1. turns its surroundings into sensory drive (Poisson rates on groups of sensory
   neurons: taste, smell, looming vision -- balls and other flies --, touch);
2. reads smoothed firing rates of descending / motor neurons back from its brain
   and turns them into behavior: escape takeoff and turning away (giant fiber,
   DNa02/DNa01), feeding arrest and proboscis extension (proboscis motor
   neurons), grooming (bristle-driven DNs).
The neurons decide *whether* and *which way*; the motor programs that carry out
a takeoff, a flight or a grooming bout are scripted, because the brain model
stops at the neck (the ventral nerve cord that patterns movements is not in
FlyWire). Two behaviors are scripted reflexes outside the brain model, and
labeled as such in the UI: odor-guided walking (no descending neuron in the
model encodes odor side) and flight itself (no flight controller).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import mujoco
import numpy as np
from flygym import Simulation
from flygym.compose import ActuatorType, FlatGroundWorld
from flygym.utils.math import Rotation3D
from flygym_demo.complex_terrain import make_locomotion_fly

from flysim.locomotion import WalkingController

if TYPE_CHECKING:
    from flysim.brain_link import BrainLink
    from flysim.world import Item, World

# Physics step. 0.2 ms walks, grips and flies exactly like FlyGym's default 0.1 ms
# (same gait, no instabilities in stress tests) at half the cost.
TIMESTEP = 2e-4
# Walking controller period (physics steps): 1 ms, far faster than the ~12 Hz gait.
CONTROL_EVERY = 5
# Brain <-> body exchange period (physics steps): 10 ms. Each fly's physics runs
# this many steps on its own thread between exchanges.
SENSE_EVERY = 50
WARMUP_S = 0.05
# mm/s. Walking is ~15, a ball knock ~150; beyond this the physics has diverged.
MAX_GROUND_SPEED = 800.0

# Push-tool spring: pulling the grabbed point 1 mm away from the cursor produces
# PUSH_STIFFNESS_BW times the fly's body weight, capped at PUSH_MAX_BW and damped
# (seconds of the grabbed point's velocity). Stronger pulls accelerate the body
# faster than the tiny leg segments can follow and the physics diverges.
PUSH_STIFFNESS_BW = 6.0
PUSH_MAX_BW = 4.0
PUSH_DAMPING_S = 0.1

# --- food and smell
DROP_RADIUS = 0.8  # mm, sugar/bitter drops on the floor
TASTE_HZ = 150.0  # labellar gustatory neurons when the proboscis touches a drop
# Leg gustatory neurons project to the ventral nerve cord, which the brain model
# lacks; a foot in a drop is fed to the labellar neurons at a lower rate instead.
LEG_TASTE_HZ = 60.0
FEED_SECONDS = 6.0  # a drop is used up after this much feeding (by one fly)
ODOR_SIGMA = 35.0  # mm, width of the odor plume around a source
# Relative odor strength: a sugary drop stands for fruit juice, which smells
# (fermentation volatiles activate the same vinegar-sensitive receptor neurons).
ODOR_STRENGTH = {"vinegar": 1.0, "sugar": 0.6}
ODOR_MAX_HZ = 100.0
ODOR_DETECT = 0.03  # concentration that makes the fly go look for the source
ODOR_ARRIVED_MM = 2.0

# --- vision
LOOM_GAIN = 40.0  # Hz of LPLC2 drive per rad/s of angular expansion
LOOM_MAX_HZ = 150.0
FLY_VISUAL_RADIUS = 1.3  # mm, how big another fly looks
# Only flying neighbors loom: one walking up at ~13 mm/s expands on the retina
# about as fast as a real threat, and even scaled down to 15 % it set off
# startle cascades that kept flies from ever sharing food.
TOUCH_HZ = 50.0

# --- behavior readouts (Hz, smoothed rates)
TURN_NORM_HZ = 40.0
TURN_GAIN = 0.6
# DNa02 steering was validated for looming (it flips with the threat side), so
# brain steering is applied while/just after something looms.
LOOM_STEER_WINDOW_S = 1.0
ODOR_TURN_GAIN = 2.0
GIANT_FIBER_HZ = 40.0
ESCAPE_COOLDOWN_S = 1.5
FEED_ON_HZ, FEED_OFF_HZ = 20.0, 10.0
GROOM_ON_HZ, GROOM_OFF_HZ = 25.0, 12.0

# --- flight (scripted)
FLIGHT_DURATION_S = 1.4
FLIGHT_DISTANCE_MM = 55.0
FLIGHT_ALTITUDE_MM = 14.0
WINGBEAT_VISUAL_HZ = 14.0  # drawn wingbeat; the real ~200 Hz can't be shown at 30 fps

HEAD_PARTS = ("head", "eye", "rostrum", "haustellum", "pedicel", "funiculus", "arista")


@dataclass
class Flight:
    start: np.ndarray
    direction: np.ndarray  # unit, horizontal
    t0: float
    yaw: float


def odor_at(p: np.ndarray, sources: list[Item]) -> tuple[float, np.ndarray]:
    """Concentration (0..1) and its horizontal gradient at point p."""
    conc, grad = 0.0, np.zeros(2)
    for s in sources:
        d = p[:2] - s.pos[:2]
        c = ODOR_STRENGTH[s.kind] * np.exp(-(d @ d) / (2 * ODOR_SIGMA**2))
        conc += c
        grad += -c * d / ODOR_SIGMA**2
    return min(conc, 1.0), grad


def _rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle) * k + (1 - np.cos(angle)) * k @ k


class Fly:
    """Each fly has its own MuJoCo simulation (same world frame, own ground
    plane): flies never collide through MuJoCo anyway, and separate simulations
    step in parallel on separate cores (MuJoCo releases the GIL)."""

    def __init__(self, world: World, index: int, name: str, spawn_pos, spawn_yaw: float,
                 brain: BrainLink | None):
        self.world = world
        self.index = index
        self.name = name
        self.brain = brain
        arena = FlatGroundWorld(half_size=200)
        flygym_fly = make_locomotion_fly(name, colorize=True)
        arena.add_fly(flygym_fly, spawn_pos,
                      Rotation3D("quat", (np.cos(spawn_yaw / 2), 0, 0, np.sin(spawn_yaw / 2))),
                      # FlyGym's contact sensors aren't prefixed per fly and
                      # clash between flies; we don't use them.
                      add_ground_contact_sensors=False)
        self.sim = Simulation(arena, timestep=TIMESTEP)
        model = self.sim.mj_model
        self.model, self.data = model, self.sim.mj_data

        def body(part):
            return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{name}/{part}")

        def geom(part):
            return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"{name}/{part}")

        self.thorax = body("c_thorax")
        self._rostrum_body, self._haustellum_body = body("c_rostrum"), body("c_haustellum")
        self._rostrum_geom, self._haustellum_geom = geom("c_rostrum"), geom("c_haustellum")
        self._wing = {"left": (body("l_wing"), geom("l_wing")), "right": (body("r_wing"), geom("r_wing"))}
        self._antenna_geom = {"left": geom("l_funiculus"), "right": geom("r_funiculus")}
        self._eye_geom = {"left": geom("l_eye"), "right": geom("r_eye")}
        self._tarsus_geoms = [geom(f"{leg}_tarsus5") for leg in ("lf", "lm", "lh", "rf", "rm", "rh")]
        # Every geom of this fly (to route clicks and pushes).
        prefix = f"{name}/"
        self.geom_ids = {
            g for g in range(model.ngeom)
            if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or "").startswith(prefix)
        }
        joint = model.body_jntadr[self.thorax]
        self._free_qpos = model.jnt_qposadr[joint]
        self._free_qvel = model.jnt_dofadr[joint]

        self.mass = model.body_subtreemass[self.thorax]
        self.body_weight = self.mass * np.linalg.norm(model.opt.gravity)
        self._push_k = PUSH_STIFFNESS_BW * self.body_weight
        self._push_max = PUSH_MAX_BW * self.body_weight

        self.controller = WalkingController(
            timestep=model.opt.timestep * CONTROL_EVERY,
            output_dof_order=flygym_fly.get_actuated_jointdofs_order(ActuatorType.POSITION),
            seed=index,
        )
        self.walking = False
        self.reset()

    def reset(self):
        self.sim.reset()
        self.data.xfrc_applied[:] = 0
        self.controller.reset()
        self.push = None  # (body_id, geom_id, local_point, target)
        # Descending drive (left, right). 1 = normal forward walking; asymmetry
        # turns, negative values walk backwards on that side.
        self.descending = np.zeros(2)
        self.behavior = {
            "feeding": False, "grooming": False, "escaping": False, "flying": False,
            "seeking_odor": False, "turn": 0.0,
        }
        self.drive: dict[str, float] = {}
        self.proboscis = 0.0  # extension 0..1 (visual)
        self._last_escape = -np.inf
        self._last_loom = -np.inf
        self._odor_turn = 0.0
        self._groom_phase = 0.0
        self._theta_prev: dict = {}
        self.flight: Flight | None = None
        self._landed_at = -np.inf
        self.extra_force = np.zeros(3)  # e.g. bumping into other flies
        self.sim.warmup(WARMUP_S)
        self._snapshot()
        self.recoveries = 0
        self._steps = 0
        if self.brain is not None:
            self.brain.reset()

    # ------------------------------------------------------------ physics chunk

    def run_chunk(self, n_steps: int):
        """Advance this fly's physics n_steps (runs on a worker thread)."""
        for _ in range(n_steps):
            if self._steps % CONTROL_EVERY == 0:
                self.control()
            self.data.xfrc_applied[:] = 0
            self.apply_forces()
            t_before = self.data.time
            self.sim.step()
            self._steps += 1
            if self.data.time < t_before:  # MuJoCo auto-reset after a divergence
                self._recover()
        # A divergence that stays finite shows up as an absurd body speed.
        qv = self._free_qvel
        if self.flight is None and np.abs(self.data.qvel[qv : qv + 3]).max() > MAX_GROUND_SPEED:
            self._recover()
        else:
            self._snapshot()

    def _snapshot(self):
        d = self.data
        self._good = (d.time, d.qpos.copy(), d.act.copy())

    def _recover(self):
        """The physics blew up (usually huge contact/adhesion forces) and MuJoCo
        reset this fly to its spawn state, which looks like teleporting. Restore
        the last good state, at rest, and drop whatever was forcing the body."""
        t, qpos, act = self._good
        d = self.data
        d.time = t
        d.qpos[:] = qpos
        d.qvel[:] = 0.0
        d.act[:] = act
        d.qacc_warmstart[:] = 0.0
        d.xfrc_applied[:] = 0.0
        self.push = None
        self.flight = None
        mujoco.mj_forward(self.model, d)
        self.recoveries += 1

    # ------------------------------------------------------------ geometry helpers

    @property
    def position(self) -> np.ndarray:
        return self.data.xpos[self.thorax]

    def frame(self) -> np.ndarray:
        """Columns: fly forward, left, up, in world coordinates."""
        return self.data.xmat[self.thorax].reshape(3, 3)

    def head(self) -> np.ndarray:
        return 0.5 * (self.data.geom_xpos[self._eye_geom["left"]] + self.data.geom_xpos[self._eye_geom["right"]])

    def bearing_sin(self, point: np.ndarray) -> float:
        """sin of the azimuth of `point` in the fly frame (+1 = straight left)."""
        rel = self.frame().T @ (point - self.position)
        return float(rel[1] / (np.hypot(rel[0], rel[1]) + 1e-9))

    def proboscis_tip(self) -> np.ndarray:
        return self.data.geom_xpos[self._haustellum_geom]

    def feet(self) -> np.ndarray:
        return self.data.geom_xpos[self._tarsus_geoms]

    def touching(self, item: Item) -> bool:
        tip, feet = self.proboscis_tip(), self.feet()
        return bool(
            (np.hypot(*(tip[:2] - item.pos[:2])) < DROP_RADIUS + 0.3 and tip[2] < 0.8)
            or (np.hypot(*(feet[:, :2] - item.pos[:2]).T) < DROP_RADIUS).any()
        )

    # ------------------------------------------------------------ interaction

    def set_push(self, geom_id: int, local_point: np.ndarray, target: np.ndarray):
        self.push = (int(self.model.geom_bodyid[geom_id]), geom_id, local_point, target)

    def clear_push(self):
        self.push = None

    def take_off(self, direction: np.ndarray | None = None):
        """Scripted flight: lift off, fly FLIGHT_DISTANCE_MM, land."""
        if self.flight is not None:
            return
        if direction is None:
            direction = self.frame()[:, 0].copy()
        direction = np.array([direction[0], direction[1], 0.0])
        direction /= np.linalg.norm(direction) + 1e-9
        start = self.data.qpos[self._free_qpos : self._free_qpos + 3].copy()
        self.flight = Flight(start=start, direction=direction, t0=self.data.time,
                             yaw=float(np.arctan2(direction[1], direction[0])))

    # ------------------------------------------------------------ senses

    def _loom(self, key, center: np.ndarray, radius: float, dt: float, add, scale: float = 1.0):
        """LPLC2 drive from an object's angular expansion, per eye."""
        for side, sign in (("left", 1.0), ("right", -1.0)):
            eye = self.data.geom_xpos[self._eye_geom[side]]
            d = max(np.linalg.norm(center - eye), radius * 1.01)
            theta = 2 * np.arctan(radius / d)
            prev = self._theta_prev.get((key, side), theta)
            self._theta_prev[(key, side)] = theta
            expansion = max(0.0, (theta - prev) / dt)
            field_weight = np.clip(0.5 + sign * self.bearing_sin(center), 0.0, 1.0)
            add(f"looming_{side}", min(LOOM_MAX_HZ, LOOM_GAIN * expansion * scale) * field_weight)

    def sense(self, dt: float) -> dict[str, float]:
        drive: dict[str, float] = {}

        def add(name, hz):
            drive[name] = max(drive.get(name, 0.0), float(hz))

        items = self.world.items
        tip, feet = self.proboscis_tip(), self.feet()
        for item in items:
            if item.kind in ("sugar", "bitter"):
                if np.hypot(*(tip[:2] - item.pos[:2])) < DROP_RADIUS + 0.3 and tip[2] < 0.8:
                    add(item.kind, TASTE_HZ)
                if (np.hypot(*(feet[:, :2] - item.pos[:2]).T) < DROP_RADIUS).any():
                    add(item.kind, LEG_TASTE_HZ)

        # Smell: each antenna's olfactory receptor neurons fire with the odor
        # concentration where that antenna is.
        sources = [i for i in items if i.kind in ODOR_STRENGTH]
        if sources:
            for side in ("left", "right"):
                conc, _ = odor_at(self.data.geom_xpos[self._antenna_geom[side]], sources)
                add(f"odor_vinegar_{side}", ODOR_MAX_HZ * conc)

        # Vision: thrown balls and the other flies loom when they come at us.
        for pr in self.world.projectiles:
            self._loom(id(pr), pr.pos, pr.radius, dt, add)
        for other in self.world.flies:
            if other is not self:
                if other.flight is not None:
                    self._loom(other.name, other.position, FLY_VISUAL_RADIUS, dt, add)
                else:
                    self._theta_prev.pop((other.name, "left"), None)
                    self._theta_prev.pop((other.name, "right"), None)

        if self.push is not None:
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, self.push[1]) or ""
            part = name.rsplit("/", 1)[-1]
            group = "head_bristle" if any(p in part for p in HEAD_PARTS) else "touch"
            sides = {"l": ["left"], "r": ["right"]}.get(part[:1], ["left", "right"])
            for side in sides:
                add(f"{group}_{side}", TOUCH_HZ)
        drive = {k: v for k, v in drive.items() if v > 0.5}
        if any(k.startswith("looming") for k in drive):
            self._last_loom = self.data.time
        return drive

    def odor_reflex(self):
        """Scripted olfactory search (not from the brain model): walk up the odor
        gradient until close to the source."""
        b = self.behavior
        sources = [i for i in self.world.items if i.kind in ODOR_STRENGTH]
        head, tip = self.head(), self.proboscis_tip()
        conc, grad = odor_at(head, sources) if sources else (0.0, np.zeros(2))
        # Food: keep going until the proboscis is on the drop (taste then stops
        # the fly through the brain). Pure odor sources: stop close by.
        near = any(
            np.hypot(*(tip[:2] - s.pos[:2])) < DROP_RADIUS * 0.5 if s.kind == "sugar"
            else np.hypot(*(head[:2] - s.pos[:2])) < ODOR_ARRIVED_MM
            for s in sources
        )
        b["seeking_odor"] = conc > ODOR_DETECT and not near
        if b["seeking_odor"] and np.linalg.norm(grad) > 1e-9:
            fwd = self.frame()[:2, 0]
            g = grad / np.linalg.norm(grad)
            cross = fwd[0] * g[1] - fwd[1] * g[0]  # >0: source to the left
            steer = cross if fwd @ g > 0 else (np.sign(cross) or 1.0)
            self._odor_turn = float(np.clip(ODOR_TURN_GAIN * steer, -1, 1))
        else:
            self._odor_turn = 0.0

    # ------------------------------------------------------------ brain -> behavior

    def act(self, rates: dict[str, float]):
        b = self.behavior
        t = self.data.time

        left = rates["dna02_left"] + 0.5 * rates["dna01_left"]
        right = rates["dna02_right"] + 0.5 * rates["dna01_right"]
        brain_turn = float(np.clip((left - right) / TURN_NORM_HZ, -1.0, 1.0))

        if rates["proboscis"] > FEED_ON_HZ:
            b["feeding"] = True
        elif rates["proboscis"] < FEED_OFF_HZ:
            b["feeding"] = False

        if rates["grooming"] > GROOM_ON_HZ:
            b["grooming"] = True
        elif rates["grooming"] < GROOM_OFF_HZ:
            b["grooming"] = False

        gf = max(rates["giant_fiber_left"], rates["giant_fiber_right"])
        if gf > GIANT_FIBER_HZ and t - self._last_escape > ESCAPE_COOLDOWN_S and self.flight is None:
            self._escape()
        b["escaping"] = t - self._last_escape < 0.5

        b["turn"] = brain_turn if t - self._last_loom < LOOM_STEER_WINDOW_S else self._odor_turn
        # A feeding or grooming fly stays put: no steering.
        if b["feeding"] or b["grooming"]:
            b["turn"] = 0.0

    def _escape(self):
        """Giant-fiber escape: take off away from the nearest looming object."""
        self._last_escape = self.data.time
        me = self.position
        away = -self.frame()[:, 0]
        threats = [p.pos for p in self.world.projectiles]
        threats += [o.position for o in self.world.flies if o is not self and o.flight is not None]
        if threats:
            nearest = min(threats, key=lambda p: np.linalg.norm(p - me))
            v = me - nearest
            v[2] = 0
            if np.linalg.norm(v) > 1e-6:
                away = v / np.linalg.norm(v)
        self.take_off(away)

    # ------------------------------------------------------------ per-step control

    def control(self):
        """Joint targets and adhesion (every CONTROL_EVERY physics steps)."""
        b = self.behavior
        busy = b["feeding"] or b["grooming"] or self.flight is not None
        wants_to_walk = self.walking or b["seeking_odor"]
        walk = 1.0 if wants_to_walk and not busy else 0.0
        turn = 0.0 if self.flight is not None else b["turn"]
        target = np.array([walk - TURN_GAIN * turn, walk + TURN_GAIN * turn])
        # Ease the drive in/out so starting and stopping look natural.
        self.descending += 0.04 * (target - self.descending)
        angles, adhesion = self.controller.step(self.descending)
        if np.abs(self.descending).max() < 0.05:
            # Standing still: keep all feet stuck to the ground.
            adhesion[:] = True
        if b["grooming"] and self.flight is None:
            angles = self._grooming_pose(angles)
            adhesion[[0, 3]] = False
        if self.flight is not None or self.push is not None or self.extra_force.any():
            # Airborne, held by the user or jostled by another fly: feet let go
            # (pushing against full adhesion otherwise tears the leg joints apart).
            adhesion[:] = False
        self.sim.set_actuator_inputs(self.name, ActuatorType.POSITION, angles)
        self.sim.set_leg_adhesion_states(self.name, adhesion.astype(float))
        ext_target = 1.0 if b["feeding"] else 0.0
        self.proboscis += 0.02 * (ext_target - self.proboscis)

    def apply_forces(self):
        """External forces for this physics step (xfrc_applied is zeroed before)."""
        if self.push is not None and self.flight is None:
            self._apply_push()
        if self.flight is not None:
            self._fly()
        elif self.extra_force.any():
            self.data.xfrc_applied[self.thorax, :3] += self.extra_force
        self._stabilize()

    def _fly(self):
        """Move the body along the scripted flight path; land at the end."""
        f = self.flight
        u = (self.data.time - f.t0) / FLIGHT_DURATION_S
        qp, qv = self._free_qpos, self._free_qvel
        if u >= 1.0:
            self.flight = None
            self._landed_at = self.data.time
            self.data.qvel[qv : qv + 6] = 0.0
            return
        ease = 0.5 - 0.5 * np.cos(np.pi * u)  # 0 -> 1, slow at both ends
        pos = f.start + f.direction * FLIGHT_DISTANCE_MM * ease
        pos[2] = f.start[2] + FLIGHT_ALTITUDE_MM * np.sin(np.pi * u)
        speed = FLIGHT_DISTANCE_MM * 0.5 * np.pi * np.sin(np.pi * u) / FLIGHT_DURATION_S
        climb = FLIGHT_ALTITUDE_MM * np.pi * np.cos(np.pi * u) / FLIGHT_DURATION_S
        self.data.qpos[qp : qp + 3] = pos
        self.data.qpos[qp + 3 : qp + 7] = [np.cos(f.yaw / 2), 0.0, 0.0, np.sin(f.yaw / 2)]
        self.data.qvel[qv : qv + 3] = np.array([*(f.direction[:2] * speed), climb])
        self.data.qvel[qv + 3 : qv + 6] = 0.0

    def _stabilize(self):
        """Just after landing, or when tipped over, halteres-like attitude
        control keeps the body upright."""
        xmat = self.data.xmat[self.thorax]
        if self.data.time - self._landed_at < 0.5 or xmat[8] < 0.5:
            # tilt = up x world_z, with up = third column of the body frame.
            tilt = np.array([xmat[5], -xmat[2], 0.0])
            if xmat[8] < 0.0:
                # Upside down the cross product vanishes; roll over sideways.
                n = np.linalg.norm(tilt)
                tilt = tilt / n if n > 1e-3 else self.frame()[:, 0].copy()
            qv = self._free_qvel
            omega = self.data.qvel[qv + 3 : qv + 6]
            k = self.mass * 4000.0
            self.data.xfrc_applied[self.thorax, 3:] += k * tilt - k * 0.02 * omega

    def _grooming_pose(self, angles: np.ndarray) -> np.ndarray:
        """Front legs sweep forward over the head at ~6 Hz (scripted program)."""
        self._groom_phase += 2 * np.pi * 6.0 * self.controller.cpg.timestep
        steps = self.controller
        swing_mid = np.pi * 0.35
        phase = swing_mid + 0.9 * np.sin(self._groom_phase)
        out = angles.copy()
        for leg_idx in (0, 3):
            pos = phase / (2 * np.pi) * len(steps._table[0, 0])
            i0 = int(pos) % (steps._table.shape[2] - 1)
            leg_angles = steps._neutral[leg_idx] + 1.6 * (steps._table[leg_idx, :, i0] - steps._neutral[leg_idx])
            sel = steps._out_leg == leg_idx
            out[sel] = leg_angles[steps._out_dof[sel]]
        return out

    def _apply_push(self):
        body_id, geom_id, local, target = self.push
        xmat = self.data.geom_xmat[geom_id].reshape(3, 3)
        anchor = self.data.geom_xpos[geom_id] + xmat @ local
        vel = np.zeros(6)
        mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_GEOM, geom_id, vel, 0)
        force = self._push_k * (target - anchor - PUSH_DAMPING_S * vel[3:])
        norm = np.linalg.norm(force)
        if norm > self._push_max:
            force *= self._push_max / norm
        torque = np.cross(anchor - self.data.xipos[body_id], force)
        self.data.xfrc_applied[body_id, :3] += force
        self.data.xfrc_applied[body_id, 3:] += torque

    # ------------------------------------------------------------ rendering

    def visual_overrides(self, out: dict):
        """Drawn, not simulated: proboscis extension (rostrum and haustellum
        rotate about the head's left-right axis) and flapping wings in flight."""
        frame = self.frame()

        def turn(geom, pivot, r):
            pos, mat = out.get(geom, (self.data.geom_xpos[geom], self.data.geom_xmat[geom].reshape(3, 3)))
            out[geom] = (pivot + r @ (pos - pivot), r @ mat)

        if self.proboscis > 0.01:
            pivot1 = self.data.xpos[self._rostrum_body]
            r1 = _rotation(frame[:, 1], -1.4 * self.proboscis)
            turn(self._rostrum_geom, pivot1, r1)
            turn(self._haustellum_geom, pivot1, r1)
            # The haustellum unfolds a bit further around its own joint.
            pivot2 = pivot1 + r1 @ (self.data.xpos[self._haustellum_body] - pivot1)
            turn(self._haustellum_geom, pivot2, _rotation(frame[:, 1], -0.9 * self.proboscis))

        if self.flight is not None:
            u = (self.data.time - self.flight.t0) / FLIGHT_DURATION_S
            spread = np.radians(80.0) * min(1.0, 6 * u, 6 * (1 - u))
            beat = np.radians(45.0) * np.sin(2 * np.pi * WINGBEAT_VISUAL_HZ * (self.data.time - self.flight.t0))
            for side, sign in (("left", 1.0), ("right", -1.0)):
                body, geom = self._wing[side]
                hinge = self.data.xpos[body]
                # Swing the folded wing out sideways, then flap it up and down.
                turn(geom, hinge, _rotation(frame[:, 2], -sign * spread))
                turn(geom, hinge, _rotation(frame[:, 0], sign * beat))

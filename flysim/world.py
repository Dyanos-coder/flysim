"""The simulated world: one NeuroMechFly on flat ground, driven by a CPG walking
controller that takes a two-sided descending signal (left, right), and
optionally by the connectome brain.

With a brain attached, every SENSE_EVERY physics steps the world:
1. turns the fly's surroundings into sensory drive (Poisson rates on groups of
   sensory neurons: taste, smell, looming vision, touch);
2. reads smoothed firing rates of descending / motor neurons and turns them into
   behavior: escape takeoff and turning away (giant fiber, DNa02/DNa01),
   feeding arrest and proboscis extension (proboscis motor neurons), grooming
   (bristle-driven DNs).
The neurons decide *whether* and *which way*; the motor programs that carry out
a takeoff, a flight or a grooming bout are scripted, because the brain model
stops at the neck (the ventral nerve cord that patterns movements is not in
FlyWire). Two behaviors are scripted reflexes outside the brain model, and
labeled as such in the UI: odor-guided walking (no descending neuron in the
model encodes odor side) and flight itself (no flight controller).
"""

from dataclasses import dataclass, field
import itertools

import mujoco
import numpy as np
from flygym import Simulation
from flygym.compose import ActuatorType, FlatGroundWorld
from flygym.utils.math import Rotation3D
from flygym_demo.complex_terrain import make_locomotion_fly

from flysim.locomotion import WalkingController

FLY_NAME = "nmf"
SPAWN_POS = (0.0, 0.0, 0.8)  # mm
# The controller runs slower than the 0.1 ms physics step; joint targets are held
# in between. 0.5 ms is still far faster than the ~12 Hz stepping rhythm.
CONTROL_EVERY = 5
# Brain <-> body exchange period (physics steps of 0.1 ms): 10 ms.
SENSE_EVERY = 100
WARMUP_S = 0.05
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
FEED_SECONDS = 6.0  # a drop is used up after this much feeding
ODOR_SIGMA = 25.0  # mm, width of the odor plume around a source
# Relative odor strength: a sugary drop stands for fruit juice, which smells
# (fermentation volatiles activate the same vinegar-sensitive receptor neurons).
ODOR_STRENGTH = {"vinegar": 1.0, "sugar": 0.6}
ODOR_MAX_HZ = 100.0
ODOR_DETECT = 0.03  # concentration that makes the fly go look for the source
ODOR_ARRIVED_MM = 2.0

# --- looming and projectiles
THREAT_RADIUS = 3.0  # mm, the ball launched by the "Menace" button
THREAT_SPEED = 120.0  # mm/s
THREAT_START_DIST = 60.0  # mm
THROW_RADIUS = 1.6  # mm, balls thrown with the mouse
THROW_SPEED = 160.0  # mm/s
THROW_START_DIST = 45.0  # mm from the aimed point, back toward the camera
LOOM_GAIN = 40.0  # Hz of LPLC2 drive per rad/s of angular expansion
LOOM_MAX_HZ = 150.0
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
class Item:
    id: int
    kind: str  # "sugar" | "bitter" | "vinegar"
    pos: np.ndarray
    amount: float = 1.0  # sugar drops shrink as the fly eats


@dataclass
class Projectile:
    pos: np.ndarray
    vel: np.ndarray
    radius: float
    lifetime: float
    age: float = 0.0
    hit: bool = False
    theta_prev: dict = field(default_factory=dict)


@dataclass
class Flight:
    start: np.ndarray
    direction: np.ndarray  # unit, horizontal
    t0: float
    yaw: float


def _odor_at(p: np.ndarray, sources: list[Item]) -> tuple[float, np.ndarray]:
    """Concentration (0..1) and its horizontal gradient at point p."""
    conc, grad = 0.0, np.zeros(2)
    for s in sources:
        d = p[:2] - s.pos[:2]
        c = ODOR_STRENGTH[s.kind] * np.exp(-(d @ d) / (2 * ODOR_SIGMA**2))
        conc += c
        grad += -c * d / ODOR_SIGMA**2
    return min(conc, 1.0), grad


class World:
    def __init__(self, brain=None):
        fly = make_locomotion_fly(FLY_NAME, colorize=True)
        arena = FlatGroundWorld(half_size=200)
        arena.add_fly(fly, SPAWN_POS, Rotation3D("quat", (1, 0, 0, 0)))

        self.sim = Simulation(arena)
        self.model: mujoco.MjModel = self.sim.mj_model
        self.data: mujoco.MjData = self.sim.mj_data
        self.controller = WalkingController(
            timestep=self.model.opt.timestep * CONTROL_EVERY,
            output_dof_order=fly.get_actuated_jointdofs_order(ActuatorType.POSITION),
        )
        self.brain = brain

        def body(name):
            return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"{FLY_NAME}/{name}")

        def geom(name):
            return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"{FLY_NAME}/{name}")

        root = fly.bodyseg_to_mjcfbody[fly.root_segment].name
        self.follow_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, root)
        self._rostrum_body, self._haustellum_body = body("c_rostrum"), body("c_haustellum")
        self._rostrum_geom, self._haustellum_geom = geom("c_rostrum"), geom("c_haustellum")
        self._wing = {"left": (body("l_wing"), geom("l_wing")), "right": (body("r_wing"), geom("r_wing"))}
        self._antenna_geom = {"left": geom("l_funiculus"), "right": geom("r_funiculus")}
        self._eye_geom = {"left": geom("l_eye"), "right": geom("r_eye")}
        self._tarsus_geoms = [geom(f"{leg}_tarsus5") for leg in ("lf", "lm", "lh", "rf", "rm", "rh")]
        joint = self.model.body_jntadr[self.follow_body]
        self._free_qpos = self.model.jnt_qposadr[joint]
        self._free_qvel = self.model.jnt_dofadr[joint]

        fly_mass = self.model.body_subtreemass[self.follow_body]
        self._fly_mass = fly_mass
        body_weight = fly_mass * np.linalg.norm(self.model.opt.gravity)
        self._push_k = PUSH_STIFFNESS_BW * body_weight
        self._push_max = PUSH_MAX_BW * body_weight
        self._push = None  # (body_id, geom_id, local_point, target)

        self.walking = False
        self.items: list[Item] = []
        self._item_ids = itertools.count(1)
        self.projectiles: list[Projectile] = []
        self.reset()

    def reset(self):
        self.sim.reset()
        self.controller.reset()
        self._push = None
        self._steps = 0
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
        self.flight: Flight | None = None
        self._landed_at = -np.inf
        self.projectiles = []
        self.data.xfrc_applied[:] = 0
        self.sim.warmup(WARMUP_S)
        self._snapshot()
        self.recoveries = 0
        if self.brain is not None:
            self.brain.reset()

    def _snapshot(self):
        d = self.data
        self._good = (d.time, d.qpos.copy(), d.act.copy())

    def _recover(self):
        """The physics blew up (usually huge contact/adhesion forces) and MuJoCo
        reset the whole world to its initial state, which looks like the fly
        teleporting. Restore the last good state, at rest, and drop whatever
        was forcing the body."""
        t, qpos, act = self._good
        d = self.data
        d.time = t
        d.qpos[:] = qpos
        d.qvel[:] = 0.0
        d.act[:] = act
        d.qacc_warmstart[:] = 0.0
        d.xfrc_applied[:] = 0.0
        self._push = None
        self.flight = None
        mujoco.mj_forward(self.model, d)
        self.recoveries += 1

    # ------------------------------------------------------------ interaction

    def set_push(self, geom_id: int, local_point: np.ndarray, target: np.ndarray):
        body_id = int(self.model.geom_bodyid[geom_id])
        self._push = (body_id, geom_id, local_point, target)

    def clear_push(self):
        self._push = None

    def add_item(self, kind: str, x: float, y: float) -> Item:
        item = Item(next(self._item_ids), kind, np.array([x, y, 0.0]))
        self.items.append(item)
        return item

    def clear_items(self):
        self.items = []

    def _head(self) -> np.ndarray:
        return 0.5 * (self.data.geom_xpos[self._eye_geom["left"]] + self.data.geom_xpos[self._eye_geom["right"]])

    def launch_threat(self, from_left: bool | None = None):
        """A big dark ball flies at the fly's head from the front-left or front-right."""
        if from_left is None:
            from_left = bool(np.random.rand() < 0.5)
        azimuth = np.radians(55.0 if from_left else -55.0)
        direction_fly = np.array([np.cos(azimuth), np.sin(azimuth), 0.35])
        direction = self._thorax_frame() @ (direction_fly / np.linalg.norm(direction_fly))
        start = self._head() + direction * THREAT_START_DIST
        self.projectiles.append(Projectile(
            pos=start, vel=-direction * THREAT_SPEED, radius=THREAT_RADIUS,
            lifetime=THREAT_START_DIST / THREAT_SPEED + 0.8,
        ))

    def throw(self, camera: np.ndarray, target: np.ndarray):
        """A ball thrown from the viewer's side toward `target`."""
        direction = target - camera
        direction /= np.linalg.norm(direction) + 1e-9
        start = target - direction * THROW_START_DIST
        self.projectiles.append(Projectile(
            pos=start, vel=direction * THROW_SPEED, radius=THROW_RADIUS,
            lifetime=2 * THROW_START_DIST / THROW_SPEED,
        ))

    def take_off(self, direction: np.ndarray | None = None):
        """Scripted flight: lift off, fly FLIGHT_DISTANCE_MM, land."""
        if self.flight is not None:
            return
        if direction is None:
            direction = self._thorax_frame()[:, 0].copy()
        direction = np.array([direction[0], direction[1], 0.0])
        direction /= np.linalg.norm(direction) + 1e-9
        start = self.data.qpos[self._free_qpos : self._free_qpos + 3].copy()
        self.flight = Flight(start=start, direction=direction, t0=self.data.time,
                             yaw=float(np.arctan2(direction[1], direction[0])))

    # ------------------------------------------------------------ geometry helpers

    def _thorax_frame(self) -> np.ndarray:
        """Columns: fly forward, left, up, in world coordinates."""
        return self.data.xmat[self.follow_body].reshape(3, 3)

    def _bearing_sin(self, point: np.ndarray) -> float:
        """sin of the azimuth of `point` in the fly frame (+1 = straight left)."""
        rel = self._thorax_frame().T @ (point - self.data.xpos[self.follow_body])
        return float(rel[1] / (np.hypot(rel[0], rel[1]) + 1e-9))

    # ------------------------------------------------------------ senses

    def _sense(self, dt: float) -> dict[str, float]:
        drive: dict[str, float] = {}

        def add(name, hz):
            drive[name] = max(drive.get(name, 0.0), float(hz))

        tip = self.data.geom_xpos[self._haustellum_geom]
        feet = self.data.geom_xpos[self._tarsus_geoms]
        for item in self.items:
            if item.kind in ("sugar", "bitter"):
                if np.hypot(*(tip[:2] - item.pos[:2])) < DROP_RADIUS + 0.3 and tip[2] < 0.8:
                    add(item.kind, TASTE_HZ)
                if (np.hypot(*(feet[:, :2] - item.pos[:2]).T) < DROP_RADIUS).any():
                    add(item.kind, LEG_TASTE_HZ)

        # Smell: each antenna's olfactory receptor neurons fire with the odor
        # concentration where that antenna is.
        sources = [i for i in self.items if i.kind in ODOR_STRENGTH]
        if sources:
            for side in ("left", "right"):
                conc, _ = _odor_at(self.data.geom_xpos[self._antenna_geom[side]], sources)
                add(f"odor_vinegar_{side}", ODOR_MAX_HZ * conc)

        for pr in self.projectiles:
            for side, sign in (("left", 1.0), ("right", -1.0)):
                eye = self.data.geom_xpos[self._eye_geom[side]]
                d = max(np.linalg.norm(pr.pos - eye), pr.radius * 1.01)
                theta = 2 * np.arctan(pr.radius / d)
                prev = pr.theta_prev.get(side, theta)
                pr.theta_prev[side] = theta
                expansion = max(0.0, (theta - prev) / dt)
                field_weight = np.clip(0.5 + sign * self._bearing_sin(pr.pos), 0.0, 1.0)
                add(f"looming_{side}", min(LOOM_MAX_HZ, LOOM_GAIN * expansion) * field_weight)

        if self._push is not None:
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, self._push[1]) or ""
            part = name.rsplit("/", 1)[-1]
            group = "head_bristle" if any(p in part for p in HEAD_PARTS) else "touch"
            sides = {"l": ["left"], "r": ["right"]}.get(part[:1], ["left", "right"])
            for side in sides:
                add(f"{group}_{side}", TOUCH_HZ)
        drive = {k: v for k, v in drive.items() if v > 0.5}
        if any(k.startswith("looming") for k in drive):
            self._last_loom = self.data.time
        return drive

    def _odor_reflex(self):
        """Scripted olfactory search (not from the brain model): walk up the odor
        gradient until close to the source."""
        b = self.behavior
        sources = [i for i in self.items if i.kind in ODOR_STRENGTH]
        head = self._head()
        tip = self.data.geom_xpos[self._haustellum_geom]
        conc, grad = _odor_at(head, sources) if sources else (0.0, np.zeros(2))
        # Food: keep going until the proboscis is on the drop (taste then stops
        # the fly through the brain). Pure odor sources: stop close by.
        near = any(
            np.hypot(*(tip[:2] - s.pos[:2])) < DROP_RADIUS * 0.5 if s.kind == "sugar"
            else np.hypot(*(head[:2] - s.pos[:2])) < ODOR_ARRIVED_MM
            for s in sources
        )
        b["seeking_odor"] = conc > ODOR_DETECT and not near
        if b["seeking_odor"] and np.linalg.norm(grad) > 1e-9:
            fwd = self._thorax_frame()[:2, 0]
            g = grad / np.linalg.norm(grad)
            cross = fwd[0] * g[1] - fwd[1] * g[0]  # >0: source to the left
            self._odor_turn = float(np.clip(ODOR_TURN_GAIN * (cross if fwd @ g > 0 else np.sign(cross) or 1.0), -1, 1))
        else:
            self._odor_turn = 0.0

    def _eat(self, dt: float):
        if not self.behavior["feeding"]:
            return
        tip = self.data.geom_xpos[self._haustellum_geom]
        feet = self.data.geom_xpos[self._tarsus_geoms]
        for item in self.items:
            if item.kind != "sugar":
                continue
            touching = np.hypot(*(tip[:2] - item.pos[:2])) < DROP_RADIUS + 0.3 or \
                (np.hypot(*(feet[:, :2] - item.pos[:2]).T) < DROP_RADIUS).any()
            if touching:
                item.amount -= dt / FEED_SECONDS
        self.items = [i for i in self.items if i.amount > 0]

    # ------------------------------------------------------------ brain -> behavior

    def _act(self, rates: dict[str, float]):
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

        if t - self._last_loom < LOOM_STEER_WINDOW_S:
            b["turn"] = brain_turn
        else:
            b["turn"] = self._odor_turn
        # A feeding or grooming fly stays put: no steering.
        if b["feeding"] or b["grooming"]:
            b["turn"] = 0.0

    def _escape(self):
        """Giant-fiber escape: take off away from the nearest looming object."""
        self._last_escape = self.data.time
        away = -self._thorax_frame()[:, 0]
        if self.projectiles:
            me = self.data.xpos[self.follow_body]
            nearest = min(self.projectiles, key=lambda p: np.linalg.norm(p.pos - me))
            v = me - nearest.pos
            v[2] = 0
            if np.linalg.norm(v) > 1e-6:
                away = v / np.linalg.norm(v)
        self.take_off(away)

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
        xmat = self.data.xmat[self.follow_body]
        if self.data.time - self._landed_at < 0.5 or xmat[8] < 0.5:
            # tilt = up x world_z, with up = third column of the body frame.
            tilt = np.array([xmat[5], -xmat[2], 0.0])
            if xmat[8] < 0.0:
                # Upside down the cross product vanishes; roll over sideways.
                n = np.linalg.norm(tilt)
                tilt = tilt / n if n > 1e-3 else self._thorax_frame()[:, 0].copy()
            qv = self._free_qvel
            omega = self.data.qvel[qv + 3 : qv + 6]
            k = self._fly_mass * 4000.0
            self.data.xfrc_applied[self.follow_body, 3:] += k * tilt - k * 0.02 * omega

    # ------------------------------------------------------------ stepping

    def _update_projectiles(self, dt: float):
        me = self.data.xpos[self.follow_body]
        kept = []
        for pr in self.projectiles:
            pr.pos = pr.pos + pr.vel * dt
            pr.age += dt
            if not pr.hit and np.linalg.norm(pr.pos - me) < pr.radius + 1.2 and self.flight is None:
                # Knock the fly along the ball's path; the ball bounces off.
                pr.hit = True
                qv = self._free_qvel
                push = pr.vel / (np.linalg.norm(pr.vel) + 1e-9)
                self.data.qvel[qv : qv + 3] += push * 120.0 + np.array([0, 0, 60.0])
                pr.vel = -0.3 * pr.vel + np.array([0, 0, 40.0])
            if pr.age < pr.lifetime and pr.pos[2] > -pr.radius:
                kept.append(pr)
        self.projectiles = kept

    def step(self):
        dt_sense = SENSE_EVERY * self.model.opt.timestep
        if self._steps % SENSE_EVERY == 0:
            self._update_projectiles(dt_sense)
            self._odor_reflex()
            self._eat(dt_sense)
            if self.brain is not None:
                self.drive = self._sense(dt_sense)
                self.brain.set_drive(self.drive)
                self.brain.advance_to(self.data.time)
                self._act(self.brain.rates)
            else:
                self.behavior["turn"] = self._odor_turn
            self.behavior["flying"] = self.flight is not None

        if self._steps % CONTROL_EVERY == 0:
            b = self.behavior
            busy = b["feeding"] or b["grooming"] or self.flight is not None
            wants_to_walk = self.walking or b["seeking_odor"]
            walk = 1.0 if wants_to_walk and not busy else 0.0
            turn = 0.0 if self.flight is not None else b["turn"]
            target = np.array([walk - TURN_GAIN * turn, walk + TURN_GAIN * turn])
            # Ease the drive in/out so starting and stopping look natural.
            self.descending += 0.02 * (target - self.descending)
            angles, adhesion = self.controller.step(self.descending)
            if np.abs(self.descending).max() < 0.05:
                # Standing still: keep all feet stuck to the ground.
                adhesion[:] = True
            if b["grooming"] and self.flight is None:
                angles = self._grooming_pose(angles)
                adhesion[[0, 3]] = False
            if self.flight is not None or self._push is not None:
                # Airborne, or held by the user: feet let go (pulling against
                # full adhesion otherwise tears the leg joints apart).
                adhesion[:] = False
            self.sim.set_actuator_inputs(FLY_NAME, ActuatorType.POSITION, angles)
            self.sim.set_leg_adhesion_states(FLY_NAME, adhesion.astype(float))
            ext_target = 1.0 if b["feeding"] else 0.0
            self.proboscis += 0.01 * (ext_target - self.proboscis)

        self.data.xfrc_applied[:] = 0
        if self._push is not None and self.flight is None:
            self._apply_push()
        if self.flight is not None:
            self._fly()
        self._stabilize()
        t_before = self.data.time
        self.sim.step()
        self._steps += 1
        if self.data.time < t_before:  # MuJoCo auto-reset after a divergence
            self._recover()
        elif self._steps % SENSE_EVERY == 0:
            self._snapshot()

    def _grooming_pose(self, angles: np.ndarray) -> np.ndarray:
        """Front legs sweep forward over the head at ~6 Hz (scripted program)."""
        self._groom_phase += 2 * np.pi * 6.0 * CONTROL_EVERY * self.model.opt.timestep
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
        body_id, geom_id, local, target = self._push
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

    def projectile_list(self) -> list[tuple[float, float, float, float]]:
        return [(*p.pos, p.radius) for p in self.projectiles]

    def visual_overrides(self) -> dict[int, tuple[np.ndarray, np.ndarray]]:
        """Drawn, not simulated: proboscis extension (rostrum and haustellum
        rotate about the head's left-right axis) and flapping wings in flight."""
        frame = self._thorax_frame()
        out = {}

        def rot(axis, angle):
            k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
            return np.eye(3) + np.sin(angle) * k + (1 - np.cos(angle)) * k @ k

        def turn(geom, pivot, r):
            pos, mat = out.get(geom, (self.data.geom_xpos[geom], self.data.geom_xmat[geom].reshape(3, 3)))
            out[geom] = (pivot + r @ (pos - pivot), r @ mat)

        if self.proboscis > 0.01:
            pivot1 = self.data.xpos[self._rostrum_body]
            r1 = rot(frame[:, 1], -1.4 * self.proboscis)
            turn(self._rostrum_geom, pivot1, r1)
            turn(self._haustellum_geom, pivot1, r1)
            # The haustellum unfolds a bit further around its own joint.
            pivot2 = pivot1 + r1 @ (self.data.xpos[self._haustellum_body] - pivot1)
            turn(self._haustellum_geom, pivot2, rot(frame[:, 1], -0.9 * self.proboscis))

        if self.flight is not None:
            u = (self.data.time - self.flight.t0) / FLIGHT_DURATION_S
            spread = np.radians(80.0) * min(1.0, 6 * u, 6 * (1 - u))
            beat = np.radians(45.0) * np.sin(2 * np.pi * WINGBEAT_VISUAL_HZ * (self.data.time - self.flight.t0))
            for side, sign in (("left", 1.0), ("right", -1.0)):
                body, geom = self._wing[side]
                hinge = self.data.xpos[body]
                # Swing the folded wing out sideways, then flap it up and down.
                turn(geom, hinge, rot(frame[:, 2], -sign * spread))
                turn(geom, hinge, rot(frame[:, 0], sign * beat))
        return out

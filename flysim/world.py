"""The shared world: several flies on flat ground, food, odor sources and
thrown balls. Each fly (flysim/fly.py) has its own physics simulation, walking
controller and, optionally, connectome brain.

The world advances in chunks of SENSE_EVERY physics steps (10 ms). Between
chunks, on the calling thread, it exchanges everything flies share: what each
one senses (including the others), brains in and out, food, balls, bumping.
During a chunk every fly's physics runs on its own worker thread.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import itertools

import numpy as np

from flysim.fly import FEED_SECONDS, SENSE_EVERY, TIMESTEP, Fly

SPAWN_RADIUS = 7.0  # mm, flies start on a circle around the origin
MAX_FLIES = 6  # each fly costs one physics thread and one brain thread
CHUNK_S = SENSE_EVERY * TIMESTEP

# --- balls
THREAT_RADIUS = 3.0  # mm, the ball launched by the "Menace" button
THREAT_SPEED = 120.0  # mm/s
THREAT_START_DIST = 60.0  # mm
THROW_RADIUS = 1.6  # mm, balls thrown with the mouse
THROW_SPEED = 160.0  # mm/s
THROW_START_DIST = 45.0  # mm from the aimed point, back toward the camera

# --- flies bumping into each other (soft: pushing a body whose feet grip the
# ground is what makes the leg joints diverge)
BUMP_DISTANCE = 1.8  # mm between thoraxes; two flies can share a drop
BUMP_STIFFNESS_BW = 1.5  # body weights of push at full overlap


@dataclass
class Item:
    id: int
    kind: str  # "sugar" | "bitter" | "vinegar"
    pos: np.ndarray
    amount: float = 1.0  # sugar drops shrink as flies eat


@dataclass
class Projectile:
    pos: np.ndarray
    vel: np.ndarray
    radius: float
    lifetime: float
    age: float = 0.0
    hit: bool = False


class World:
    def __init__(self, n_flies: int = 1, brains: list | None = None):
        rng = np.random.default_rng(1)
        brains = brains or [None] * n_flies
        self.items: list[Item] = []
        self._item_ids = itertools.count(1)
        self.projectiles: list[Projectile] = []
        self.flies: list[Fly] = []
        for i in range(n_flies):
            angle = 2 * np.pi * i / n_flies
            if n_flies == 1:
                pos, yaw = (0.0, 0.0, 0.8), 0.0
            else:
                pos = (SPAWN_RADIUS * np.cos(angle), SPAWN_RADIUS * np.sin(angle), 0.8)
                yaw = angle + np.pi / 2 + rng.uniform(-0.6, 0.6)
            self.flies.append(Fly(self, i, f"fly{i}", pos, yaw, brains[i]))
        self._pool = ThreadPoolExecutor(max_workers=MAX_FLIES, thread_name_prefix="fly-physics")
        self._fly_names = itertools.count(n_flies)
        self._spawn_rng = np.random.default_rng(7)

    def build_fly(self, brain=None) -> Fly:
        """A new fly near the middle, clear of the others, not yet in the world.
        Building (MuJoCo model compile + warmup) takes ~1 s, so callers may do it
        off the simulation thread and `insert_fly` it afterwards."""
        index = len(self.flies)
        name = f"fly{next(self._fly_names)}"  # unique even after removals
        others = [f.position[:2].copy() for f in self.flies]
        for _ in range(50):
            r = self._spawn_rng.uniform(3.0, 10.0)
            a = self._spawn_rng.uniform(0, 2 * np.pi)
            xy = np.array([r * np.cos(a), r * np.sin(a)])
            if all(np.linalg.norm(xy - o) > 4.0 for o in others):
                break
        yaw = self._spawn_rng.uniform(-np.pi, np.pi)
        fly = Fly(self, index, name, (xy[0], xy[1], 0.8), yaw, brain)
        # Start at the world's current time so every fly shares one clock.
        fly.data.time = self.time
        fly._snapshot()
        if brain is not None:
            brain.sync_clock(self.time)
        return fly

    def insert_fly(self, fly: Fly):
        self.flies.append(fly)

    def remove_fly(self, fly: Fly):
        """Take a fly out of the world (at least one stays)."""
        if len(self.flies) <= 1 or fly not in self.flies:
            return
        self.flies.remove(fly)
        for i, f in enumerate(self.flies):
            f.index = i
        for other in self.flies:  # forget how big it looked to the others
            other._theta_prev = {k: v for k, v in other._theta_prev.items() if k[0] != fly.name}
        if fly.brain is not None:
            fly.brain.stop()

    @property
    def time(self) -> float:
        return self.flies[0].data.time

    def reset(self):
        self.projectiles = []
        for fly in self.flies:
            fly.reset()

    # ------------------------------------------------------------ items and balls

    def add_item(self, kind: str, x: float, y: float) -> Item:
        item = Item(next(self._item_ids), kind, np.array([x, y, 0.0]))
        self.items.append(item)
        return item

    def clear_items(self):
        self.items = []

    def launch_threat(self, fly: Fly, from_left: bool | None = None):
        """A big dark ball flies at a fly's head from the front-left or front-right."""
        if from_left is None:
            from_left = bool(np.random.rand() < 0.5)
        azimuth = np.radians(55.0 if from_left else -55.0)
        direction_fly = np.array([np.cos(azimuth), np.sin(azimuth), 0.35])
        direction = fly.frame() @ (direction_fly / np.linalg.norm(direction_fly))
        start = fly.head() + direction * THREAT_START_DIST
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

    def projectile_list(self) -> list[tuple[float, float, float, float]]:
        return [(*p.pos, p.radius) for p in self.projectiles]

    def _update_projectiles(self, dt: float):
        kept = []
        for pr in self.projectiles:
            pr.pos = pr.pos + pr.vel * dt
            pr.age += dt
            for fly in self.flies:
                if not pr.hit and fly.flight is None and np.linalg.norm(pr.pos - fly.position) < pr.radius + 1.2:
                    # Knock the fly along the ball's path; the ball bounces off.
                    pr.hit = True
                    qv = fly._free_qvel
                    push = pr.vel / (np.linalg.norm(pr.vel) + 1e-9)
                    fly.data.qvel[qv : qv + 3] += push * 120.0 + np.array([0, 0, 60.0])
                    pr.vel = -0.3 * pr.vel + np.array([0, 0, 40.0])
            if pr.age < pr.lifetime and pr.pos[2] > -pr.radius:
                kept.append(pr)
        self.projectiles = kept

    def _eat(self, dt: float):
        for fly in self.flies:
            if not fly.behavior["feeding"]:
                continue
            for item in self.items:
                if item.kind == "sugar" and fly.touching(item):
                    item.amount -= dt / FEED_SECONDS
        self.items = [i for i in self.items if i.amount > 0]

    def _bump(self):
        """Soft repulsion between flies standing too close."""
        for fly in self.flies:
            fly.extra_force[:] = 0.0
        for a, b in itertools.combinations(self.flies, 2):
            if a.flight is not None or b.flight is not None:
                continue
            d = a.position[:2] - b.position[:2]
            dist = np.linalg.norm(d)
            if 1e-6 < dist < BUMP_DISTANCE:
                overlap = (BUMP_DISTANCE - dist) / BUMP_DISTANCE
                f = np.array([*(d / dist), 0.0]) * BUMP_STIFFNESS_BW * a.body_weight * overlap
                a.extra_force += f
                b.extra_force -= f

    # ------------------------------------------------------------ stepping

    def step_chunk(self):
        """Exchange between flies, then advance every fly's physics by one chunk
        (SENSE_EVERY steps) in parallel."""
        self._update_projectiles(CHUNK_S)
        self._eat(CHUNK_S)
        self._bump()
        for fly in self.flies:
            if fly.brain is not None:
                fly.drive = fly.sense(CHUNK_S)
                fly.brain.set_drive(fly.drive)
        # Brains run their next chunk on their own threads, overlapping physics.
        for fly in self.flies:
            if fly.brain is not None:
                fly.brain.advance_to(fly.data.time)
        for fly in self.flies:
            if fly.brain is not None:
                fly.act(fly.brain.rates)
            fly.behavior["flying"] = fly.flight is not None
        list(self._pool.map(lambda f: f.run_chunk(SENSE_EVERY), self.flies))

    @property
    def recoveries(self) -> int:
        return sum(f.recoveries for f in self.flies)

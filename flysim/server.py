"""FlySim server: runs the world in a background thread and streams it to browsers.

Run with:  uv run python -m flysim.server  [--flies N] [--no-brain] [--cpu-brain]

Flies can also be added at runtime from the UI (up to world.MAX_FLIES).
"""

import argparse
import asyncio
import json
import queue
import threading
import time
from pathlib import Path

import numpy as np
import uvicorn
from fastapi import FastAPI, Response, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles

from flysim.brain_gpu import GpuBrainLink
from flysim.brain_link import BrainLink, BrainModel
from flysim.export import frame_bytes, scene_description, spikes_bytes, visible_geoms
from flysim.fly import Fly
from flysim.perf import disable_windows_power_throttling, tune_gil_switching
from flysim.world import CHUNK_S, MAX_FLIES, World

WEB_DIR = Path(__file__).resolve().parent.parent / "web"
STREAM_HZ = 30
STATUS_HZ = 5
# Max wall-clock time per tick spent stepping, so a slow sim degrades to slow
# motion instead of falling ever further behind.
STEP_BUDGET_S = 0.03


class Simulation(threading.Thread):
    def __init__(self, n_flies: int, with_brain: bool, gpu: bool = True):
        super().__init__(daemon=True)
        self.brain_model = None
        self.gpu_engine = None
        brains = None
        if with_brain:
            print("Loading the FlyWire brain (139k neurons, 15M connections)...")
            self.brain_model = BrainModel()
            print(f"  ready in {self.brain_model.load_seconds:.0f} s")
            if gpu:
                self.gpu_engine = _gpu_engine(self.brain_model)
            brains = [self._new_brain(seed=i) for i in range(n_flies)]
            for b in brains:
                b.start()
        self.world = World(n_flies=n_flies, brains=brains)
        self._rebuild_parts()
        self.building = False  # a fly is being added in the background
        self.selected = 0
        self._watch_selected()
        self.commands: queue.Queue = queue.Queue()
        self.lock = threading.Lock()
        self.rtf = 0.0
        self.latest_frame = self._frame()

    def _new_brain(self, seed: int):
        """One fly's brain: a slot on the GPU engine, or a CPU brain thread."""
        if self.gpu_engine is not None:
            return GpuBrainLink(self.gpu_engine)
        return BrainLink(self.brain_model, seed=seed)

    def _watch_selected(self):
        """Only the selected fly's brain streams its spiking neurons (map)."""
        for i, fly in enumerate(self.world.flies):
            if fly.brain is not None:
                fly.brain.watched = i == self.selected

    def _rebuild_parts(self):
        """Scene parts: fly 0's model brings the ground, then every fly's body.
        Bumping `scene_version` makes every client reload the scene."""
        self.parts = [
            (fly, visible_geoms(fly.model, include_world=fly.index == 0)) for fly in self.world.flies
        ]
        self.scene_version = getattr(self, "scene_version", 0) + 1

    def _build_fly(self):
        """Off the simulation thread: new brain + new fly physics (~1 s)."""
        try:
            brain = None
            if self.brain_model is not None:
                brain = self._new_brain(seed=len(self.world.flies))
                brain.start()
            fly = self.world.build_fly(brain)
            self.commands.put({"type": "_insert_fly", "fly": fly})
        except Exception:
            self.building = False
            raise

    @property
    def selected_fly(self):
        return self.world.flies[self.selected]

    def init_message(self) -> str:
        desc = scene_description([(fly.model, ids, fly.index) for fly, ids in self.parts])
        # Index (in the scene's geom list) of each fly's thorax, for the camera.
        follow, offset = [], 0
        for fly, ids in self.parts:
            rows = np.flatnonzero(fly.model.geom_bodyid[ids] == fly.thorax)
            follow.append(offset + int(rows[0]) if len(rows) else -1)
            offset += len(ids)
        desc.update(type="init", follow_geoms=follow, selected=self.selected,
                    has_brain=self.brain_model is not None, max_flies=MAX_FLIES)
        return json.dumps(desc)

    def status_message(self) -> str:
        w = self.world
        sel = self.selected_fly
        status = {
            "type": "status",
            "selected": self.selected,
            "can_add_fly": not self.building and len(w.flies) < MAX_FLIES,
            "can_remove_fly": not self.building and len(w.flies) > 1,
            "building": self.building,
            "behavior": sel.behavior,
            "drive": sel.drive,
            "flies": [
                {"behavior": {k: v for k, v in f.behavior.items() if v is True}}
                for f in w.flies
            ],
            "items": [
                {"id": i.id, "kind": i.kind, "x": float(i.pos[0]), "y": float(i.pos[1]), "amount": round(i.amount, 3)}
                for i in w.items
            ],
        }
        if sel.brain is not None:
            status["brain"] = {
                "rates": sel.brain.rates,
                "n_active": int(sel.brain.n_active),
                # One shared engine on the GPU; one thread per brain on the CPU.
                "load": self.gpu_engine.load if self.gpu_engine is not None
                else sum(f.brain.load for f in w.flies if f.brain),
                "device": "GPU" if self.gpu_engine is not None else "CPU",
            }
        return json.dumps(status)

    def _geom(self, scene_index: int):
        """Scene geom index -> (fly, model geom id)."""
        for fly, ids in self.parts:
            if scene_index < len(ids):
                return fly, int(ids[scene_index])
            scene_index -= len(ids)
        return None, -1

    def handle(self, msg: dict):
        kind = msg.get("type")
        w = self.world
        sel = self.selected_fly
        if kind == "select":
            self.selected = int(np.clip(msg.get("fly", 0), 0, len(w.flies) - 1))
            self._watch_selected()
        elif kind == "add_fly":
            if not self.building and len(w.flies) < MAX_FLIES:
                self.building = True
                threading.Thread(target=self._build_fly, daemon=True).start()
        elif kind == "remove_fly":
            if not self.building and len(w.flies) > 1:
                w.remove_fly(sel)
                self._rebuild_parts()
                self.selected = min(self.selected, len(w.flies) - 1)
                self._watch_selected()
        elif kind == "_insert_fly" and isinstance(msg.get("fly"), Fly):  # from _build_fly only
            w.insert_fly(msg["fly"])
            self._rebuild_parts()
            self.selected = len(w.flies) - 1
            self._watch_selected()
            self.building = False
        elif kind == "reset":
            w.reset()
        elif kind == "push":
            idx = int(msg.get("geom", -1))
            if idx < 0:
                for f in w.flies:
                    f.clear_push()
                return
            fly, gid = self._geom(idx)
            if fly is not None and gid in fly.geom_ids:
                fly.set_push(gid, np.array(msg["local"]), np.array(msg["target"]))
        elif kind == "item" and msg.get("kind") in ("sugar", "bitter", "vinegar"):
            w.add_item(msg["kind"], float(msg["x"]), float(msg["y"]))
        elif kind == "clear_items":
            w.clear_items()
        elif kind == "threat":
            w.launch_threat(sel)
        elif kind == "throw":
            w.throw(np.array(msg["from"], dtype=float), np.array(msg["to"], dtype=float))
        elif kind == "take_off":
            sel.take_off()

    def _frame(self) -> bytes:
        parts = []
        for fly, ids in self.parts:
            overrides: dict = {}
            fly.visual_overrides(overrides)
            parts.append((fly.data, ids, overrides))
        return frame_bytes(parts, self.world.time, self.rtf, self.world.projectile_list())

    def run(self):
        wall_prev = time.perf_counter()
        debt = 0.0  # simulated time owed to the wall clock
        while True:
            while not self.commands.empty():
                self.handle(self.commands.get_nowait())

            tick_start = time.perf_counter()
            wall_elapsed = min(tick_start - wall_prev, 0.1)
            wall_prev = tick_start

            sim_start = self.world.time
            debt += wall_elapsed
            while debt >= CHUNK_S and time.perf_counter() - tick_start < STEP_BUDGET_S:
                self.world.step_chunk()
                debt -= CHUNK_S
            debt = min(debt, CHUNK_S)
            sim_advanced = self.world.time - sim_start
            if wall_elapsed > 0 and sim_advanced >= 0:
                self.rtf = 0.9 * self.rtf + 0.1 * (sim_advanced / wall_elapsed)

            frame = self._frame()
            with self.lock:
                self.latest_frame = frame

            spare = 1 / 120 - (time.perf_counter() - tick_start)
            if spare > 0:
                time.sleep(spare)


def _gpu_engine(model):
    """The shared GPU brain engine, or None (with the reason) if unavailable."""
    try:
        from flysim.brain_gpu import GpuBrainEngine

        engine = GpuBrainEngine(model, capacity=MAX_FLIES)  # compiles kernels, records the graph
        print("  brains on the GPU")
        return engine
    except Exception as e:  # no CuPy, no NVIDIA GPU, driver issue...
        print(f"  GPU unavailable ({type(e).__name__}: {e}); brains on the CPU")
        return None


def _parse_args():
    parser = argparse.ArgumentParser(description="FlySim server")
    parser.add_argument("--flies", type=int, default=1, help="flies at start (default 1; more can be added from the UI)")
    parser.add_argument("--no-brain", action="store_true", help="physics only, no connectome")
    parser.add_argument("--cpu-brain", action="store_true", help="run brains on the CPU even if a GPU is available")
    args, _ = parser.parse_known_args()
    return args


disable_windows_power_throttling()
tune_gil_switching()
_args = _parse_args()
sim = Simulation(n_flies=int(np.clip(_args.flies, 1, MAX_FLIES)), with_brain=not _args.no_brain,
                 gpu=not _args.cpu_brain)
app = FastAPI()


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    await ws.send_text(sim.init_message())
    scene_version = sim.scene_version

    async def receive():
        while True:
            sim.commands.put(json.loads(await ws.receive_text()))

    receiver = asyncio.create_task(receive())
    # Schedule against fixed deadlines: Windows' ~15 ms sleep granularity would
    # otherwise drag a naive sleep(1/30) loop down to ~22 fps.
    next_frame = time.perf_counter()
    n = 0
    watched, spike_seq = None, 0
    try:
        while not receiver.done():
            if sim.scene_version != scene_version:
                scene_version = sim.scene_version
                await ws.send_text(sim.init_message())
            with sim.lock:
                frame = sim.latest_frame
            await ws.send_bytes(frame)
            brain = sim.selected_fly.brain
            if brain is not None:
                if brain is not watched:  # selection changed: start from now
                    watched, spike_seq = brain, brain.spikes_since(-1)[0]
                spike_seq, spiked = brain.spikes_since(spike_seq)
                if len(spiked):
                    await ws.send_bytes(spikes_bytes(spiked))
            if n % (STREAM_HZ // STATUS_HZ) == 0:
                await ws.send_text(sim.status_message())
            n += 1
            next_frame += 1 / STREAM_HZ
            now = time.perf_counter()
            if next_frame < now:
                next_frame = now
            await asyncio.sleep(next_frame - now)
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        receiver.cancel()


@app.get("/api/brain/points")
def brain_points():
    if sim.brain_model is None:
        return Response(status_code=404)
    return Response(sim.brain_model.map_points(), media_type="application/octet-stream")


@app.get("/api/brain/meta")
def brain_meta():
    if sim.brain_model is None:
        return Response(status_code=404)
    return sim.brain_model.map_meta()


@app.middleware("http")
async def no_cache(request, call_next):
    # The frontend changes often during development; always revalidate.
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-cache"
    return response


app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")


def main():
    sim.start()
    print(f"FlySim ({len(sim.world.flies)} flies): http://localhost:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")


if __name__ == "__main__":
    main()

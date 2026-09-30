"""FlySim server: runs the physics in a background thread and streams it to browsers.

Run with:  uv run python -m flysim.server  [--no-brain]
"""

import asyncio
import json
import queue
import sys
import threading
import time
from pathlib import Path

import numpy as np
import uvicorn
from fastapi import FastAPI, Response, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles

from flysim.export import frame_bytes, scene_description, spikes_bytes, visible_geoms
from flysim.perf import disable_windows_power_throttling
from flysim.brain_link import BrainLink
from flysim.world import World

WEB_DIR = Path(__file__).resolve().parent.parent / "web"
STREAM_HZ = 30
STATUS_HZ = 5
# Max wall-clock time per tick spent stepping physics, so a slow sim degrades to
# slow motion instead of falling ever further behind.
STEP_BUDGET_S = 0.025


class Simulation(threading.Thread):
    def __init__(self, with_brain: bool = True):
        super().__init__(daemon=True)
        self.brain = None
        if with_brain:
            print("Loading the FlyWire brain (139k neurons, 15M connections)...")
            self.brain = BrainLink()
            print(f"  ready in {self.brain.load_seconds:.0f} s")
            self.brain.start()
        self.world = World(brain=self.brain)
        self.geom_ids = visible_geoms(self.world.model)
        self.commands: queue.Queue = queue.Queue()
        self.lock = threading.Lock()
        self.rtf = 0.0
        self.latest_frame = frame_bytes(self.world.data, self.geom_ids, 0.0)

    def init_message(self) -> str:
        desc = scene_description(self.world.model, self.geom_ids)
        follow = np.flatnonzero(self.world.model.geom_bodyid[self.geom_ids] == self.world.follow_body)
        desc.update(
            type="init",
            follow_geom=int(follow[0]) if len(follow) else -1,
            walking=self.world.walking,
            has_brain=self.brain is not None,
        )
        return json.dumps(desc)

    def status_message(self) -> str:
        w = self.world
        status = {
            "type": "status",
            "walking": w.walking,
            "behavior": w.behavior,
            "drive": w.drive,
            "items": [
                {"id": i.id, "kind": i.kind, "x": float(i.pos[0]), "y": float(i.pos[1]), "amount": round(i.amount, 3)}
                for i in w.items
            ],
        }
        if self.brain is not None:
            status["brain"] = {
                "rates": self.brain.rates,
                "n_active": int(self.brain.n_active),
                "load": self.brain.load,
            }
        return json.dumps(status)

    def handle(self, msg: dict):
        kind = msg.get("type")
        w = self.world
        if kind == "walk":
            w.walking = bool(msg.get("on"))
        elif kind == "reset":
            w.reset()
        elif kind == "push":
            idx = int(msg.get("geom", -1))
            if idx < 0:
                w.clear_push()
            else:
                w.set_push(int(self.geom_ids[idx]), np.array(msg["local"]), np.array(msg["target"]))
        elif kind == "item" and msg.get("kind") in ("sugar", "bitter", "vinegar"):
            w.add_item(msg["kind"], float(msg["x"]), float(msg["y"]))
        elif kind == "clear_items":
            w.clear_items()
        elif kind == "threat":
            w.launch_threat()
        elif kind == "throw":
            w.throw(np.array(msg["from"], dtype=float), np.array(msg["to"], dtype=float))
        elif kind == "take_off":
            w.take_off()

    def _frame(self) -> bytes:
        w = self.world
        return frame_bytes(w.data, self.geom_ids, self.rtf, w.visual_overrides(), w.projectile_list())

    def run(self):
        dt = self.world.model.opt.timestep
        wall_prev = time.perf_counter()
        while True:
            while not self.commands.empty():
                self.handle(self.commands.get_nowait())

            tick_start = time.perf_counter()
            wall_elapsed = min(tick_start - wall_prev, 0.1)
            wall_prev = tick_start

            sim_start = self.world.data.time
            n_target = int(round(wall_elapsed / dt))
            for _ in range(n_target):
                self.world.step()
                if time.perf_counter() - tick_start > STEP_BUDGET_S:
                    break
            sim_advanced = self.world.data.time - sim_start
            if wall_elapsed > 0 and sim_advanced >= 0:
                self.rtf = 0.9 * self.rtf + 0.1 * (sim_advanced / wall_elapsed)

            frame = self._frame()
            with self.lock:
                self.latest_frame = frame

            spare = 1 / 120 - (time.perf_counter() - tick_start)
            if spare > 0:
                time.sleep(spare)


disable_windows_power_throttling()
sim = Simulation(with_brain="--no-brain" not in sys.argv)
app = FastAPI()


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    await ws.send_text(sim.init_message())

    async def receive():
        while True:
            sim.commands.put(json.loads(await ws.receive_text()))

    receiver = asyncio.create_task(receive())
    # Schedule against fixed deadlines: Windows' ~15 ms sleep granularity would
    # otherwise drag a naive sleep(1/30) loop down to ~22 fps.
    next_frame = time.perf_counter()
    n = 0
    spike_seq = sim.brain.spikes_since(-1)[0] if sim.brain else 0
    try:
        while not receiver.done():
            with sim.lock:
                frame = sim.latest_frame
            await ws.send_bytes(frame)
            if sim.brain is not None:
                spike_seq, spiked = sim.brain.spikes_since(spike_seq)
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
    if sim.brain is None:
        return Response(status_code=404)
    return Response(sim.brain.map_points(), media_type="application/octet-stream")


@app.get("/api/brain/meta")
def brain_meta():
    if sim.brain is None:
        return Response(status_code=404)
    return sim.brain.map_meta()


@app.middleware("http")
async def no_cache(request, call_next):
    # The frontend changes often during development; always revalidate.
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-cache"
    return response


app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")


def main():
    sim.start()
    print("FlySim: http://localhost:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")


if __name__ == "__main__":
    main()

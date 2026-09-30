// FlySim frontend: renders MuJoCo geoms streamed from the Python server.
//
// Protocol (WebSocket /ws):
//   server -> client, JSON  {type: "init", geoms: [...], meshes: {...}, ...}
//   server -> client, JSON  {type: "status", behavior, brain: {rates, ...}, items} at ~5 Hz
//   server -> client, binary, starting with a uint32 tag:
//       1 = pose:   float32 [time, realtime_factor, n_projectiles,
//                   n_geoms * 12 (xpos[3] + xmat[9]), n_projectiles * 4 (x, y, z, r)]
//       2 = spikes: uint32 indices of neurons that spiked since the last message
//   client -> server, JSON  {type: "push" | "walk" | "reset" | "item" | "clear_items" |
//                            "threat" | "throw" | "take_off", ...}

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { BrainView } from "./brain.js";

const container = document.getElementById("viewport");
const $ = (id) => document.getElementById(id);

// ---------------------------------------------------------------- scene setup

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(container.clientWidth, container.clientHeight);
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
renderer.outputColorSpace = THREE.SRGBColorSpace;
container.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0d1016);
scene.fog = new THREE.Fog(0x0d1016, 30, 120);

// MuJoCo is Z-up.
THREE.Object3D.DEFAULT_UP.set(0, 0, 1);

const camera = new THREE.PerspectiveCamera(40, container.clientWidth / container.clientHeight, 0.05, 500);
camera.up.set(0, 0, 1);
camera.position.set(-6, -6, 4);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.target.set(0, 0, 0.5);

scene.add(new THREE.HemisphereLight(0xdfe8ff, 0x302820, 1.2));
const sun = new THREE.DirectionalLight(0xffffff, 2.2);
sun.position.set(-10, -6, 20);
sun.castShadow = true;
sun.shadow.mapSize.set(2048, 2048);
sun.shadow.camera.left = sun.shadow.camera.bottom = -10;
sun.shadow.camera.right = sun.shadow.camera.top = 10;
sun.shadow.camera.near = 1;
sun.shadow.camera.far = 60;
scene.add(sun);
scene.add(sun.target);

// The viewport shrinks when the brain map opens, so track the container.
new ResizeObserver(() => {
  const w = container.clientWidth, h = container.clientHeight;
  if (!w || !h) return;
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
  renderer.setSize(w, h);
}).observe(container);

// ---------------------------------------------------------------- geometry

function b64ToTyped(b64, Type) {
  const bin = atob(b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return new Type(bytes.buffer);
}

function buildMeshGeometry(mesh) {
  const verts = b64ToTyped(mesh.vertices, Float32Array);
  const faces = b64ToTyped(mesh.faces, Int32Array);
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.BufferAttribute(verts, 3));
  g.setIndex(new THREE.BufferAttribute(new Uint32Array(faces), 1));
  g.computeVertexNormals();
  return g;
}

// MuJoCo geom types: 0 plane, 2 sphere, 3 capsule, 4 ellipsoid, 5 cylinder, 6 box, 7 mesh
function buildPrimitive(g) {
  const s = g.size;
  switch (g.type) {
    case 0: {
      const half = s[0] > 0 ? s[0] : 50;
      return new THREE.PlaneGeometry(half * 2, (s[1] > 0 ? s[1] : 50) * 2);
    }
    case 2: return new THREE.SphereGeometry(s[0], 24, 16);
    case 3: return new THREE.CapsuleGeometry(s[0], s[1] * 2, 8, 16).rotateX(Math.PI / 2);
    case 4: return new THREE.SphereGeometry(1, 24, 16).scale(s[0], s[1], s[2]);
    case 5: return new THREE.CylinderGeometry(s[0], s[0], s[1] * 2, 24).rotateX(Math.PI / 2);
    case 6: return new THREE.BoxGeometry(s[0] * 2, s[1] * 2, s[2] * 2);
    default: return null;
  }
}

function floorMaterial() {
  const c = document.createElement("canvas");
  c.width = c.height = 256;
  const ctx = c.getContext("2d");
  ctx.fillStyle = "#1a1f29";
  ctx.fillRect(0, 0, 256, 256);
  ctx.fillStyle = "#212735";
  ctx.fillRect(0, 0, 128, 128);
  ctx.fillRect(128, 128, 128, 128);
  const tex = new THREE.CanvasTexture(c);
  tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
  tex.repeat.set(40, 40);
  tex.colorSpace = THREE.SRGBColorSpace;
  return new THREE.MeshStandardMaterial({ map: tex, roughness: 0.95 });
}

// Drosophila melanogaster palette, keyed by NeuroMechFly body-part name.
// FlyGym colors the fly with textures we don't stream, so we restyle it here.
const FLY_STYLE = [
  [/eye$/, { color: 0xa3201a, roughness: 0.25, clearcoat: 1, clearcoatRoughness: 0.2 }],
  [/wing$/, { color: 0xdde6f2, opacity: 0.22, roughness: 0.1, iridescence: 1, iridescenceIOR: 1.6 }],
  [/arista$/, { color: 0x4a3a2c }],
  [/(pedicel|funiculus)$/, { color: 0x9c7141 }],
  [/haltere$/, { color: 0xcaa46c }],
  [/(rostrum|haustellum)$/, { color: 0xc9a27a }],
  [/abdomen12$/, { color: 0xc59b5d }],
  [/abdomen3$/, { color: 0x8a6236 }],
  [/abdomen[45]$/, { color: 0x5e3f24 }],
  [/abdomen6$/, { color: 0x3f2a19 }],
  [/(coxa|trochanterfemur)$/, { color: 0xc6a06a }],
  [/(tibia|tarsus\d)$/, { color: 0xae8756 }],
  [/(thorax|head)$/, { color: 0xb3844a, roughness: 0.5, sheen: 0.6, sheenColor: 0xe0b77a }],
];

function flyMaterial(name) {
  const match = FLY_STYLE.find(([re]) => re.test(name));
  if (!match) return null;
  const { color, opacity = 1, ...rest } = match[1];
  return new THREE.MeshPhysicalMaterial({
    color,
    roughness: 0.45,
    metalness: 0,
    transparent: opacity < 1,
    opacity,
    depthWrite: opacity >= 1,
    side: THREE.DoubleSide,
    ...rest,
  });
}

// Per-geom scene objects, indexed like the server's geom list.
let geomObjects = [];
let flyRootGeom = -1;
const geomToBody = new Map();

function buildScene(init) {
  for (const o of geomObjects) if (o) scene.remove(o);
  geomObjects = [];
  geomToBody.clear();

  const meshGeoms = {};
  for (const [name, m] of Object.entries(init.meshes)) meshGeoms[name] = buildMeshGeometry(m);

  init.geoms.forEach((g, i) => {
    const geometry = g.type === 7 ? meshGeoms[g.mesh] : buildPrimitive(g);
    if (!geometry) { geomObjects.push(null); return; }

    let material = g.type === 0 ? floorMaterial() : flyMaterial(g.name);
    if (!material) {
      const [r, gg, b, a] = g.rgba;
      material = new THREE.MeshStandardMaterial({
        color: new THREE.Color(r, gg, b),
        roughness: 0.55,
        metalness: 0.05,
        transparent: a < 1,
        opacity: a,
        side: g.type === 7 ? THREE.DoubleSide : THREE.FrontSide,
      });
    }

    const obj = new THREE.Mesh(geometry, material);
    obj.matrixAutoUpdate = false;
    obj.castShadow = g.type !== 0 && !material.transparent;
    obj.receiveShadow = true;
    obj.userData.geomIndex = i;
    obj.userData.body = g.body;
    scene.add(obj);
    geomObjects.push(obj);
    geomToBody.set(i, g.body);
  });

  flyRootGeom = init.follow_geom ?? -1;
}

const tmpMatrix = new THREE.Matrix4();
const FRAME_HEADER = 3;

// ---------------------------------------------------------------- world items

const ballGeometry = new THREE.SphereGeometry(1, 32, 20);
const ballMaterial = new THREE.MeshStandardMaterial({ color: 0x1b1d24, roughness: 0.3, metalness: 0.25 });
const balls = [];

function updateProjectiles(f, offset, count) {
  while (balls.length < count) {
    const m = new THREE.Mesh(ballGeometry, ballMaterial);
    m.castShadow = true;
    scene.add(m);
    balls.push(m);
  }
  balls.forEach((m, k) => {
    m.visible = k < count;
    if (k < count) {
      const o = offset + k * 4;
      m.position.set(f[o], f[o + 1], f[o + 2]);
      m.scale.setScalar(f[o + 3]);
    }
  });
}

const ITEM_STYLE = {
  sugar: { color: 0xf5f0e6, emissive: 0x2a2418 },
  bitter: { color: 0x7fd36b, emissive: 0x0d2a0a },
};
const DROP_RADIUS = 0.8;
const itemMeshes = new Map(); // id -> mesh

function vinegarMesh(it) {
  // A small cup, and a faint haze showing roughly where the smell reaches.
  const group = new THREE.Group();
  const cup = new THREE.Mesh(
    new THREE.CylinderGeometry(1.4, 1.1, 1.6, 32, 1, true).rotateX(Math.PI / 2),
    new THREE.MeshStandardMaterial({ color: 0xd7dde8, roughness: 0.25, side: THREE.DoubleSide }),
  );
  cup.position.z = 0.8;
  const liquid = new THREE.Mesh(
    new THREE.CircleGeometry(1.35, 32),
    new THREE.MeshStandardMaterial({ color: 0x8c2f39, roughness: 0.1, emissive: 0x2a0508 }),
  );
  liquid.position.z = 1.3;
  const haze = new THREE.Mesh(
    new THREE.SphereGeometry(18, 32, 16),
    new THREE.MeshBasicMaterial({ color: 0xb04858, transparent: true, opacity: 0.05, depthWrite: false }),
  );
  haze.scale.z = 0.25;
  group.add(cup, liquid, haze);
  group.position.set(it.x, it.y, 0);
  cup.castShadow = true;
  return group;
}

function syncItems(items) {
  const seen = new Set();
  for (const it of items) {
    seen.add(it.id);
    const existing = itemMeshes.get(it.id);
    if (existing) {
      if (it.kind === "sugar") {
        const r = DROP_RADIUS * Math.sqrt(Math.max(it.amount, 0.05));
        existing.scale.set(r, DROP_RADIUS * 0.35 * it.amount + 0.02, r);
      }
      continue;
    }
    if (it.kind === "vinegar") {
      const g = vinegarMesh(it);
      scene.add(g);
      itemMeshes.set(it.id, g);
      continue;
    }
    const style = ITEM_STYLE[it.kind];
    if (!style) continue;
    // A flattened, glossy droplet.
    const mesh = new THREE.Mesh(
      new THREE.SphereGeometry(1, 32, 16, 0, Math.PI * 2, 0, Math.PI / 2),
      new THREE.MeshPhysicalMaterial({
        ...style, roughness: 0.05, transmission: 0.35, thickness: 0.4, clearcoat: 1, transparent: true, opacity: 0.9,
      }),
    );
    // Three's hemisphere domes along local +Y: turn it to world +Z (up) and
    // flatten that axis.
    mesh.rotation.x = Math.PI / 2;
    mesh.scale.set(DROP_RADIUS, DROP_RADIUS * 0.35, DROP_RADIUS);
    mesh.position.set(it.x, it.y, 0);
    mesh.receiveShadow = true;
    scene.add(mesh);
    itemMeshes.set(it.id, mesh);
  }
  for (const [id, mesh] of itemMeshes) {
    if (!seen.has(id)) {
      scene.remove(mesh);
      itemMeshes.delete(id);
    }
  }
}

// ---------------------------------------------------------------- brain panel

// [key or [left, right] keys, label, role]; pairs are drawn as a left|right split bar.
const NEURON_ROWS = [
  [["giant_fiber_left", "giant_fiber_right"], "Fibre géante", "fuite"],
  [["dna02_left", "dna02_right"], "DNa02", "virage"],
  [["dna01_left", "dna01_right"], "DNa01", "virage"],
  ["proboscis", "MN trompe", "manger"],
  ["grooming", "DN toilettage", "se nettoyer"],
];
const RATE_FULL_SCALE = 150; // Hz

function buildNeuronPanel() {
  const root = $("neurons");
  root.innerHTML = "";
  for (const [key, label, role] of NEURON_ROWS) {
    const row = document.createElement("div");
    const pair = Array.isArray(key);
    row.className = "neuron" + (pair ? " pair" : "");
    row.title = role;
    row.innerHTML = `<span class="label">${label}</span><span class="bar">${
      pair ? "<span><i></i></span><span><i></i></span>" : "<i></i>"
    }</span><span class="hz"></span>`;
    row.dataset.key = JSON.stringify(key);
    root.appendChild(row);
  }
}

const BEHAVIOR_LABELS = {
  feeding: "🍬 mange",
  grooming: "🧹 se nettoie",
  escaping: "💨 fuite (fibre géante)",
  flying: "🪽 vole (scripté)",
  seeking_odor: "👃 suit une odeur (scripté)",
};

function senseLabel(group) {
  if (group.startsWith("sugar")) return "sucre";
  if (group.startsWith("bitter")) return "amer";
  if (group.startsWith("looming")) return "menace";
  if (group.startsWith("odor")) return "odeur";
  return "toucher";
}

function applyStatus(st) {
  syncItems(st.items || []);
  setWalking(st.walking);
  const chips = Object.entries(BEHAVIOR_LABELS)
    .filter(([k]) => st.behavior && st.behavior[k])
    .map(([, v]) => `<span class="chip">${v}</span>`);
  const turn = st.behavior ? st.behavior.turn : 0;
  if (Math.abs(turn) > 0.25) chips.push(`<span class="chip">${turn > 0 ? "↰ tourne à gauche" : "↱ tourne à droite"}</span>`);
  const senses = Object.keys(st.drive || {});
  if (senses.length) chips.push(`<span class="chip">sent : ${[...new Set(senses.map(senseLabel))].join(", ")}</span>`);
  $("behaviors").innerHTML = chips.join("");

  if (!st.brain) return;
  $("brain-section").hidden = false;
  $("brain-meta").textContent =
    `· ${st.brain.n_active.toLocaleString("fr-FR")} neurones actifs · charge ${st.brain.load.toFixed(2)}`;
  for (const row of $("neurons").children) {
    const key = JSON.parse(row.dataset.key);
    const rates = st.brain.rates;
    const bars = row.querySelectorAll("i");
    if (Array.isArray(key)) {
      const [l, r] = key.map((k) => rates[k] || 0);
      bars[0].style.width = `${Math.min(100, (l / RATE_FULL_SCALE) * 100)}%`;
      bars[1].style.width = `${Math.min(100, (r / RATE_FULL_SCALE) * 100)}%`;
      row.querySelector(".hz").textContent = `${Math.round(l)}|${Math.round(r)}`;
    } else {
      const v = rates[key] || 0;
      bars[0].style.width = `${Math.min(100, (v / RATE_FULL_SCALE) * 100)}%`;
      row.querySelector(".hz").textContent = Math.round(v);
    }
  }
}

function applyFrame(buf) {
  const f = new Float32Array(buf, 4);
  $("st-time").textContent = f[0].toFixed(2) + " s";
  $("st-rtf").textContent = "×" + f[1].toFixed(2);

  updateProjectiles(f, FRAME_HEADER + geomObjects.length * 12, f[2]);

  for (let i = 0; i < geomObjects.length; i++) {
    const obj = geomObjects[i];
    if (!obj) continue;
    const o = FRAME_HEADER + i * 12;
    // MuJoCo xmat is row-major 3x3.
    tmpMatrix.set(
      f[o + 3], f[o + 4], f[o + 5], f[o + 0],
      f[o + 6], f[o + 7], f[o + 8], f[o + 1],
      f[o + 9], f[o + 10], f[o + 11], f[o + 2],
      0, 0, 0, 1,
    );
    obj.matrix.copy(tmpMatrix);
    obj.matrixWorldNeedsUpdate = true;
  }
}

// ---------------------------------------------------------------- networking

let ws;

function connect() {
  ws = new WebSocket(`ws://${location.host}/ws`);
  ws.binaryType = "arraybuffer";
  $("st-conn").textContent = "connexion…";

  ws.onopen = () => ($("st-conn").textContent = "ok");
  ws.onclose = () => {
    $("st-conn").textContent = "perdue";
    setTimeout(connect, 1500);
  };
  ws.onmessage = (ev) => {
    if (typeof ev.data === "string") {
      const msg = JSON.parse(ev.data);
      if (msg.type === "init") {
        buildScene(msg);
        setWalking(msg.walking);
        if (msg.has_brain) {
          buildNeuronPanel();
          if (!brainLoading) setBrainOpen(true);
        }
      } else if (msg.type === "status") {
        applyStatus(msg);
      }
    } else {
      const tag = new DataView(ev.data).getUint32(0, true);
      if (tag === 1) applyFrame(ev.data);
      else if (tag === 2) brain.spikes(new Uint32Array(ev.data, 4));
    }
  };
}

function send(msg) {
  if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify(msg));
}

// ---------------------------------------------------------------- UI / tools

let tool = "orbit";
let following = true;
let walking = false;

function setTool(t) {
  tool = t;
  document.querySelectorAll("[data-tool]").forEach((b) => b.classList.toggle("active", b.dataset.tool === t));
  controls.enabled = t === "orbit";
  $("tool-hint").textContent = {
    orbit: "Glisser pour tourner, molette pour zoomer.",
    push: "Clique sur la mouche et glisse : elle sent le contact (tête → toilettage).",
    sugar: "Clique sur le sol : goutte de jus sucré. Elle la sent et vient la manger.",
    bitter: "Clique sur le sol pour poser une goutte amère (sans odeur).",
    vinegar: "Clique sur le sol pour poser une source d'odeur de vinaigre.",
    throw: "Clique où viser (sur la mouche ou le sol) : une balle part de ton côté.",
  }[t];
}
document.querySelectorAll("[data-tool]").forEach((b) => b.addEventListener("click", () => setTool(b.dataset.tool)));

function setWalking(w) {
  walking = w;
  $("btn-walk").classList.toggle("active", w);
}
$("btn-walk").addEventListener("click", () => {
  setWalking(!walking);
  send({ type: "walk", on: walking });
});
$("btn-follow").addEventListener("click", () => {
  following = !following;
  $("btn-follow").classList.toggle("active", following);
});
$("btn-reset").addEventListener("click", () => send({ type: "reset" }));
$("btn-threat").addEventListener("click", () => send({ type: "threat" }));
$("btn-fly").addEventListener("click", () => send({ type: "take_off" }));
$("btn-clear").addEventListener("click", () => send({ type: "clear_items" }));

// Push tool: grab a fly body part, drag on a plane facing the camera, the
// server applies a spring force between the grabbed point and the cursor.
const raycaster = new THREE.Raycaster();
const pointer = new THREE.Vector2();
let grab = null; // {body, localPoint, plane}

function pointerToNdc(ev) {
  const r = renderer.domElement.getBoundingClientRect();
  pointer.set(((ev.clientX - r.left) / r.width) * 2 - 1, -((ev.clientY - r.top) / r.height) * 2 + 1);
}

const groundPlane = new THREE.Plane(new THREE.Vector3(0, 0, 1), 0);

renderer.domElement.addEventListener("pointerdown", (ev) => {
  if (tool === "throw") {
    pointerToNdc(ev);
    raycaster.setFromCamera(pointer, camera);
    const hits = raycaster.intersectObjects(geomObjects.filter((o) => o && o.userData.body > 0));
    const target = hits.length ? hits[0].point : new THREE.Vector3();
    if (hits.length || raycaster.ray.intersectPlane(groundPlane, target)) {
      send({ type: "throw", from: camera.position.toArray(), to: target.toArray() });
    }
    return;
  }
  if (tool === "sugar" || tool === "bitter" || tool === "vinegar") {
    pointerToNdc(ev);
    raycaster.setFromCamera(pointer, camera);
    const p = new THREE.Vector3();
    if (raycaster.ray.intersectPlane(groundPlane, p)) send({ type: "item", kind: tool, x: p.x, y: p.y });
    return;
  }
  if (tool !== "push") return;
  pointerToNdc(ev);
  raycaster.setFromCamera(pointer, camera);
  const hits = raycaster.intersectObjects(geomObjects.filter((o) => o && o.userData.body > 0));
  if (!hits.length) return;
  const hit = hits[0];
  const normal = camera.getWorldDirection(new THREE.Vector3()).negate();
  grab = {
    body: hit.object.userData.body,
    // Grab point expressed in the geom's local frame so it follows the body.
    local: hit.object.worldToLocal(hit.point.clone()),
    object: hit.object,
    plane: new THREE.Plane().setFromNormalAndCoplanarPoint(normal, hit.point),
    target: hit.point.clone(),
  };
  renderer.domElement.setPointerCapture(ev.pointerId);
});

renderer.domElement.addEventListener("pointermove", (ev) => {
  if (!grab) return;
  pointerToNdc(ev);
  raycaster.setFromCamera(pointer, camera);
  raycaster.ray.intersectPlane(grab.plane, grab.target);
});

function endGrab() {
  if (!grab) return;
  grab = null;
  send({ type: "push", geom: -1 });
}
renderer.domElement.addEventListener("pointerup", endGrab);
renderer.domElement.addEventListener("pointercancel", endGrab);

const grabLine = new THREE.Line(
  new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(), new THREE.Vector3()]),
  new THREE.LineBasicMaterial({ color: 0xf2b33d }),
);
grabLine.visible = false;
scene.add(grabLine);

let lastPushSent = 0;

function updateGrab(now) {
  grabLine.visible = !!grab;
  if (!grab) return;
  const anchor = grab.local.clone().applyMatrix4(grab.object.matrixWorld);
  grabLine.geometry.setFromPoints([anchor, grab.target]);
  if (now - lastPushSent > 30) {
    lastPushSent = now;
    send({
      type: "push",
      geom: grab.object.userData.geomIndex,
      local: grab.local.toArray(),
      target: grab.target.toArray(),
    });
  }
}

// ---------------------------------------------------------------- brain map

const brain = new BrainView($("brain-canvas"));
let brainLoading = null;

function setBrainOpen(open) {
  document.body.classList.toggle("brain-open", open);
  $("brainview").hidden = !open;
  $("btn-brain").classList.toggle("active", open);
  if (open) loadBrain();
}

function loadBrain() {
  if (!brainLoading) {
    brainLoading = brain.load().then(() => {
      $("brain-count").textContent = brain.n.toLocaleString("fr-FR");
    });
  }
  return brainLoading;
}

$("btn-brain").addEventListener("click", () => setBrainOpen($("brainview").hidden));
$("btn-brain-hide").addEventListener("click", () => setBrainOpen(false));

// ---------------------------------------------------------------- main loop

const followPos = new THREE.Vector3();
// Camera catch-up rate (1/s); frame-rate independent.
const FOLLOW_RATE = 8;
let frames = 0;
let fpsT0 = performance.now();
let lastNow = performance.now();

function animate(now) {
  requestAnimationFrame(animate);
  const dt = Math.min((now - lastNow) / 1000, 0.1);
  lastNow = now;

  if (following && flyRootGeom >= 0 && geomObjects[flyRootGeom]) {
    followPos.setFromMatrixPosition(geomObjects[flyRootGeom].matrix);
    const alpha = 1 - Math.exp(-FOLLOW_RATE * dt);
    const delta = followPos.clone().sub(controls.target).multiplyScalar(alpha);
    controls.target.add(delta);
    camera.position.add(delta);
    sun.position.set(followPos.x - 10, followPos.y - 6, 20);
    sun.target.position.copy(followPos);
  }

  updateGrab(now);
  controls.update();
  renderer.render(scene, camera);
  if (!$("brainview").hidden) {
    brain.render(dt);
    $("brain-rate").textContent = `${Math.round(brain.spikesPerSecond).toLocaleString("fr-FR")} neurones qui tirent / s`;
  }

  frames++;
  if (now - fpsT0 > 1000) {
    $("st-fps").textContent = frames;
    frames = 0;
    fpsT0 = now;
  }
}

setTool("orbit");
connect();
requestAnimationFrame(animate);

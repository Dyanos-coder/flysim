// Live map of the FlyWire brain: every neuron is a point at its real position,
// colored by family, flashing when it spikes.

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { CSS2DRenderer, CSS2DObject } from "three/addons/renderers/CSS2DRenderer.js";

// Colors per FlyWire super-class (order matches the server's class list).
const CLASS_STYLE = {
  optic: [0x38c6d9, "Lobes optiques"],
  central: [0x8a5cf6, "Cerveau central"],
  sensory: [0xf2c14e, "Neurones sensoriels"],
  visual_projection: [0xe052c8, "Projection visuelle"],
  ascending: [0x5fd08a, "Ascendants (du corps)"],
  descending: [0xff8a3d, "Descendants (vers le corps)"],
  sensory_ascending: [0xb8e05a, "Sensoriels ascendants"],
  visual_centrifugal: [0x4f7cff, "Visuels centrifuges"],
  motor: [0xff4d5e, "Moteurs"],
  endocrine: [0xffffff, "Endocrines"],
};
const UNKNOWN_COLOR = 0x777777;

// Key groups (names from /api/brain/meta) with a label on the map.
const LABELS = [
  ["in_sugar", "goût sucré", "input"],
  ["in_bitter", "goût amer", "input"],
  ["in_odor_vinegar", "odorat", "input"],
  ["in_looming", "LPLC2 · menace", "input"],
  ["in_head_bristle", "soies de la tête", "input"],
  [["giant_fiber_left", "giant_fiber_right"], "fibre géante", "output"],
  [["dna02_left", "dna02_right"], "DNa02 · virage", "output"],
  ["proboscis", "MN trompe", "output"],
  ["grooming", "DN toilettage", "output"],
];

const DECAY_S = 0.25;

const VERTEX = /* glsl */ `
  attribute vec3 color;
  attribute float activity;
  uniform float uScale;
  varying vec3 vColor;
  varying float vAlpha;
  void main() {
    vec4 mv = modelViewMatrix * vec4(position, 1.0);
    gl_Position = projectionMatrix * mv;
    float a = activity;
    gl_PointSize = uScale * (1.0 + 2.6 * a) / -mv.z;
    vColor = mix(color * 0.55, vec3(1.0) * 0.55 + color * 0.9, a);
    vAlpha = 0.16 + 0.84 * a;
  }
`;
const FRAGMENT = /* glsl */ `
  varying vec3 vColor;
  varying float vAlpha;
  void main() {
    float d = length(gl_PointCoord - 0.5);
    float falloff = smoothstep(0.5, 0.0, d);
    gl_FragColor = vec4(vColor * falloff * vAlpha, 1.0);
  }
`;

export class BrainView {
  constructor(container) {
    this.container = container;
    this.ready = false;
    this.spikesPerSecond = 0;
    this._spikeCount = 0;
    this._spikeT0 = performance.now();

    this.renderer = new THREE.WebGLRenderer({ antialias: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.setClearColor(0x080a12, 1);
    container.appendChild(this.renderer.domElement);

    this.labelRenderer = new CSS2DRenderer();
    this.labelRenderer.domElement.className = "brain-labels";
    container.appendChild(this.labelRenderer.domElement);

    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(35, 1, 1, 10000);
    this.camera.position.set(0, 40, 1150);
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.autoRotate = true;
    this.controls.autoRotateSpeed = 0.35;
    // Stop the slow spin once the user grabs the brain.
    this.controls.addEventListener("start", () => (this.controls.autoRotate = false));

    this.resize();
    new ResizeObserver(() => this.resize()).observe(container);
  }

  resize() {
    const w = this.container.clientWidth, h = this.container.clientHeight;
    if (!w || !h) return;
    this.renderer.setSize(w, h);
    this.labelRenderer.setSize(w, h);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    if (this.material) this.material.uniforms.uScale.value = 5.5 * h;
  }

  async load() {
    const [buf, meta] = await Promise.all([
      fetch("/api/brain/points").then((r) => r.arrayBuffer()),
      fetch("/api/brain/meta").then((r) => r.json()),
    ]);
    const n = new DataView(buf).getUint32(0, true);
    const positions = new Float32Array(buf, 4, n * 3);
    const classes = new Uint8Array(buf, 4 + n * 12, n);
    this.n = n;

    const colors = new Float32Array(n * 3);
    const palette = meta.classes.map((c) => new THREE.Color(CLASS_STYLE[c]?.[0] ?? UNKNOWN_COLOR));
    const unknown = new THREE.Color(UNKNOWN_COLOR);
    for (let i = 0; i < n; i++) {
      const c = palette[classes[i]] ?? unknown;
      colors[i * 3] = c.r; colors[i * 3 + 1] = c.g; colors[i * 3 + 2] = c.b;
    }
    this.activity = new Float32Array(n);

    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    geo.setAttribute("color", new THREE.BufferAttribute(colors, 3));
    this.activityAttr = new THREE.BufferAttribute(this.activity, 1);
    this.activityAttr.setUsage(THREE.DynamicDrawUsage);
    geo.setAttribute("activity", this.activityAttr);

    this.material = new THREE.ShaderMaterial({
      vertexShader: VERTEX,
      fragmentShader: FRAGMENT,
      uniforms: { uScale: { value: 5.5 * this.container.clientHeight } },
      blending: THREE.AdditiveBlending,
      depthWrite: false,
      transparent: true,
    });
    this.points = new THREE.Points(geo, this.material);
    this.scene.add(this.points);

    this._buildLabels(meta.groups, positions);
    this._buildLegend(meta.classes, classes);
    this.ready = true;
  }

  _buildLabels(groups, positions) {
    // Key neurons cluster (e.g. taste and proboscis around the subesophageal
    // zone), so each label sits off to the side -- inputs left, outputs right --
    // joined to its true location by a thin line.
    this.labels = [];
    const slot = { input: 0, output: 0 };
    const total = { input: LABELS.filter((l) => l[2] === "input").length, output: LABELS.filter((l) => l[2] === "output").length };
    for (const [keys, text, kind] of LABELS) {
      const idx = (Array.isArray(keys) ? keys : [keys]).flatMap((k) => groups[k] || []);
      if (!idx.length) continue;
      const c = new THREE.Vector3();
      for (const i of idx) c.add(new THREE.Vector3(positions[i * 3], positions[i * 3 + 1], positions[i * 3 + 2]));
      c.divideScalar(idx.length);
      const k = slot[kind]++;
      const dx = kind === "input" ? -170 : 170;
      const dy = (k - (total[kind] - 1) / 2) * 30;
      const el = document.createElement("div");
      el.className = "brain-anchor";
      el.innerHTML = `<svg><line x1="0" y1="0" x2="${dx}" y2="${dy}"/></svg><i></i>` +
        `<span class="brain-label ${kind}" style="left:${dx}px;top:${dy}px">${text}</span>`;
      const obj = new CSS2DObject(el);
      obj.position.copy(c);
      this.scene.add(obj);
      this.labels.push({ el, idx });
    }
  }

  _buildLegend(classNames, classes) {
    const counts = new Map();
    for (let i = 0; i < classes.length; i++) counts.set(classes[i], (counts.get(classes[i]) || 0) + 1);
    const legend = document.getElementById("brain-legend");
    legend.innerHTML = classNames
      .map((c, k) => [c, counts.get(k) || 0])
      .filter(([, cnt]) => cnt > 0)
      .sort((a, b) => b[1] - a[1])
      .map(([c, cnt]) => {
        const [color, label] = CLASS_STYLE[c] || [UNKNOWN_COLOR, c];
        return `<span><i style="background:#${color.toString(16).padStart(6, "0")}"></i>${label} <em>${cnt.toLocaleString("fr-FR")}</em></span>`;
      })
      .join("");
  }

  /** Forget the current glow (e.g. when another fly's brain is shown). */
  clearActivity() {
    if (!this.ready) return;
    this.activity.fill(0);
    this.activityAttr.needsUpdate = true;
  }

  /** Indices of neurons that just spiked. */
  spikes(indices) {
    if (!this.ready) return;
    const a = this.activity;
    for (let k = 0; k < indices.length; k++) a[indices[k]] = 1;
    this._spikeCount += indices.length;
  }

  render(dt) {
    if (!this.ready || !this.container.clientWidth) return;
    const a = this.activity;
    const f = Math.exp(-dt / DECAY_S);
    for (let i = 0; i < a.length; i++) if (a[i] > 0.003) a[i] *= f; else a[i] = 0;
    this.activityAttr.needsUpdate = true;

    for (const { el, idx } of this.labels) {
      let s = 0;
      for (const i of idx) s += a[i];
      el.classList.toggle("lit", s / idx.length > 0.08);
    }

    const now = performance.now();
    if (now - this._spikeT0 > 1000) {
      this.spikesPerSecond = (this._spikeCount * 1000) / (now - this._spikeT0);
      this._spikeCount = 0;
      this._spikeT0 = now;
    }
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
    this.labelRenderer.render(this.scene, this.camera);
  }
}

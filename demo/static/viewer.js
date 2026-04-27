/*
 * Raum demo viewer — Three.js.
 *
 * Renders a Gaussian cloud as coloured point sprites. The splat has a
 * soft radial falloff in the fragment shader (exp(-2*r^2)), which gives
 * the volumetric look without needing a real 3DGS tile-based splatter.
 * For a 1000-3000 splat demo at 60fps, sprites are more than enough.
 */

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

const VIEWER = document.getElementById("viewer");

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0a0a0f);

const camera = new THREE.PerspectiveCamera(
  45, VIEWER.clientWidth / VIEWER.clientHeight, 0.1, 100,
);
camera.position.set(3.5, 2.2, 4.5);

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(window.devicePixelRatio);
renderer.setSize(VIEWER.clientWidth, VIEWER.clientHeight);
VIEWER.appendChild(renderer.domElement);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.target.set(0, 0, 0);

// ── Axis + ground guides ────────────────────────────────
const axes = new THREE.AxesHelper(1.2);
axes.material.transparent = true;
axes.material.opacity = 0.35;
scene.add(axes);

const grid = new THREE.GridHelper(6, 12, 0x2a2a38, 0x1a1a24);
grid.position.y = -1.5;
scene.add(grid);

// ── Splat material ──────────────────────────────────────
/* The GL_POINTS primitive gives us a per-splat size in pixels. We map
 * physical scale + camera distance to pixel size in the vertex shader,
 * then fade each fragment by its distance from the centre. */
const splatVS = `
  attribute float aSize;
  attribute float aOpacity;
  attribute vec3 aColor;

  varying vec3 vColor;
  varying float vOpacity;

  void main() {
    vec4 mv = modelViewMatrix * vec4(position, 1.0);
    gl_Position = projectionMatrix * mv;

    // Perspective scaling: bigger when closer. 600 tuned for a
    // ~900px tall canvas.
    float dist = -mv.z;
    gl_PointSize = max(2.0, (aSize * 600.0) / dist);

    vColor = aColor;
    vOpacity = aOpacity;
  }
`;

/* Hard-core / soft-rim profile: the inner ~55% of the radius is fully
 * opaque, the outer rim ramps alpha to zero, and anything too faint is
 * discarded. This matters because we run dozens to hundreds of coloured
 * splats per object; with a pure Gaussian falloff + transparent blending,
 * overlapping splats of different colours pixel-mix into a tinted smear
 * (the "colours combining" bug). Hard cores let the GPU depth-test properly
 * so the nearer colour wins and only the thin rim feathers. */
const splatFS = `
  precision mediump float;
  varying vec3 vColor;
  varying float vOpacity;

  void main() {
    vec2 p = gl_PointCoord - 0.5;
    float r2 = dot(p, p) * 4.0;       // 0 at centre, 1 at edge
    if (r2 > 1.0) discard;
    float a = vOpacity * (1.0 - smoothstep(0.55, 1.0, r2));
    if (a < 0.15) discard;
    gl_FragColor = vec4(vColor, a);
  }
`;

let cloud = null;

// ── HTML label overlay ─────────────────────────────────
// Each predicted object gets a small div anchored to its world-space
// centre. We project its position on every frame and update transforms.
// This is cheaper and legible at any zoom, vs. a 3D ring sprite.
const labelOverlay = document.createElement("div");
labelOverlay.className = "label-overlay";
VIEWER.appendChild(labelOverlay);

let labelEntries = [];   // [{el, pos:Vector3}]
let labelsVisible = true;

function clearLabels() {
  for (const e of labelEntries) e.el.remove();
  labelEntries = [];
}

export function setMarkers(objects) {
  clearLabels();
  if (!objects || objects.length === 0) return;
  for (const o of objects) {
    const el = document.createElement("div");
    el.className = "scene-label";
    el.innerHTML =
      `<span class="scene-label-name">${o.template}</span>` +
      `<span class="scene-label-conf">${Math.round(o.confidence * 100)}%</span>`;
    labelOverlay.appendChild(el);
    labelEntries.push({
      el,
      pos: new THREE.Vector3(o.position[0], o.position[1], o.position[2]),
    });
  }
}

export function setMarkersVisible(v) {
  labelsVisible = v;
  labelOverlay.style.display = v ? "" : "none";
}

function updateLabels() {
  if (!labelsVisible || labelEntries.length === 0) return;
  const w = VIEWER.clientWidth;
  const h = VIEWER.clientHeight;
  for (const entry of labelEntries) {
    const p = entry.pos.clone().project(camera);
    // In front of the camera only.
    if (p.z < -1 || p.z > 1) {
      entry.el.style.display = "none";
      continue;
    }
    entry.el.style.display = "";
    const x = (p.x * 0.5 + 0.5) * w;
    const y = (-p.y * 0.5 + 0.5) * h;
    entry.el.style.transform = `translate(-50%, -120%) translate(${x}px, ${y}px)`;
  }
}

function disposeCloud() {
  if (!cloud) return;
  scene.remove(cloud);
  cloud.geometry.dispose();
  cloud.material.dispose();
  cloud = null;
}

/**
 * Render a new cloud from the /generate payload.
 * splats = { means: [[x,y,z]...], scales: [[sx,sy,sz]...],
 *            opacities: [o...], colors: [[r,g,b]...] }
 */
export function setCloud(splats) {
  disposeCloud();
  const n = splats.means.length;
  if (n === 0) return;

  const positions = new Float32Array(n * 3);
  const colors = new Float32Array(n * 3);
  const sizes = new Float32Array(n);
  const opacities = new Float32Array(n);

  for (let i = 0; i < n; i++) {
    positions[3 * i + 0] = splats.means[i][0];
    positions[3 * i + 1] = splats.means[i][1];
    positions[3 * i + 2] = splats.means[i][2];
    colors[3 * i + 0] = splats.colors[i][0];
    colors[3 * i + 1] = splats.colors[i][1];
    colors[3 * i + 2] = splats.colors[i][2];
    // Use mean scale over axes. (True 3DGS is anisotropic; for v0 a
    // circular sprite is fine.)
    const s = splats.scales[i];
    sizes[i] = (s[0] + s[1] + s[2]) / 3.0;
    opacities[i] = splats.opacities[i];
  }

  const geom = new THREE.BufferGeometry();
  geom.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  geom.setAttribute("aColor", new THREE.BufferAttribute(colors, 3));
  geom.setAttribute("aSize", new THREE.BufferAttribute(sizes, 1));
  geom.setAttribute("aOpacity", new THREE.BufferAttribute(opacities, 1));

  /* depthWrite MUST be true here. The hard-core shader discards sub-0.15
   * alpha fragments so the remaining core is effectively opaque; writing
   * depth lets nearer splats occlude further ones, which is what keeps
   * coloured objects from bleeding through each other. */
  const mat = new THREE.ShaderMaterial({
    vertexShader: splatVS,
    fragmentShader: splatFS,
    transparent: true,
    depthWrite: true,
    depthTest: true,
    blending: THREE.NormalBlending,
  });

  cloud = new THREE.Points(geom, mat);
  scene.add(cloud);

  // Frame the cloud
  geom.computeBoundingSphere();
  const bs = geom.boundingSphere;
  if (bs && isFinite(bs.radius)) {
    const r = Math.max(bs.radius, 1.0);
    controls.target.copy(bs.center);
    const dir = camera.position.clone().sub(controls.target).normalize();
    camera.position.copy(controls.target).addScaledVector(dir, r * 3.5);
    controls.update();
  }
}

// ── Resize ──────────────────────────────────────────────
function onResize() {
  const w = VIEWER.clientWidth;
  const h = VIEWER.clientHeight;
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
  renderer.setSize(w, h, false);
}
window.addEventListener("resize", onResize);

// ── Loop ────────────────────────────────────────────────
function tick() {
  controls.update();
  renderer.render(scene, camera);
  updateLabels();
  requestAnimationFrame(tick);
}
tick();

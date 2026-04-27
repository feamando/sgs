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

const splatFS = `
  precision mediump float;
  varying vec3 vColor;
  varying float vOpacity;

  void main() {
    vec2 p = gl_PointCoord - 0.5;
    float r2 = dot(p, p) * 4.0;       // 0 at centre, 1 at edge
    if (r2 > 1.0) discard;
    float a = vOpacity * exp(-3.0 * r2);
    gl_FragColor = vec4(vColor, a);
  }
`;

let cloud = null;
let markers = null;   // coarse-mean outline markers per predicted object

function disposeCloud() {
  if (!cloud) return;
  scene.remove(cloud);
  cloud.geometry.dispose();
  cloud.material.dispose();
  cloud = null;
}

function disposeMarkers() {
  if (!markers) return;
  scene.remove(markers);
  markers.geometry.dispose();
  markers.material.dispose();
  markers = null;
}

// Ring marker material: a hollow circle. Makes predicted object
// centres visible without occluding the template splats.
const markerVS = `
  attribute float aSize;
  attribute vec3 aColor;
  varying vec3 vColor;
  void main() {
    vec4 mv = modelViewMatrix * vec4(position, 1.0);
    gl_Position = projectionMatrix * mv;
    float dist = -mv.z;
    gl_PointSize = max(8.0, (aSize * 1200.0) / dist);
    vColor = aColor;
  }
`;
const markerFS = `
  precision mediump float;
  varying vec3 vColor;
  void main() {
    vec2 p = gl_PointCoord - 0.5;
    float r = length(p) * 2.0;     // 0 at centre, 1 at edge
    // Hollow ring: alpha peaks near r=0.85
    float ring = smoothstep(0.7, 0.85, r) * (1.0 - smoothstep(0.92, 1.0, r));
    if (ring < 0.02) discard;
    gl_FragColor = vec4(vColor, ring);
  }
`;

/** Render coarse-mean outline rings for each predicted object. */
export function setMarkers(objects) {
  disposeMarkers();
  if (!objects || objects.length === 0) return;
  const n = objects.length;
  const positions = new Float32Array(n * 3);
  const colors = new Float32Array(n * 3);
  const sizes = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const o = objects[i];
    positions[3 * i + 0] = o.position[0];
    positions[3 * i + 1] = o.position[1];
    positions[3 * i + 2] = o.position[2];
    // Subtle accent tint (amber), faded by template confidence.
    const c = 0.6 + 0.4 * o.confidence;
    colors[3 * i + 0] = 1.0 * c;
    colors[3 * i + 1] = 0.7 * c;
    colors[3 * i + 2] = 0.28 * c;
    sizes[i] = 0.25 + 0.25 * Math.max(o.scale, 0.1);
  }
  const geom = new THREE.BufferGeometry();
  geom.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  geom.setAttribute("aColor", new THREE.BufferAttribute(colors, 3));
  geom.setAttribute("aSize", new THREE.BufferAttribute(sizes, 1));
  const mat = new THREE.ShaderMaterial({
    vertexShader: markerVS,
    fragmentShader: markerFS,
    transparent: true,
    depthWrite: false,
    blending: THREE.NormalBlending,
  });
  markers = new THREE.Points(geom, mat);
  scene.add(markers);
}

export function setMarkersVisible(v) {
  if (markers) markers.visible = v;
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

  const mat = new THREE.ShaderMaterial({
    vertexShader: splatVS,
    fragmentShader: splatFS,
    transparent: true,
    depthWrite: false,
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
  requestAnimationFrame(tick);
}
tick();

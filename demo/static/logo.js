/*
 * Radiance Labs animated logo: a tiny rotating sphere of Gaussian splats.
 * Runs on a 2D canvas (no WebGL) at ~36x36 px. Each frame, we project
 * a Fibonacci sphere of points and paint each with a radial-gradient
 * splat so it reads as a soft 3D blob rather than a disc.
 */

const PHI = (1 + Math.sqrt(5)) / 2;

function fibonacciSphere(n) {
  const pts = new Float32Array(n * 3);
  for (let i = 0; i < n; i++) {
    const theta = 2 * Math.PI * i / PHI;
    const phi = Math.acos(1 - 2 * (i + 0.5) / n);
    pts[3 * i + 0] = Math.sin(phi) * Math.cos(theta);
    pts[3 * i + 1] = Math.sin(phi) * Math.sin(theta);
    pts[3 * i + 2] = Math.cos(phi);
  }
  return pts;
}

export function mountLogo(host, { size = 36, n = 180 } = {}) {
  const dpr = Math.max(1, window.devicePixelRatio || 1);
  const canvas = document.createElement("canvas");
  canvas.width = size * dpr;
  canvas.height = size * dpr;
  canvas.style.width = size + "px";
  canvas.style.height = size + "px";
  canvas.style.display = "block";
  canvas.style.borderRadius = "50%";
  host.appendChild(canvas);
  const ctx = canvas.getContext("2d");

  const points = fibonacciSphere(n);
  const buf = new Float32Array(n * 3); // rotated
  const order = new Int32Array(n);
  for (let i = 0; i < n; i++) order[i] = i;

  const cx = canvas.width / 2;
  const cy = canvas.height / 2;
  const R = canvas.width * 0.42;
  const splatRadius = canvas.width * 0.11;
  let t0 = performance.now();

  function frame(now) {
    const t = (now - t0) * 0.001;
    const cA = Math.cos(t * 0.6), sA = Math.sin(t * 0.6);
    const cB = Math.cos(t * 0.35), sB = Math.sin(t * 0.35);

    // Rotate points around y then x.
    for (let i = 0; i < n; i++) {
      const x = points[3 * i + 0];
      const y = points[3 * i + 1];
      const z = points[3 * i + 2];
      const x1 = cA * x + sA * z;
      const z1 = -sA * x + cA * z;
      const y1 = cB * y + sB * z1;
      const z2 = -sB * y + cB * z1;
      buf[3 * i + 0] = x1;
      buf[3 * i + 1] = y1;
      buf[3 * i + 2] = z2;
    }

    // Sort back-to-front by z (larger z = closer to camera after above).
    order.sort((a, b) => buf[3 * a + 2] - buf[3 * b + 2]);

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.globalCompositeOperation = "lighter";
    for (let k = 0; k < n; k++) {
      const i = order[k];
      const z = buf[3 * i + 2];
      const depth = (z + 1) * 0.5;              // 0..1
      const px = cx + buf[3 * i + 0] * R;
      const py = cy - buf[3 * i + 1] * R;
      const r = splatRadius * (0.55 + 0.45 * depth);
      const alpha = 0.14 + 0.55 * depth;
      const grad = ctx.createRadialGradient(px, py, 0, px, py, r);
      grad.addColorStop(0.0, `rgba(255, 193, 100, ${alpha})`);
      grad.addColorStop(0.45, `rgba(255, 150, 80, ${alpha * 0.5})`);
      grad.addColorStop(1.0, "rgba(255, 80, 50, 0)");
      ctx.fillStyle = grad;
      ctx.beginPath();
      ctx.arc(px, py, r, 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.globalCompositeOperation = "source-over";
    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
}

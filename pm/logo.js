/*
 * Radiance Labs logo — multi-Gaussian dithered WebGL2 splat.
 * Ported from docs/brand/logos/webgl_logos_v5.html (`logo-multi`).
 * Three overlapping coloured spheres, each rendered through a simplex/
 * sphere shader with random dithering, additively blended into one glowy
 * warm blob.
 */

const VERT = `#version 300 es
precision mediump float;
layout(location=0) in vec4 a_position;
void main(){ gl_Position = a_position; }`;

const FRAG = `#version 300 es
precision mediump float;
uniform float u_time;
uniform vec2 u_resolution;
uniform vec4 u_colorBack;
uniform vec4 u_colorFront;
uniform float u_pxSize;
uniform vec2 u_offset;
out vec4 fragColor;

float hash21(vec2 p){
  p = fract(p * vec2(0.3183099, 0.3678794)) + 0.1;
  p += dot(p, p + 19.19);
  return fract(p.x * p.y);
}

void main(){
  float t = 0.5 * u_time;
  vec2 uv = gl_FragCoord.xy / u_resolution.xy;
  uv -= 0.5;
  uv += u_offset;

  float pxSize = u_pxSize;
  vec2 pxSizeUv = gl_FragCoord.xy;
  pxSizeUv -= 0.5 * u_resolution;
  pxSizeUv /= pxSize;

  vec2 pixelizedUv = floor(pxSizeUv) * pxSize / u_resolution.xy;
  pixelizedUv += 0.5;
  pixelizedUv -= 0.5;
  pixelizedUv += u_offset;

  vec2 shape_uv = pixelizedUv;
  vec2 ditheringNoise_uv = uv * u_resolution;

  // Sphere shape (from webgl_logos_v5 shader, u_shape >= 6.5 branch).
  shape_uv *= 2.0;
  float d = 1.0 - pow(length(shape_uv), 2.0);
  vec3 pos = vec3(shape_uv, sqrt(max(d, 0.0)));
  vec3 lp = normalize(vec3(cos(1.5 * t), 0.8, sin(1.25 * t)));
  float shape = 0.5 + 0.5 * dot(lp, pos);
  shape *= step(0.0, d);

  // Random dithering.
  float dithering = step(hash21(ditheringNoise_uv), shape) - 0.5;
  float res = step(0.5, shape + dithering);

  vec3 fgColor = u_colorFront.rgb * u_colorFront.a;
  float fgOpacity = u_colorFront.a;
  vec3 bgColor = u_colorBack.rgb * u_colorBack.a;
  float bgOpacity = u_colorBack.a;

  vec3 color = fgColor * res;
  float opacity = fgOpacity * res;
  color += bgColor * (1.0 - opacity);
  opacity += bgOpacity * (1.0 - opacity);
  fragColor = vec4(color, opacity);
}`;

const SPLATS = [
  { color: [0xF4 / 255, 0xA3 / 255, 0x00 / 255], offset: [-0.12, -0.05], speed: 0.9, opacity: 0.45 },
  { color: [0xFF / 255, 0x6B / 255, 0x6B / 255], offset: [ 0.10,  0.08], speed: 1.1, opacity: 0.40 },
  { color: [0xFF / 255, 0xD7 / 255, 0x00 / 255], offset: [ 0.05, -0.12], speed: 0.7, opacity: 0.42 },
];

export function mountLogo(host, { size = 36 } = {}) {
  const dpr = Math.max(1, window.devicePixelRatio || 1);
  const canvas = document.createElement("canvas");
  canvas.width = size * dpr;
  canvas.height = size * dpr;
  canvas.style.width = size + "px";
  canvas.style.height = size + "px";
  canvas.style.display = "block";
  canvas.style.borderRadius = "50%";
  host.appendChild(canvas);

  const gl = canvas.getContext("webgl2", { alpha: true, premultipliedAlpha: false });
  if (!gl) return;
  gl.enable(gl.BLEND);
  gl.blendFunc(gl.ONE, gl.ONE);

  function compile(type, src) {
    const s = gl.createShader(type);
    gl.shaderSource(s, src);
    gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) {
      console.error(gl.getShaderInfoLog(s));
      return null;
    }
    return s;
  }
  const vs = compile(gl.VERTEX_SHADER, VERT);
  const fs = compile(gl.FRAGMENT_SHADER, FRAG);
  const prog = gl.createProgram();
  gl.attachShader(prog, vs);
  gl.attachShader(prog, fs);
  gl.linkProgram(prog);
  if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) return;

  const buf = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, buf);
  gl.bufferData(gl.ARRAY_BUFFER,
    new Float32Array([-1,-1, 1,-1, -1,1, -1,1, 1,-1, 1,1]), gl.STATIC_DRAW);
  const loc = gl.getAttribLocation(prog, "a_position");
  gl.enableVertexAttribArray(loc);
  gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0);

  const u = {
    time:   gl.getUniformLocation(prog, "u_time"),
    res:    gl.getUniformLocation(prog, "u_resolution"),
    cb:     gl.getUniformLocation(prog, "u_colorBack"),
    cf:     gl.getUniformLocation(prog, "u_colorFront"),
    px:     gl.getUniformLocation(prog, "u_pxSize"),
    offset: gl.getUniformLocation(prog, "u_offset"),
  };

  const start = performance.now();
  function render() {
    const tSec = (performance.now() - start) * 0.001;
    gl.viewport(0, 0, canvas.width, canvas.height);
    gl.clearColor(0, 0, 0, 0);
    gl.clear(gl.COLOR_BUFFER_BIT);
    gl.useProgram(prog);
    gl.uniform2f(u.res, canvas.width, canvas.height);
    gl.uniform4f(u.cb, 0, 0, 0, 0);
    gl.uniform1f(u.px, 2);
    for (const s of SPLATS) {
      gl.uniform4f(u.cf, s.color[0], s.color[1], s.color[2], s.opacity);
      gl.uniform1f(u.time, tSec * s.speed);
      gl.uniform2f(u.offset, s.offset[0], s.offset[1]);
      gl.drawArrays(gl.TRIANGLES, 0, 6);
    }
    requestAnimationFrame(render);
  }
  render();
}

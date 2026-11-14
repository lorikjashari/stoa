// OQVAR landing: the loader, the chrome mark (raw WebGL, raymarched from a baked distance field), the scroll-scrubbed
// specimen, the drawing-type tabs, the benchmark chart and the live register count. No dependencies.
(() => {
"use strict";
const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const root = document.documentElement;
const reduce = matchMedia("(prefers-reduced-motion: reduce)").matches;

// ================================================================== timeline maths (after Lycoris Specimen)
const clamp01 = (x) => (x <= 0 ? 0 : x > 1 ? 1 : x);
const smoothstep = (a, b, x) => { const t = clamp01((x - a) / (b - a)); return t * t * (3 - 2 * t); };
const easeInOut = (t) => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2);
const easeOut = (t) => 1 - Math.pow(1 - t, 3);
const easeOutBack = (t) => 1 + 2.70158 * Math.pow(t - 1, 3) + 1.70158 * Math.pow(t - 1, 2);
const lerp = (a, b, t) => a + (b - a) * t;
const progressFrom = (top, height, viewport) => { const travel = height - viewport; return travel <= 0 ? 0 : clamp01(-top / travel); };
// scroll progress -> frame coordinate 0..n-1; each frame holds for `hold` of its slot before the next morph
const sceneCoord = (p, n, hold) => {
  const t = clamp01(p) * (n - 1), i = Math.min(Math.floor(t), n - 2), h = hold / 2;
  return i + easeInOut(clamp01((t - i - h) / (1 - 2 * h)));
};
// how present an element of frame `scene` is at `coord`; delay staggers it in, and mirrored, out
const reveal = (coord, scene, delay) => {
  const d = coord - scene, lag = d < 0 ? delay : 1 - delay;
  return clamp01((0.6 - Math.abs(d) - lag * 0.2) / 0.25);
};

// camera per frame: turn about the vertical, elevation, mark radius as a share of the short side, where it lands (clip space)
const KEYS = [
  { yaw: 0.55, pitch: 0.16, size: 0.44, ox: 0, oy: 0.33 },
  { yaw: 0.0, pitch: 0.0, size: 0.36, ox: 0, oy: 0.06 },
  { yaw: -0.75, pitch: 0.22, size: 0.62, ox: 0.5, oy: 0.0 },
  { yaw: 0.9, pitch: -0.42, size: 0.5, ox: -0.53, oy: -0.08 },
  { yaw: 2.9, pitch: 0.8, size: 0.56, ox: 0, oy: 0.82 },
];
const KEYS_TALL = [{ size: 0.5, oy: 0.36 }, { size: 0.46, oy: 0.08 }, { size: 0.52, ox: 0.18, oy: -0.42 }, { size: 0.52, ox: -0.18, oy: -0.44 }, { size: 0.66, oy: 0.84 }];
const LOAD_KEY = { yaw: 0, pitch: 0.12, size: 0.3, ox: 0, oy: 0.06 };
const SCENES = KEYS.length, HOLD = 0.34, FOV = (28 * Math.PI) / 180, RADIUS = 0.8, DEPTH = 0.11;
const keyAt = (coord, tall) => {
  const i = Math.max(0, Math.min(SCENES - 1, Math.floor(coord))), j = Math.min(SCENES - 1, i + 1), f = coord - i;
  const a = { ...KEYS[i], ...(tall ? KEYS_TALL[i] : {}) }, b = { ...KEYS[j], ...(tall ? KEYS_TALL[j] : {}) };
  return { yaw: lerp(a.yaw, b.yaw, f), pitch: lerp(a.pitch, b.pitch, f), size: lerp(a.size, b.size, f), ox: lerp(a.ox, b.ox, f), oy: lerp(a.oy, b.oy, f) };
};
const mixKey = (a, b, t) => ({ yaw: lerp(a.yaw, b.yaw, t), pitch: lerp(a.pitch, b.pitch, t), size: lerp(a.size, b.size, t), ox: lerp(a.ox, b.ox, t), oy: lerp(a.oy, b.oy, t) });

// ================================================================== the chrome mark
// An extruded mesh of the logo (outline + cap triangles baked by brand/src/logo_mesh.py). The rounded edges live in
// the shading: near every rim the normal bends towards the face it meets, steered by the baked distance texture.
const VERT = `
attribute vec3 a_p;   // x, y, which face of the slab (-1 back, +1 front)
attribute vec2 a_k;   // kind (0 cap, 1 side wall), material (0 body, 1 eye)
uniform mat4 u_vp; uniform vec2 u_off; uniform float u_depth; uniform float u_eyez; uniform vec2 u_eyexy;
varying vec3 v_w; varying vec2 v_k;
void main() {
  float eye = a_k.y;
  vec3 p = vec3(a_p.xy + u_eyexy * eye, a_p.z * u_depth * mix(1.0, 1.45, eye) + u_eyez * eye);
  v_w = p; v_k = a_k;
  gl_Position = u_vp * vec4(p, 1.0);
  gl_Position.xy += u_off * gl_Position.w;
}`;
const FRAG = `
precision highp float;
uniform sampler2D u_sdf; uniform vec3 u_cam; uniform float u_depth; uniform float u_eyez; uniform float u_alpha; uniform vec2 u_eyexy;
uniform vec3 u_bone; uniform vec3 u_red; uniform vec3 u_hot;
varying vec3 v_w; varying vec2 v_k;
const float BAND = 0.09375;   // distance band of the texture, in logo units (48 texels of 1024 over -1..1)
const float RND = 0.05;       // how far the rounded edge reaches in from each rim
const float TX = 0.00195;     // one texel, in logo units
float sd(vec2 p, float eye) {
  vec4 t = texture2D(u_sdf, vec2(p.x * 0.5 + 0.5, 0.5 - p.y * 0.5));
  return (mix(t.r, t.g, eye) - 0.5) * (2.0 * BAND);
}
// a dark studio: one overhead softbox, a tall side strip, a back rim and a low horizon line
float studio(vec3 r) {
  float key = pow(max(dot(r, normalize(vec3(-0.45, 0.85, 0.35))), 0.0), 14.0) * 2.6;
  float strip = smoothstep(0.55, 0.8, r.x) * smoothstep(-0.7, 0.3, r.y) * 1.4;
  float rim = smoothstep(0.55, 0.95, -r.z) * smoothstep(-0.2, 0.5, r.y) * 0.9;
  float hz = exp(-abs(r.y - 0.05) * 7.0) * 0.4;
  return key + strip + rim + hz;
}
void main() {
  float eye = v_k.y;
  float h = u_depth * mix(1.0, 1.45, eye);
  float lz = v_w.z - u_eyez * eye;
  float zs = lz < 0.0 ? -1.0 : 1.0;
  vec2 p = v_w.xy - u_eyexy * eye;
  vec2 g = vec2(sd(p + vec2(TX, 0.0), eye) - sd(p - vec2(TX, 0.0), eye), sd(p + vec2(0.0, TX), eye) - sd(p - vec2(0.0, TX), eye));
  g = dot(g, g) > 1e-12 ? normalize(g) : vec2(0.0);
  vec3 n;
  if (v_k.x < 0.5) {
    float a = (1.0 - clamp(-sd(p, eye) / RND, 0.0, 1.0)) * 0.7854;   // cap: 45 degrees at the rim, flat inside
    n = vec3(g * sin(a), zs * cos(a));
  } else {
    float b = (1.0 - clamp((h - abs(lz)) / RND, 0.0, 1.0)) * 0.7854; // side wall: 45 degrees at the rim, upright between
    n = vec3(g * cos(b), zs * sin(b));
  }
  n = dot(n, n) > 1e-8 ? normalize(n) : vec3(0.0, 0.0, zs);
  vec3 rd = normalize(v_w - u_cam);
  vec3 r = reflect(rd, n);
  float e = studio(r);
  float fr = pow(1.0 - max(dot(n, -rd), 0.0), 3.0);
  float dif = max(dot(n, normalize(vec3(-0.3, 0.8, 0.6))), 0.0);
  vec3 col = eye < 0.5
    ? u_bone * (0.012 + 0.04 * dif) + u_bone * e * 0.5 + u_hot * pow(e, 3.0) * 0.12 + u_bone * fr * 0.5
    : u_red * (0.04 + 0.22 * dif) + u_red * e * 1.15 + u_hot * pow(e, 3.0) * 0.3 + u_red * fr * 1.1;
  col = col / (1.0 + col);
  gl_FragColor = vec4(pow(col, vec3(1.0 / 2.2)) * u_alpha, u_alpha);
}`;

const linear = (hex, gain = 1) => { const n = parseInt(hex.slice(1), 16); return [16, 8, 0].map((s) => Math.pow(((n >> s) & 255) / 255, 2.2) * gain); };

// outlines + cap triangles -> front cap, back cap and side walls; interleaved x, y, face, kind, material
function extrude(mesh) {
  const v = [], idx = [];
  const vert = (x, y, z, kind, mat) => { v.push(x, y, z, kind, mat); return v.length / 5 - 1; };
  [mesh.body, mesh.eye].forEach((part, mat) => {
    const P = part.pts, n = P.length / 2, front = [], back = [];
    for (let i = 0; i < n; i++) { front.push(vert(P[2 * i], P[2 * i + 1], 1, 0, mat)); back.push(vert(P[2 * i], P[2 * i + 1], -1, 0, mat)); }
    for (let t = 0; t < part.tris.length; t += 3) {
      const [a, b, c] = part.tris.slice(t, t + 3);
      idx.push(front[a], front[b], front[c], back[a], back[c], back[b]);
    }
    let start = 0;
    for (const end of part.rings) {
      const f = [], k = [], m = end - start;
      for (let i = start; i < end; i++) { f.push(vert(P[2 * i], P[2 * i + 1], 1, 1, mat)); k.push(vert(P[2 * i], P[2 * i + 1], -1, 1, mat)); }
      for (let i = 0; i < m; i++) { const j = (i + 1) % m; idx.push(f[i], f[j], k[j], f[i], k[j], k[i]); }
      start = end;
    }
  });
  return { data: new Float32Array(v), index: new Uint16Array(idx) };
}

function createMark(canvas) {
  let gl;
  try { gl = canvas.getContext("webgl", { alpha: true, antialias: true, premultipliedAlpha: true, powerPreference: "high-performance" }); } catch { gl = null; }
  if (!gl) return null;
  const compile = (type, src) => { const s = gl.createShader(type); gl.shaderSource(s, src); gl.compileShader(s); return gl.getShaderParameter(s, gl.COMPILE_STATUS) ? s : null; };
  const vs = compile(gl.VERTEX_SHADER, VERT), fs = compile(gl.FRAGMENT_SHADER, FRAG);
  if (!vs || !fs) return null;
  const prog = gl.createProgram();
  gl.attachShader(prog, vs); gl.attachShader(prog, fs); gl.linkProgram(prog);
  if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) return null;
  gl.useProgram(prog);
  const u = {};
  for (const n of ["u_vp", "u_off", "u_depth", "u_eyez", "u_eyexy", "u_alpha", "u_cam", "u_sdf", "u_bone", "u_red", "u_hot"]) u[n] = gl.getUniformLocation(prog, n);
  const red = linear("#e0301e", 2.2);
  gl.uniform3fv(u.u_bone, linear("#ece6d6"));
  gl.uniform3fv(u.u_red, red);
  gl.uniform3f(u.u_hot, 1, 0.55 + red[1], 0.5 + red[2]);
  gl.uniform1i(u.u_sdf, 0);
  gl.enable(gl.DEPTH_TEST);
  let count = 0, hasTex = false;
  return {
    setTexture(img) {
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, gl.createTexture());
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGB, gl.RGB, gl.UNSIGNED_BYTE, img);
      for (const [k, v] of [[gl.TEXTURE_MIN_FILTER, gl.LINEAR], [gl.TEXTURE_MAG_FILTER, gl.LINEAR], [gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE], [gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE]]) gl.texParameteri(gl.TEXTURE_2D, k, v);
      hasTex = true;
    },
    setMesh(mesh) {
      const { data, index } = extrude(mesh);
      gl.bindBuffer(gl.ARRAY_BUFFER, gl.createBuffer());
      gl.bufferData(gl.ARRAY_BUFFER, data, gl.STATIC_DRAW);
      gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, gl.createBuffer());
      gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, index, gl.STATIC_DRAW);
      const at = (name, size, off) => { const a = gl.getAttribLocation(prog, name); gl.enableVertexAttribArray(a); gl.vertexAttribPointer(a, size, gl.FLOAT, false, 20, off * 4); };
      at("a_p", 3, 0);
      at("a_k", 2, 3);
      count = index.length;
    },
    draw(cam, depth, eyez, alpha, eyexy) {
      gl.viewport(0, 0, canvas.width, canvas.height);
      gl.clearColor(0, 0, 0, 0);
      gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
      if (!hasTex || !count || alpha <= 0) return;
      gl.uniformMatrix4fv(u.u_vp, false, cam.vp);
      gl.uniform2f(u.u_off, cam.ox, cam.oy);
      gl.uniform3fv(u.u_cam, cam.eye);
      gl.uniform1f(u.u_depth, depth);
      gl.uniform1f(u.u_eyez, eyez);
      gl.uniform2f(u.u_eyexy, eyexy[0], eyexy[1]);
      gl.uniform1f(u.u_alpha, alpha);
      gl.drawElements(gl.TRIANGLES, count, gl.UNSIGNED_SHORT, 0);
    },
    lost: () => gl.isContextLost(),
  };
}

// orbit camera around the mark -> view-projection matrix (column-major), eye position and screen offset
function orbit(k, W, H) {
  const pitch = Math.max(-1.45, Math.min(1.45, k.pitch)), tanf = Math.tan(FOV / 2);
  const dist = (RADIUS * H) / (k.size * Math.min(W, H) * tanf);
  const eye = [dist * Math.sin(k.yaw) * Math.cos(pitch), dist * Math.sin(pitch), dist * Math.cos(k.yaw) * Math.cos(pitch)];
  const back = eye.map((v) => v / dist);
  const rl = Math.hypot(back[2], back[0]) || 1;
  const right = [back[2] / rl, 0, -back[0] / rl];
  const up = [back[1] * right[2] - back[2] * right[1], back[2] * right[0] - back[0] * right[2], back[0] * right[1] - back[1] * right[0]];
  const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  const view = [right[0], up[0], back[0], 0, right[1], up[1], back[1], 0, right[2], up[2], back[2], 0, -dot(right, eye), -dot(up, eye), -dot(back, eye), 1];
  const near = Math.max(0.05, dist - 2.5), far = dist + 2.5, f = 1 / tanf;
  const proj = [f / (W / H), 0, 0, 0, 0, f, 0, 0, 0, 0, (far + near) / (near - far), -1, 0, 0, (2 * far * near) / (near - far), 0];
  const vp = new Float32Array(16);
  for (let c = 0; c < 4; c++) for (let r = 0; r < 4; r++) vp[c * 4 + r] = proj[r] * view[c * 4] + proj[4 + r] * view[c * 4 + 1] + proj[8 + r] * view[c * 4 + 2] + proj[12 + r] * view[c * 4 + 3];
  return { vp, eye, ox: k.ox, oy: k.oy };
}

// ================================================================== the specimen: loader, frames, morphs
function specimen() {
  const section = $("#top"), stage = $("#stage"), canvas = $("#logo3d"), glow = $("#glow"), flat = $("#logo2d");
  const loaderEl = $("#loader"), countEl = $("#ld-n"), stepEl = $("#ld-step"), barEl = $("#ld-bar"), nav = $(".nav"), outline = $("#ld-mark");
  const outlineLen = Math.ceil($("#oq-mark path").getTotalLength() + 50);
  outline.style.strokeDasharray = `${outlineLen} ${outlineLen}`;
  const navButtons = $$("#st-nav button");
  const items = $$("[data-sc]", stage).map((el) => ({ el, scene: +el.dataset.sc, fx: el.dataset.fx, delay: +el.dataset.d || 0, last: -1 }));
  const chars = $$("[data-ch]", $("#word"));
  const mark = createMark(canvas);
  if (!mark) root.classList.add("no-gl");

  // ---- assets: the distance field and the display face
  let markOk = !!mark, texReady = !mark, fontsReady = false;
  if (mark) {
    const img = new Image();
    Promise.all([
      new Promise((ok, no) => { img.onload = ok; img.onerror = no; img.src = "/web/oqvar-sdf.png"; }),
      fetch("/web/oqvar-mark.json?v=59e2226f").then((r) => (r.ok ? r.json() : Promise.reject(r.status))),
    ]).then(([, mesh]) => { mark.setTexture(img); mark.setMesh(mesh); })
      .catch(() => { markOk = false; root.classList.add("no-gl"); })
      .finally(() => { texReady = true; });
  }
  (document.fonts ? document.fonts.ready : Promise.resolve()).then(() => { fontsReady = true; measure(); });

  // ---- loader state
  const born = performance.now();
  let loading = root.classList.contains("is-loading");
  let shown = 0, introAt = loading ? -1 : born - 5000;
  if (loading) { history.scrollRestoration = "manual"; scrollTo(0, 0); }
  const MIN_LOAD = reduce ? 700 : 2800, MAX_LOAD = 5500;
  const STEPS = [[0, "Tracing the mark"], [0.3, "Pouring the chrome"], [0.52, "Extruding"], [0.78, "Placing the eye"], [0.99, "Ready"]];
  let doneAt = -1;
  const finishLoading = (now) => {
    if (!loading) return;
    loading = false; introAt = now;
    root.classList.remove("is-loading");
    root.classList.add("ready");
    loaderEl.classList.add("leaving");
    setTimeout(() => loaderEl.remove(), 1000);
  };
  setTimeout(() => finishLoading(performance.now()), MAX_LOAD + 1200);  // whatever happens, the page opens
  if (!loading) root.classList.add("ready");

  // ---- measuring; `quality` drops the render resolution on slow GPUs (CSS upscales the canvas)
  let W = 1, H = 1, nat = [], quality = 1, slow = 0;
  function measure() {
    const r = stage.getBoundingClientRect();
    W = Math.max(1, r.width); H = Math.max(1, r.height);
    const dpr = Math.min(window.devicePixelRatio || 1, 1.75, Math.sqrt(2.6e6 / (W * H))) * quality;
    const cw = Math.round(W * dpr), ch = Math.round(H * dpr);
    if (canvas.width !== cw || canvas.height !== ch) { canvas.width = cw; canvas.height = ch; }
    const saved = chars.map((c) => c.style.transform);
    chars.forEach((c) => (c.style.transform = "none"));
    nat = chars.map((c) => { const b = c.getBoundingClientRect(); return { x: b.left + b.width / 2 - r.left, y: b.top + b.height / 2 - r.top, w: b.width }; });
    chars.forEach((c, i) => (c.style.transform = saved[i]));
  }
  new ResizeObserver(measure).observe(stage);
  measure();

  // ---- input: pointer tilt and a drag to spin
  const pointer = { x: 0, y: 0, tx: 0, ty: 0 };
  let spin = 0, spinVel = 0, drag = null;
  stage.addEventListener("pointermove", (e) => {
    const r = stage.getBoundingClientRect();
    pointer.tx = ((e.clientX - r.left) / r.width) * 2 - 1;
    pointer.ty = ((e.clientY - r.top) / r.height) * 2 - 1;
    if (drag && drag.id === e.pointerId) { spinVel += (e.clientX - drag.x) * 0.0022; drag.x = e.clientX; }
  });
  stage.addEventListener("pointerdown", (e) => {
    if (e.pointerType !== "mouse" || e.button !== 0 || e.target.closest("a, button")) return;
    drag = { id: e.pointerId, x: e.clientX };
    stage.style.cursor = "grabbing";
  });
  addEventListener("pointerup", () => { drag = null; stage.style.cursor = ""; });

  // ---- frame nav
  const jump = (i) => {
    const travel = section.offsetHeight - stage.clientHeight;
    scrollTo({ top: section.offsetTop + (i / (SCENES - 1)) * travel, behavior: reduce ? "auto" : "smooth" });
  };
  navButtons.forEach((b) => b.addEventListener("click", () => jump(+b.dataset.go)));
  let activeScene = -1;

  const adapt = (real) => {
    if (quality <= 0.5) return;
    slow = real > 0.034 ? slow + 1 : Math.max(0, slow - 1);
    if (slow > 24) { quality = Math.max(0.5, quality * 0.75); slow = 0; measure(); }
  };

  // ---- the loop
  let raf = 0, running = false, visible = true, last = performance.now(), time = 0, prog = -1, lastSig = "";
  let failed = false;
  const frame = (now) => {
    raf = requestAnimationFrame(frame);
    try { step(now); } catch (err) {
      if (!failed) { failed = true; console.error("OQVAR hero:", err); }
      finishLoading(performance.now());
    }
  };
  const step = (now) => {
    const real = Math.min(0.5, (now - last) / 1000);  // easing and the loader follow the clock, not the frame rate
    const dt = Math.min(0.05, real);
    last = now;
    adapt(real);
    const moving = !reduce;
    if (moving) time += dt;

    // loader progress: time, the texture and the fonts; never longer than MAX_LOAD
    const elapsed = now - born;
    if (loading) {
      // the count runs on the clock, 00 -> 99; it only waits at 99 while the mark is still downloading
      const t = clamp01(elapsed / MIN_LOAD);
      shown = Math.min(1 - Math.pow(1 - t, 1.5), texReady && fontsReady || elapsed > MAX_LOAD ? 1 : 0.99);
      countEl.textContent = shown >= 1 ? "100" : String(Math.floor(shown * 100)).padStart(2, "0");
      barEl.style.setProperty("--p", shown.toFixed(4));
      stepEl.textContent = STEPS.filter(([at]) => shown >= at).pop()[1];
      if (shown >= 1 && doneAt < 0) doneAt = now;
      if (doneAt >= 0 && now - doneAt > 260) finishLoading(now);
    }
    const intro = introAt < 0 ? 0 : reduce ? 1 : clamp01((now - introAt) / 1300);

    // scroll -> frame coordinate
    const r = section.getBoundingClientRect();
    const p = progressFrom(r.top, r.height, H);
    prog = prog < 0 || reduce ? p : prog + (p - prog) * (1 - Math.exp(-real * 9));
    const coord = sceneCoord(prog, SCENES, HOLD);
    const scene = Math.round(coord);
    if (scene !== activeScene) {
      activeScene = scene;
      navButtons.forEach((b, i) => (i === scene ? b.setAttribute("aria-current", "step") : b.removeAttribute("aria-current")));
    }

    // camera: the loader key eases into the scroll key during the intro
    const tall = H > W * 1.05;
    let k = keyAt(coord, tall);
    const grow = loading ? shown : 1;
    if (intro < 1) k = mixKey({ ...LOAD_KEY, yaw: KEYS[0].yaw * smoothstep(0.5, 1, grow), pitch: LOAD_KEY.pitch * smoothstep(0.5, 1, grow) }, k, easeInOut(intro));
    pointer.x += (pointer.tx - pointer.x) * (1 - Math.exp(-real * 4));
    pointer.y += (pointer.ty - pointer.y) * (1 - Math.exp(-real * 4));
    spin += spinVel + (moving && !loading ? dt * 0.35 : 0);  // the mark keeps turning; dragging adds to it
    spinVel *= Math.exp(-real * 3);
    const px = moving ? pointer.x : 0, py = moving ? pointer.y : 0;
    const life = moving ? intro : 0;
    const swirl = -(1 - easeOut(intro)) * Math.PI * 2;  // a full turn as it flies from the loader into the cover
    const view = {
      ...k,
      yaw: k.yaw + spin * intro + swirl + px * 0.45 * life,
      pitch: k.pitch + (Math.sin(time * 0.8) * 0.1 + py * 0.25) * life,
      ox: k.ox + Math.sin(time * 0.55) * 0.01 * life,
      oy: k.oy + Math.sin(time * 1.3) * 0.02 * life,
    };
    const eyexy = [px * 0.015 * life, -py * 0.015 * life];  // the eye looks at the pointer

    // assembly while loading: the body extrudes out of the page, then the eye drops in
    const depth = lerp(0.004, DEPTH, easeInOut(clamp01((grow - 0.5) / 0.35)));
    const eyez = (1 - easeOutBack(clamp01((grow - 0.78) / 0.22))) * 0.85;
    const alpha = smoothstep(0.32, 0.62, grow);

    // the red haze and the flat fallback follow where the mark lands
    const minDim = Math.min(W, H);
    const gx = (0.5 + k.ox / 2) * W, gy = (0.5 - k.oy / 2) * H, gr = k.size * minDim * 1.25;
    if (loading || intro < 1) {
      const ow = 0.959 * k.size * minDim;  // the mark is 1.535 logo units wide; RADIUS (0.8) maps to size * minDim / 2
      outline.style.width = `${ow.toFixed(1)}px`;
      outline.style.transform = `translate(${(gx - ow / 2).toFixed(1)}px,${(gy - (ow * 1117) / 1114 / 2).toFixed(1)}px)`;
      outline.style.strokeDashoffset = String(Math.round(outlineLen * (1 - easeInOut(clamp01(grow / 0.45)))));
      outline.style.opacity = String(1 - smoothstep(0.5, 0.72, grow));
    }
    glow.style.transform = `translate(${(gx - gr).toFixed(1)}px,${(gy - gr).toFixed(1)}px)`;
    glow.style.width = glow.style.height = `${(gr * 2).toFixed(1)}px`;
    glow.style.opacity = Math.max(alpha, smoothstep(0, 0.3, grow) * 0.7).toFixed(3);
    const sig = [view.yaw, view.pitch, view.size, view.ox, view.oy, depth, eyez, alpha, ...eyexy, canvas.width, canvas.height].map((v) => v.toFixed(4)).join();
    if (markOk && !mark.lost()) { if (sig !== lastSig) mark.draw(orbit(view, W, H), depth, eyez, alpha, eyexy); lastSig = sig; }  // nothing moved: keep the last frame
    else {
      const s = k.size * minDim * 1.25;
      flat.style.width = `${s.toFixed(1)}px`;
      flat.style.transform = `translate(${(gx - s / 2).toFixed(1)}px,${(gy - s / 2).toFixed(1)}px) perspective(900px) rotateY(${(view.yaw * 57.3 % 360).toFixed(1)}deg) rotateX(${(view.pitch * -40).toFixed(1)}deg)`;
      flat.style.opacity = alpha.toFixed(3);
    }

    // the type
    for (const it of items) {
      let v = reveal(coord, it.scene, it.delay);
      if (it.scene === 0) v *= clamp01((intro * 1.5 - it.delay * 0.5) / 0.6);
      const q = Math.round(v * 500) / 500;
      if (q === it.last) continue;
      it.last = q;
      const s = it.el.style, dir = coord < it.scene ? 1 : -1;
      s.visibility = q <= 0 ? "hidden" : "visible";
      s.opacity = String(q);
      if (reduce || it.fx === "fade") continue;
      if (it.fx === "clip") {
        const hid = ((1 - q) * 100).toFixed(1);
        s.clipPath = dir > 0 ? `inset(0 0 ${hid}% 0)` : `inset(${hid}% 0 0 0)`;
        s.transform = `translateY(${((1 - q) * 0.4 * dir).toFixed(3)}em)`;
      } else {
        s.transform = `translateY(${((1 - q) * 34 * dir).toFixed(1)}px)`;
        s.filter = q > 0.995 ? "" : `blur(${((1 - q) * 8).toFixed(1)}px)`;
      }
    }

    // cover -> mark: O and R slide out to flank the mark, Q V A fall into it
    const m = easeInOut(clamp01(coord)), out = clamp01(coord - 1);
    const k1 = keyAt(1, tall), r1 = k1.size * minDim * 0.5, cy1 = (0.5 - k1.oy / 2) * H;
    chars.forEach((c, i) => {
      const n0 = nat[i];
      if (!n0) return;
      if (i === 0 || i === chars.length - 1) {
        const side = i === 0 ? -1 : 1;
        const tx = W / 2 + side * (r1 + n0.w * 0.62 + minDim * 0.03);
        c.style.transform = `translate(${((tx - n0.x) * m).toFixed(1)}px,${((cy1 - n0.y) * m - out * 60).toFixed(1)}px)`;
        c.style.opacity = String(coord < 1 ? 1 : 1 - smoothstep(0.1, 0.45, out));
        c.style.filter = reduce || out < 0.05 ? "" : `blur(${(out * 14).toFixed(1)}px)`;
      } else {
        c.style.transform = reduce ? "" : `translate(${((W / 2 - n0.x) * m * 0.75).toFixed(1)}px,${((cy1 - n0.y) * m).toFixed(1)}px) scale(${(1 - m * 0.7).toFixed(3)})`;
        c.style.opacity = String(1 - smoothstep(0, 0.4, m));
        c.style.filter = reduce || m < 0.02 ? "" : `blur(${(m * 12).toFixed(1)}px)`;
      }
      c.style.visibility = coord > 1.6 ? "hidden" : "";
    });
  };
  const start = () => { if (running || !visible || document.hidden) return; running = true; last = performance.now(); raf = requestAnimationFrame(frame); };
  const stop = () => { running = false; cancelAnimationFrame(raf); };
  new IntersectionObserver(([e]) => { visible = e.isIntersecting; visible ? start() : stop(); }).observe(section);
  document.addEventListener("visibilitychange", () => (document.hidden ? stop() : start()));
  start();

  // the header turns solid once the page moves
  const solid = () => nav.classList.toggle("solid", scrollY > 24);
  addEventListener("scroll", solid, { passive: true });
  solid();
}

// ================================================================== drawing types: accessible tabs
function tabs() {
  const list = $$('[role="tab"]');
  const select = (i, focus) => {
    list.forEach((t, j) => {
      const on = j === i;
      t.setAttribute("aria-selected", String(on));
      t.tabIndex = on ? 0 : -1;
      $("#" + t.getAttribute("aria-controls")).hidden = !on;
    });
    if (focus) list[i].focus();
  };
  list.forEach((t, i) => {
    t.addEventListener("click", () => select(i));
    t.addEventListener("keydown", (e) => {
      if (e.altKey || e.ctrlKey || e.metaKey || !["ArrowLeft", "ArrowRight", "Home", "End"].includes(e.key)) return;
      e.preventDefault();
      const n = list.length;
      select(e.key === "Home" ? 0 : e.key === "End" ? n - 1 : (i + (e.key === "ArrowRight" ? 1 : -1) + n) % n, true);
    });
  });
}

// ================================================================== benchmark: one series, value at the tip, hover/focus tooltip, table view
const el = (tag, cls, text) => { const n = document.createElement(tag); if (cls) n.className = cls; if (text != null) n.textContent = text; return n; };
const pct = (v) => `${v.toFixed(v >= 99 ? 2 : 1)}%`;

function renderBenchmark(data) {
  const plot = $("#bm-plot"), tbody = $("#bm-table tbody"), tip = $("#bm-tip"), fig = $("#bm");
  const groups = data.groups.map((g) => ({ ...g, rows: g.rows.filter((r) => r.kind !== "base") })).filter((g) => g.rows.length);
  if (data.generated) { $("#bm-gen").textContent = data.generated; $("#bm-count").hidden = false; }
  const showTip = (row, x, y) => {
    tip.replaceChildren(el("strong", null, pct(row.pct)), el("span", null, row.label), el("span", null, row.sub));
    tip.hidden = false;
    const box = fig.getBoundingClientRect(), w = tip.offsetWidth;
    tip.style.left = `${Math.min(Math.max(x - box.left + 14, 8), box.width - w - 8)}px`;
    tip.style.top = `${y - box.top + 14}px`;
  };
  const hideTip = () => { tip.hidden = true; };
  for (const g of groups) {
    plot.append(el("p", "bm-group", g.title));
    for (const r of g.rows) {
      const row = el("div", "bm-row");
      row.tabIndex = 0;
      row.setAttribute("aria-label", `${r.label}: ${pct(r.pct)} correct. ${r.sub}`);
      const lbl = el("div", "bm-lbl");
      lbl.append(el("b", null, r.label), el("span", null, r.sub));
      const track = el("div", "bm-track");
      track.style.setProperty("--w", `${r.pct}%`);
      track.append(el("span", "bm-bar"), el("span", `bm-val${r.pct > 82 ? " inside" : ""}`, pct(r.pct)));
      row.append(lbl, track);
      row.addEventListener("pointermove", (e) => showTip(r, e.clientX, e.clientY));
      row.addEventListener("pointerleave", hideTip);
      row.addEventListener("focus", () => { const b = track.getBoundingClientRect(); showTip(r, b.left + (b.width * r.pct) / 100, b.bottom); });
      row.addEventListener("blur", hideTip);
      plot.append(row);
      const tr = document.createElement("tr");
      tr.append(el("th", null, r.label), el("td", null, g.title), el("td", null, r.sub), el("td", "r", pct(r.pct)));
      tr.firstChild.scope = "row";
      tbody.append(tr);
    }
  }
  const axis = el("div", "bm-axis"), ticks = el("div", "bm-ticks");
  for (const t of [0, 25, 50, 75, 100]) { const s = el("span", null, `${t}%`); s.style.left = `${t}%`; ticks.append(s); }
  axis.append(el("span"), ticks);
  plot.append(axis);
  if (reduce || !("IntersectionObserver" in window)) fig.classList.add("in");
  else new IntersectionObserver((entries, obs) => { if (entries.some((e) => e.isIntersecting)) { fig.classList.add("in"); obs.disconnect(); } }, { threshold: 0.15 }).observe(fig);
  $("#bm-toggle").addEventListener("click", (e) => {
    const table = !$("#bm-table").hidden;
    $("#bm-table").hidden = table;
    plot.hidden = !table;
    hideTip();
    e.currentTarget.setAttribute("aria-expanded", String(!table));
    e.currentTarget.textContent = table ? "Show as table" : "Show as chart";
  });
}

try { specimen(); } catch (err) {  // never leave the page locked behind the loader
  console.error("OQVAR hero:", err);
  root.classList.remove("is-loading");
  root.classList.add("ready", "no-gl");
}
tabs();
fetch("/web/benchmark.json").then((r) => (r.ok ? r.json() : Promise.reject(r.status))).then(renderBenchmark).catch(() => {
  $("#bm-plot").hidden = true;
  $("#bm-toggle").hidden = true;
  $("#bm-empty").hidden = false;
});
fetch("/api/config").then((r) => (r.ok ? r.json() : null)).then((c) => {
  if (c && c.drawings) $("#cv-live").textContent = `${c.drawings.toLocaleString("en-US")} drawings in the register`;
}).catch(() => {});
})();

// Blueprint3D viewer: one scene JSON -> linked 2D plan (SVG) + 3D model (three.js).
// Every element is keyed by the same ID in both views, so a 3D object always traces back to its 2D source.
//   DXF scenes: 3D is extruded from plan polygons (heights assumed, shown on screen).
//   IFC scenes: 3D is the BIM model itself (web-ifc); the 2D storey plan is projected from that geometry.
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { GLTFExporter } from "three/addons/exporters/GLTFExporter.js";

const DATASET_ROOT = "../../junction-dataset/junction-dataset/"; // served from the repo root
const WEB_IFC = "https://cdn.jsdelivr.net/npm/web-ifc@0.0.57/";
const KIND_COLORS = {
  bedroom: 0x8fb8de, living: 0xe8c48a, kitchen: 0xe59a7a, bath: 0x86d0c7, corridor: 0xcfcac0,
  storage: 0xb8aed0, outdoor: 0xa8cf8f, stair: 0xa0a4ab, other: 0xd7d3cb,
};
const EMBED = new URLSearchParams(location.search).has("embed");  // shown inside the Plumbline app: paper theme, 3D pane only
const HL = new THREE.Color(EMBED ? 0xffd21a : 0xffb020);
const $ = (s) => document.querySelector(s);
const hex = (c) => "#" + c.toString(16).padStart(6, "0");

let scene3, camera, renderer, controls, root, S = null, loadToken = 0;
const byId = new Map();      // id -> { el, group, meshes: [], svg: [] }
let highlighted = new Set();
let xray = false, isolate = false, storey = null;
let webIfc = null;

// ------------------------------------------------------------------ boot (called at the end of this module)
if (EMBED) document.body.classList.add("embed");
// stoa_qa answers carry IFC entity numbers; IFC scenes key their elements as "#<number>"
const sceneId = (i) => (byId.has(i) ? i : byId.has("#" + i) ? "#" + i : null);
const notifyHost = () => { if (window.parent !== window) window.parent.postMessage({ type: "plumbline:ready", name: $("#scene").value }, location.origin); };
window.addEventListener("message", (e) => {
  if (e.origin !== location.origin || e.data?.type !== "plumbline:highlight") return;
  const ids = (e.data.ids || []).map(sceneId).filter((i) => i != null);
  select(ids, ids.length ? `<b style="color:var(--text)">Evidence:</b> ${ids.length} element(s) highlighted.` : "");
});

async function boot() {
  init3D();
  const index = await (await fetch("scenes/index.json")).json();
  for (const s of index) {
    const fmt = s.source.endsWith(".ifc") ? "IFC" : /\.(png|jpe?g)$/i.test(s.source) ? "RASTER" : "DXF";
    $("#scene").add(new Option(`[${fmt}] ${s.name}  (${s.counts.rooms} rooms)`, s.name));
  }
  $("#scene").onchange = () => load($("#scene").value);
  $("#storey").onchange = () => { storey = +$("#storey").value; applyStorey(); };
  const fromHash = decodeURIComponent(location.hash.slice(1));
  await load(index.some((s) => s.name === fromHash) ? fromHash : index[0].name);
  notifyHost();
}

async function load(name) {
  const token = ++loadToken;
  $("#scene").value = name; location.hash = name;
  S = await (await fetch(`scenes/${name}.json`)).json();
  byId.clear(); highlighted.clear();
  $("#info").textContent = "";
  const ifc = S.source.format === "ifc";
  document.body.classList.toggle("ifc", ifc);
  if (ifc) return loadIFC(token);
  for (const g of ["rooms", "walls", "doors", "windows", "units", "fixtures"])
    for (const el of S[g]) byId.set(el.id, { el, group: g, meshes: [], svg: [] });
  build2D(); build3D();
  const a = S.assumptions;
  const raster = S.source.format === "raster";
  $("#assume").textContent = raster
    ? `APPROXIMATE · traced from a raster image · scale from ${S.source.scale_from} · storey ${a.storey_height} m assumed`
    : `Assumed heights · storey ${a.storey_height} m · door head ${a.door_head} m · window ${a.window_sill}–${a.window_head} m`;
  $("#orig img").src = DATASET_ROOT + S.source.file.replace(/\.dxf$/i, ".png");
  $("#answer").innerHTML = `<b>${S.source.file}</b><div class="method">${S.counts.rooms} rooms · ${S.counts.doors} doors · ${S.counts.windows} windows${raster ? "" : ` · ${S.counts.units} apartments`} — ${raster ? `${S.source.engine === "cubicasa5k-model" ? "detected by the CubiCasa5K network" : "traced with OpenCV"} from the scanned image (approximate; compare with Original drawing)` : "all extracted from DXF layers"}. Ask a question or click an element.</div>`;
}

// ------------------------------------------------------------------ DXF: 2D (SVG, y flipped)
const NS = "http://www.w3.org/2000/svg";
function svgAdd(svg, tag, attrs, id) {
  const e = document.createElementNS(NS, tag);
  for (const k in attrs) e.setAttribute(k, attrs[k]);
  svg.appendChild(e);
  if (id && byId.has(id)) { byId.get(id).svg.push(e); e.dataset.id = id; }
  return e;
}

function build2D() {
  const svg = $("#plan"); svg.innerHTML = "";
  const [x0, y0, x1, y1] = S.bbox, pad = 1;
  setupPanZoom(svg, [x0 - pad, -y1 - pad, x1 - x0 + 2 * pad, y1 - y0 + 2 * pad]);
  const pts = (p) => p.map(([x, y]) => `${x},${-y}`).join(" ");
  const path = (p, holes = []) => [p, ...holes].map((r) => "M" + r.map(([x, y]) => `${x},${-y}`).join("L") + "Z").join("");
  const add = (tag, attrs, id) => svgAdd(svg, tag, attrs, id);
  for (const r of S.rooms)
    add("path", { d: path(r.poly, r.holes), class: "room", "fill-rule": "evenodd", fill: hex(KIND_COLORS[r.kind]) + "66", stroke: "#999", "stroke-width": .02 }, r.id);
  for (const f of S.fixtures) add("polygon", { points: pts(f.poly), fill: f.label ? "#ffffffcc" : "none", stroke: "#888", "stroke-width": .015 }, f.id);
  for (const w of S.walls) add("path", { d: path(w.poly, w.holes ?? []), "fill-rule": "evenodd", fill: "#2b2f36", stroke: "#2b2f36", "stroke-width": .01 }, w.id);
  for (const w of S.windows) add("polygon", { points: pts(w.poly), fill: "#cfe8ff", stroke: "#3a6ea5", "stroke-width": .025 }, w.id);
  for (const d of S.doors) {
    add("polygon", { points: pts(d.leaf), fill: "#fff", stroke: "#444", "stroke-width": .02 }, d.id);
    if (d.opening) add("polyline", { points: pts(d.opening), fill: "none", stroke: "#444", "stroke-width": .015, "stroke-dasharray": ".06 .05" }, d.id);
  }
  for (const r of S.rooms) {
    const [x, y] = r.label_at;
    add("text", { x, y: -y - .08, class: "room-label" }).textContent = r.label || r.number;
    add("text", { x, y: -y + .26, class: "room-area" }).textContent = `${r.number ?? ""} · ${r.area_m2} m²`;
  }
}

function setupPanZoom(svg, vb0) {
  let vb = [...vb0];
  const setVB = () => svg.setAttribute("viewBox", vb.join(" "));
  setVB();
  svg.onclick = (e) => { const id = e.target.dataset?.id; if (id) select([id], describe(id)); };
  svg.onmousemove = (e) => tip(e, e.target.dataset?.id);
  svg.onmouseleave = () => tip(null);
  let drag = null;
  svg.onwheel = (e) => {
    e.preventDefault();
    const k = e.deltaY > 0 ? 1.15 : 1 / 1.15, r = svg.getBoundingClientRect();
    const mx = vb[0] + (e.clientX - r.left) / r.width * vb[2], my = vb[1] + (e.clientY - r.top) / r.height * vb[3];
    vb = [mx - (mx - vb[0]) * k, my - (my - vb[1]) * k, vb[2] * k, vb[3] * k]; setVB();
  };
  svg.onpointerdown = (e) => { drag = { x: e.clientX, y: e.clientY, vb: [...vb] }; };
  window.onpointerup = () => { drag = null; };
  svg.onpointermove = (e) => {
    if (!drag) return;
    const r = svg.getBoundingClientRect(), s = Math.max(vb[2] / r.width, vb[3] / r.height);
    vb[0] = drag.vb[0] - (e.clientX - drag.x) * s; vb[1] = drag.vb[1] - (e.clientY - drag.y) * s; setVB();
  };
}

// ------------------------------------------------------------------ 3D scene
function init3D() {
  const canvas = $("#three");
  renderer = new THREE.WebGLRenderer({ canvas, antialias: true, preserveDrawingBuffer: true });
  renderer.setPixelRatio(devicePixelRatio);
  scene3 = new THREE.Scene();
  scene3.background = new THREE.Color(EMBED ? 0xe7e2d5 : 0x1a1f26);
  camera = new THREE.PerspectiveCamera(45, 1, 0.1, 5000);
  controls = new OrbitControls(camera, canvas);
  controls.enableDamping = true;
  scene3.add(new THREE.HemisphereLight(0xffffff, 0x445566, 1.6));
  const sun = new THREE.DirectionalLight(0xffffff, 1.4); sun.position.set(20, 40, 15); scene3.add(sun);
  const resize = () => {
    const r = canvas.parentElement.getBoundingClientRect();
    renderer.setSize(r.width, r.height, false); camera.aspect = r.width / r.height; camera.updateProjectionMatrix();
  };
  new ResizeObserver(resize).observe(canvas.parentElement); resize();
  const ray = new THREE.Raycaster(), ptr = new THREE.Vector2();
  const pick = (e) => {
    const r = canvas.getBoundingClientRect();
    ptr.set(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1);
    ray.setFromCamera(ptr, camera);
    const hits = ray.intersectObjects(root ? root.children : [], false).filter((h) => h.object.visible && h.object.userData.id);
    // rooms are see-through volumes: prefer a solid element in front, otherwise the room
    const solid = hits.find((h) => h.object.userData.group !== "rooms" && !(xray && h.object.userData.wallLike));
    const room = hits.find((h) => h.object.userData.group === "rooms");
    return (xray ? room ?? solid : solid ?? room)?.object.userData.id;
  };
  let down = null;
  canvas.addEventListener("pointerdown", (e) => (down = [e.clientX, e.clientY]));
  canvas.addEventListener("pointerup", (e) => {
    if (down && Math.hypot(e.clientX - down[0], e.clientY - down[1]) < 4) { const id = pick(e); if (id) select([id], describe(id)); }
  });
  canvas.addEventListener("mousemove", (e) => tip(e, pick(e)));
  canvas.addEventListener("mouseleave", () => tip(null));
  (function loop() { controls.update(); renderer.render(scene3, camera); requestAnimationFrame(loop); })();
}

function newRoot() {
  if (root) { scene3.remove(root); root.traverse((o) => { o.geometry?.dispose(); }); }
  root = new THREE.Group(); root.name = S.source.file; scene3.add(root);
}

function shapeOf(poly, holes = []) {
  const s = new THREE.Shape(poly.map(([x, y]) => new THREE.Vector2(x, y)));
  for (const h of holes) s.holes.push(new THREE.Path(h.map(([x, y]) => new THREE.Vector2(x, y))));
  return s;
}

// plan (x, y) -> world (x, up, -y); extrude from z0 to z1
function prism(poly, z0, z1, material, holes = []) {
  const g = new THREE.ExtrudeGeometry(shapeOf(poly, holes), { depth: z1 - z0, bevelEnabled: false });
  g.rotateX(-Math.PI / 2); g.translate(0, z0, 0);
  const m = new THREE.Mesh(g, material);
  m.add(new THREE.LineSegments(new THREE.EdgesGeometry(g, 30), new THREE.LineBasicMaterial({ color: 0x30363f, transparent: true, opacity: .55 })));
  return m;
}

function register(mesh, id, group, wallLike = false) {
  mesh.name = id;
  mesh.userData = { id, group, wallLike, ...meta(id) };
  mesh.userData.baseColor = mesh.material.color.clone();
  byId.get(id).meshes.push(mesh); root.add(mesh);
}

function meta(id) {
  const { el } = byId.get(id);
  const out = { layer: el.layer, source: S.source.file };
  for (const k of ["number", "label", "kind", "area_m2", "unit", "width_m", "name", "net_floor_area_m2", "length_m", "net_volume_m3"])
    if (el[k] != null) out[k] = el[k];
  return out;
}

function build3D() {
  newRoot();
  const A = S.assumptions, H = A.storey_height;
  const mat = (c, o = {}) => new THREE.MeshStandardMaterial({ color: c, roughness: .85, ...o });
  const wallMat = mat(0xe9ebee);
  for (const r of S.rooms) register(prism(r.poly, 0, 0.04, mat(KIND_COLORS[r.kind] ?? KIND_COLORS.other), r.holes), r.id, "rooms");
  for (const w of S.walls) register(prism(w.poly, 0, H, wallMat, w.holes ?? []), w.id, "walls", true);
  for (const w of S.windows) {
    for (const [z0, z1] of [[0, A.window_sill], [A.window_head, H]]) register(prism(w.poly, z0, z1, wallMat), w.id, "windows", true);
    register(prism(w.poly, A.window_sill, A.window_head, mat(0x9fd3ff, { transparent: true, opacity: .45, roughness: .1 })), w.id, "windows");
  }
  for (const d of S.doors) {
    if (d.lintel) register(prism(d.lintel, A.door_head, H, wallMat), d.id, "doors", true);
    register(prism(d.leaf, 0, A.door_head, mat(0xb07a4f)), d.id, "doors");
  }
  for (const f of S.fixtures) register(prism(f.poly, 0, A.fixture_height, mat(0xf4f4f4)), f.id, "fixtures");
  applyXray(); frame();
}

// Union of boxes, ignoring stray objects far from the bulk of the model (some IFC files place spaces off-site)
function robustBox(boxes) {
  boxes = boxes.filter((b) => !b.isEmpty());
  const out = new THREE.Box3();
  if (!boxes.length) return out;
  const cs = boxes.map((b) => b.getCenter(new THREE.Vector3()));
  const q = (axis, p) => { const v = cs.map((c) => c[axis]).sort((a, b) => a - b); return v[Math.min(v.length - 1, Math.floor(p * v.length))]; };
  const lim = {};
  for (const ax of ["x", "y", "z"]) {
    const lo = q(ax, .05), hi = q(ax, .95), pad = Math.max((hi - lo) * .5, 2);
    lim[ax] = [lo - pad, hi + pad];
  }
  boxes.forEach((b, i) => { if (["x", "y", "z"].every((ax) => cs[i][ax] >= lim[ax][0] && cs[i][ax] <= lim[ax][1])) out.union(b); });
  return out;
}

function frame(top = false) {
  const box = robustBox(root.children.filter((m) => m.visible).map((m) => new THREE.Box3().setFromObject(m)));
  if (box.isEmpty()) return;
  const c = box.getCenter(new THREE.Vector3()), size = box.getSize(new THREE.Vector3());
  const span = Math.max(size.x, size.z, size.y * 1.5);
  controls.target.copy(c);
  if (top) camera.position.set(c.x, c.y + span * 1.25, c.z + .01);
  else camera.position.set(c.x + span * .55, c.y + span * .6, c.z + span * .75);
  camera.near = span / 1000; camera.far = span * 20; camera.updateProjectionMatrix();
  controls.update();
}

function applyXray() {
  for (const m of root.children) {
    if (!m.userData.wallLike) continue;
    ownMaterial(m);
    m.material.transparent = xray || m.userData.baseOpacity < 1;
    m.material.opacity = xray ? .12 : (m.userData.baseOpacity ?? 1);
    m.material.depthWrite = !xray; m.material.needsUpdate = true;
    for (const c of m.children) c.visible = !xray;
  }
}

function ownMaterial(m) {
  if (!m.material.userData.own) { m.material = m.material.clone(); m.material.userData.own = true; }
}

// ------------------------------------------------------------------ IFC: real BIM geometry via web-ifc
const IFC_STYLE = {
  rooms: { color: 0x7fb2e5, opacity: .22 }, walls: { color: 0xe9ebee }, slabs: { color: 0xc9ccd1 },
  roofs: { color: 0xb5654a }, doors: { color: 0xb07a4f }, windows: { color: 0x9fd3ff, opacity: .5 },
  stairs: { color: 0xa0a4ab }, columns: { color: 0xd0d3d8 }, beams: { color: 0xd0d3d8 },
  railings: { color: 0x8a8f96 }, furniture: { color: 0xf2efe8 }, sanitary: { color: 0xffffff },
};
const WALL_LIKE = new Set(["walls", "slabs", "roofs", "columns", "beams", "railings", "other"]);

async function loadIFC(token) {
  const storeyName = Object.fromEntries(S.storeys.map((s) => [s.id, s.name]));
  for (const r of S.rooms) {
    byId.set("#" + r.id, { el: { ...r, eid: r.id, id: "#" + r.id, layer: "IFCSPACE", storey_name: storeyName[r.storey] }, group: "rooms", meshes: [], svg: [] });
  }
  for (const e of S.elements) {
    byId.set("#" + e.id, { el: { ...e, eid: e.id, id: "#" + e.id, layer: e.ifc_type, storey_name: storeyName[e.storey] }, group: e.group, meshes: [], svg: [] });
  }
  const otherStorey = new Map();
  for (const [s, ids] of Object.entries(S.other_by_storey ?? {})) for (const i of ids) otherStorey.set(i, +s);

  // storey picker: default to the storey with the most rooms
  const sel = $("#storey"); sel.innerHTML = "";
  const roomsOn = (s) => S.rooms.filter((r) => r.storey === s.id).length;
  for (const s of S.storeys) sel.add(new Option(`${s.name}${s.elevation_m != null ? ` (${s.elevation_m} m)` : ""}`, s.id));
  storey = [...S.storeys].sort((a, b) => roomsOn(b) - roomsOn(a))[0]?.id ?? null;
  sel.value = storey;

  $("#assume").textContent = "Geometry, storeys and quantities read from the IFC model — nothing assumed.";
  $("#answer").innerHTML = `<b>${S.source.file}</b><div class="method">Loading IFC geometry…</div>`;
  $("#plan").innerHTML = "";

  if (!webIfc) {
    const W = await import(WEB_IFC + "web-ifc-api.js");
    const api = new W.IfcAPI(); api.SetWasmPath(WEB_IFC, true); await api.Init();
    webIfc = { W, api };
  }
  const { W, api } = webIfc;
  const buf = new Uint8Array(await (await fetch(DATASET_ROOT + S.source.file)).arrayBuffer());
  if (token !== loadToken) return;
  const model = api.OpenModel(buf, { COORDINATE_TO_ORIGIN: true });
  newRoot();
  const mats = new Map();
  const matFor = (group, c) => {
    const st = IFC_STYLE[group];
    const color = st ? st.color : new THREE.Color(c.x, c.y, c.z).getHex();
    const opacity = st?.opacity ?? (c.w < 1 ? c.w : 1);
    const key = color + ":" + opacity;
    if (!mats.has(key)) mats.set(key, new THREE.MeshStandardMaterial({ color, roughness: .85, transparent: opacity < 1, opacity, depthWrite: opacity >= .5, side: THREE.DoubleSide }));
    return mats.get(key);
  };
  const addMesh = (fm) => {
    const id = "#" + fm.expressID, rec = byId.get(id), group = rec?.group ?? "other";
    const pg = fm.geometries;
    for (let i = 0; i < pg.size(); i++) {
      const p = pg.get(i), g = api.GetGeometry(model, p.geometryExpressID);
      // copy out of WASM memory: these views are invalidated by later allocations
      const v = new Float32Array(api.GetVertexArray(g.GetVertexData(), g.GetVertexDataSize()));
      const ix = new Uint32Array(api.GetIndexArray(g.GetIndexData(), g.GetIndexDataSize()));
      const geo = new THREE.BufferGeometry();
      const inter = new THREE.InterleavedBuffer(v, 6);
      geo.setAttribute("position", new THREE.InterleavedBufferAttribute(inter, 3, 0));
      geo.setAttribute("normal", new THREE.InterleavedBufferAttribute(inter, 3, 3));
      geo.setIndex(new THREE.BufferAttribute(ix, 1));
      geo.applyMatrix4(new THREE.Matrix4().fromArray(p.flatTransformation));
      g.delete();
      const mesh = new THREE.Mesh(geo, matFor(group, p.color));
      mesh.userData.baseOpacity = mesh.material.opacity;
      if (rec) register(mesh, id, group, WALL_LIKE.has(group));
      else { mesh.userData = { group: "other", wallLike: true, storey: otherStorey.get(fm.expressID), baseOpacity: mesh.material.opacity }; mesh.userData.baseColor = mesh.material.color.clone(); root.add(mesh); }
    }
  };
  api.StreamAllMeshes(model, addMesh);
  api.StreamAllMeshesWithTypes(model, [W.IFCSPACE], addMesh);
  api.CloseModel(model);
  if (token !== loadToken) return;
  for (const m of root.children) if (m.userData.group === "rooms") m.renderOrder = 2;

  const c = S.counts;
  $("#answer").innerHTML = `<b>${S.source.file}</b><div class="method">${S.storeys.length} storeys · ${c.rooms} rooms · ${c.walls ?? 0} walls · ${c.doors ?? 0} doors · ${c.windows ?? 0} windows — geometry by web-ifc, facts from the IFC entities. Ask a question or click an element.</div>`;
  applyStorey(); applyXray(); frame();
}

function storeyOf(m) {
  const rec = m.userData.id && byId.get(m.userData.id);
  return rec ? rec.el.storey : m.userData.storey;
}

function applyStorey() {
  if (S?.source.format !== "ifc" || !root) return;
  const s = S.storeys.find((x) => x.id === storey);
  for (const m of root.children) m.visible = !isolate || storeyOf(m) === storey;
  $("#orig img").src = s?.plan ? DATASET_ROOT + "data/ifc_models/" + s.plan : "";
  build2DFromIFC();
  if (isolate) frame();
}

// 2D storey plan = plan projection of the storey's 3D elements (horizontal faces only); same IDs as 3D
function build2DFromIFC() {
  const svg = $("#plan"); svg.innerHTML = "";
  for (const rec of byId.values()) rec.svg = [];
  const order = ["rooms", "slabs", "furniture", "sanitary", "stairs", "walls", "columns", "railings", "windows", "doors"];
  const style = {
    rooms: (id) => ({ fill: "#8fb8de55", stroke: "#6a8fb5", "stroke-width": .03, class: "room" }),
    walls: () => ({ fill: "#2b2f36", stroke: "#2b2f36", "stroke-width": .01 }),
    columns: () => ({ fill: "#2b2f36" }), windows: () => ({ fill: "#cfe8ff", stroke: "#3a6ea5", "stroke-width": .02 }),
    doors: () => ({ fill: "#e9d2bd", stroke: "#7a4e2c", "stroke-width": .02 }), stairs: () => ({ fill: "#d4d6da", stroke: "#666", "stroke-width": .015 }),
    furniture: () => ({ fill: "none", stroke: "#aaa", "stroke-width": .015 }), sanitary: () => ({ fill: "#fff", stroke: "#888", "stroke-width": .015 }),
    railings: () => ({ fill: "#888" }),
  };
  const boxes = [], v = new THREE.Vector3(), a = new THREE.Vector3(), b = new THREE.Vector3(), c = new THREE.Vector3(), n = new THREE.Vector3();
  const r3 = (x) => Math.round(x * 1000) / 1000;
  const byGroup = {};
  for (const m of root.children) {
    const g = m.userData.group;
    if (!m.userData.id || storeyOf(m) !== storey || !style[g]) continue;
    (byGroup[g] ||= []).push(m);
  }
  for (const g of order) for (const m of byGroup[g] ?? []) {
    const pos = m.geometry.attributes.position, idx = m.geometry.index;
    const box = new THREE.Box3(); boxes.push(box);
    let d = "";
    for (let t = 0; t < idx.count; t += 3) {
      a.fromBufferAttribute(pos, idx.getX(t)); b.fromBufferAttribute(pos, idx.getX(t + 1)); c.fromBufferAttribute(pos, idx.getX(t + 2));
      n.subVectors(b, a).cross(v.subVectors(c, a));
      if (n.y < .5 * n.length()) continue; // keep upward faces only: one consistent layer forms the footprint
      d += `M${r3(a.x)},${r3(a.z)}L${r3(b.x)},${r3(b.z)}L${r3(c.x)},${r3(c.z)}Z`;
      box.expandByPoint(a).expandByPoint(b).expandByPoint(c);
    }
    if (d) {
      const st = style[g](m.userData.id);
      if (g === "rooms") { st.stroke = "none"; }
      svgAdd(svg, "path", { d, ...st }, m.userData.id);
    }
  }
  // room labels at the centre of each room's footprint
  for (const m of byGroup.rooms ?? []) {
    const bb = new THREE.Box3().setFromObject(m), ctr = bb.getCenter(new THREE.Vector3()), el = byId.get(m.userData.id).el;
    const size = Math.min(bb.max.x - bb.min.x, bb.max.z - bb.min.z);
    const fs = Math.max(.12, Math.min(.35, size / 8));
    const t = svgAdd(svg, "text", { x: ctr.x, y: ctr.z, class: "room-label", style: `font-size:${fs}px` });
    t.textContent = el.long_name || el.name || "";
    if (el.net_floor_area_m2 != null) svgAdd(svg, "text", { x: ctr.x, y: ctr.z + fs * 1.1, class: "room-area", style: `font-size:${fs * .8}px` }).textContent = `${el.name ?? ""} · ${el.net_floor_area_m2.toFixed(2)} m²`;
  }
  const box = robustBox(boxes);
  if (box.isEmpty()) { svg.innerHTML = `<text x="0" y="0" font-size="1" fill="#888">No geometry on this storey</text>`; svg.setAttribute("viewBox", "-1 -2 20 4"); return; }
  const pad = 1;
  setupPanZoom(svg, [box.min.x - pad, box.min.z - pad, box.max.x - box.min.x + 2 * pad, box.max.z - box.min.z + 2 * pad]);
  for (const id of highlighted) paint(id, true);
}

// ------------------------------------------------------------------ selection / evidence
function select(ids, html) {
  for (const id of highlighted) paint(id, false);
  highlighted = new Set(ids);
  for (const id of highlighted) paint(id, true);
  if (html) $("#info").innerHTML = html;
  // jump to the storey of the first IFC element so the 2D plan shows the evidence
  if (S.source.format === "ifc" && ids.length) {
    const st = byId.get(ids[0])?.el.storey;
    if (st != null && st !== storey && ids.every((i) => byId.get(i)?.el.storey === st)) { storey = st; $("#storey").value = st; applyStorey(); }
  }
}

function paint(id, on) {
  const rec = byId.get(id); if (!rec) return;
  for (const e of rec.svg) e.classList.toggle("hl", on);
  for (const m of rec.meshes) {
    ownMaterial(m);
    m.material.color.copy(on ? HL : m.userData.baseColor);
    m.material.emissive = new THREE.Color(on ? 0x553300 : 0x000000);
    if (rec.group === "rooms" && S.source.format === "ifc") { m.material.opacity = on ? .65 : (m.userData.baseOpacity ?? .22); m.material.needsUpdate = true; }
  }
}

function describe(id) {
  const { el, group } = byId.get(id);
  const rows = [["ID", id], ["Source", el.layer], ["File", S.source.file]];
  if (S.source.format === "ifc") {
    if (el.name) rows.splice(1, 0, ["Name", el.long_name ? `${el.long_name} (${el.name})` : el.name]);
    if (el.storey_name) rows.push(["Storey", el.storey_name]);
    if (group === "rooms") rows.push(["Net floor area", el.net_floor_area_m2 != null ? `${el.net_floor_area_m2.toFixed(2)} m² (from ${el.area_source})` : "not available"]);
    if (el.length_m != null) rows.push(["Length", `${el.length_m} m`]);
    if (el.net_side_area_m2 != null) rows.push(["Net side area", `${el.net_side_area_m2} m²`]);
    if (el.net_volume_m3 != null) rows.push(["Net volume", `${el.net_volume_m3} m³`]);
  } else {
    if (group === "rooms") {
      rows.splice(1, 0, ["Room", `${el.number ?? "—"} ${el.label ?? ""}`], ["Net area", `${el.area_m2} m² (computed from polygon${el.holes.length ? ", minus " + el.holes.length + " hole(s)" : ""})`]);
      if (el.printed_area) rows.push(["Printed on drawing", el.printed_area]);
      if (el.unit) rows.push(["Apartment", byId.get(el.unit)?.el.label ?? el.unit]);
    }
    if (group === "doors" && el.width_m) rows.splice(1, 0, ["Door", `clear opening ${el.width_m} m`]);
    if (group === "fixtures" && el.label) rows.splice(1, 0, ["Object", el.label]);
    if (S.source.format === "raster") rows.push(["Note", "Approximate: detected on a raster image"]);
  }
  return rows.map(([k, v]) => `<div><b style="color:var(--text)">${k}:</b> ${v}</div>`).join("");
}

function tip(e, id) {
  const t = $("#tip");
  if (!e || !id || !byId.has(id)) { t.style.display = "none"; return; }
  const { el, group } = byId.get(id);
  t.textContent = S.source.format === "ifc"
    ? (group === "rooms" ? `${el.label ?? id}${el.net_floor_area_m2 != null ? " · " + el.net_floor_area_m2.toFixed(2) + " m²" : ""}` : `${el.layer} ${el.name ?? ""}`)
    : group === "rooms" ? `${el.number ?? ""} ${el.label ?? ""} · ${el.area_m2} m²` : el.label ? `${el.label} · ${id}` : `${group.slice(0, -1)} · ${id}`;
  t.style.display = "block"; t.style.left = e.clientX + 12 + "px"; t.style.top = e.clientY + 12 + "px";
}

// ------------------------------------------------------------------ deterministic Q&A (an LLM router plugs in here)
const norm = (s) => (s ?? "").toLowerCase().normalize("NFC");
const quoted = (t) => [...t.matchAll(/['"‘’“”«»„]\s*([^'"‘’“”«»„]+?)\s*['"‘’“”«»„]/g)].map((m) => m[1]);

function answer(q) {
  return S.source.format === "ifc" ? answerIFC(q) : answerDXF(q);
}

function answerDXF(q) {
  const t = norm(q), R = S.rooms;
  const indoor = (r) => r.number && !r.outdoor;
  const sum = (rs) => Math.round(rs.reduce((a, r) => a + r.area_m2, 0) * 100) / 100;
  let m;
  if (/how many|count|wie viele|combien|quanti|kuinka monta/.test(t)) {
    const src = S.source.engine === "cubicasa5k-model" ? "CubiCasa5K network detections (approximate)" : S.source.format === "raster" ? "gaps traced with OpenCV (approximate)" : null;
    if (/door|tür|tuer|porte|ovi/.test(t)) return res(S.doors, `${S.doors.length} doors`, src ? `Counted ${src}.` : "Counted closed door-leaf polylines on DXF layer DOOR.");
    if (/window|fenster|fenêtre|finestr|ikkun/.test(t)) return res(S.windows, `${S.windows.length} windows`, src ? `Counted ${src}.` : "Counted closed polylines on DXF layer WINDOW.");
    if (S.source.engine === "cubicasa5k-model") {
      const fx = (label) => S.fixtures.filter((f) => f.label === label);
      const rk = (kind) => R.filter((r) => r.kind === kind);
      const how = "CubiCasa5K network detection (approximate) on the raster plan.";
      if (/wc room|wc-tilo|erillis/.test(t)) return none();
      if (/toilet|wc-istuin|wc istuin/.test(t)) return res(fx("Toilet"), `${fx("Toilet").length} toilets`, how);
      if (/sink|basin|lavuaari|pesuallas/.test(t)) return res(fx("Sink"), `${fx("Sink").length} sinks`, how);
      if (/bathtub|shower|amme|suihku/.test(t)) return res(fx("Bathtub"), `${fx("Bathtub").length} showers/bathtubs`, how + " Showers have no separate class in the model.");
      if (/sauna/.test(t)) return res(fx("Sauna Bench"), `${S.counts.saunas} saunas`, how + " Counted as clusters of sauna benches.");
      if (/bedroom|makuuhuone|\bmh\b/.test(t)) return res(rk("bedroom"), `${rk("bedroom").length} bedrooms`, how);
      if (/kitchen|keittiö|keittio/.test(t)) return res(rk("kitchen"), `${rk("kitchen").length} kitchens`, how);
      if (/bathroom|washroom|kylpyhuone|pesuhuone/.test(t)) return res(rk("bath"), `${rk("bath").length} bathrooms`, how);
      if (/balcon|parveke/.test(t)) return res(rk("outdoor"), `${rk("outdoor").length} balconies`, how);
    }
    if (/apartment|unit|wohnung|logement|apparta|asun/.test(t)) {
      const apts = S.units.filter((u) => u.apartment);
      return res(apts, `${apts.length} apartments`, "Counted labelled / full-size polygons on DXF layer UNIT_OUTLINE (tiny drafting fragments ignored).");
    }
    const lab = quoted(t)[0] ?? null;
    const hits = lab ? R.filter((r) => norm(r.label) === lab) : /bedroom/.test(t) ? R.filter((r) => r.kind === "bedroom")
      : /room|zimmer|räume/.test(t) ? R.filter(indoor) : null;
    if (hits) return res(hits, `${hits.length} ${lab ? `rooms labelled '${hits[0]?.label ?? lab}'` : "rooms"}`, "Counted labelled polygons on DXF layer ROOM_OUTLINE.");
  }
  if ((m = t.match(/room\s+(u?\d+\.\d+)/)) || (m = t.match(/\b(u?\d+\.\d{2})\b/))) {
    const r = R.find((r) => norm(r.number) === m[1]);
    if (r) return res([r], `${r.area_m2} m²`, `Room ${r.number} '${r.label}': shoelace area of its ROOM_OUTLINE polygon${r.holes.length ? " minus holes" : ""}${r.printed_area ? `. Printed on drawing: ${r.printed_area}` : ""}.`);
  }
  if ((m = t.match(/apartment\s+([a-z])\b/)) || (m = t.match(/(?:whg\.?|appt?\.?|wohnung|appartement)\s*([a-z])\b/))) {
    const u = S.units.find((u) => norm(u.label).endsWith(" " + m[1]) || norm(u.label).endsWith("." + m[1]));
    if (u) { const rs = R.filter((r) => r.unit === u.id && indoor(r)); return res([u, ...rs], `${sum(rs)} m²`, `Sum of ${rs.length} indoor rooms inside ${u.label} (outdoor spaces excluded).`); }
  }
  if (/area|fläche|surface/.test(t)) {
    const lab = quoted(t)[0] ?? null;
    const rs = lab ? R.filter((r) => norm(r.label) === lab) : R.filter(indoor);
    if (rs.length) return res(rs, `${sum(rs)} m²`, `Sum of ${rs.length} ${lab ? `rooms labelled '${lab}'` : "numbered indoor rooms"} (ROOM_OUTLINE polygons).`);
  }
  return none();
}

const IFC_WORDS = {
  doors: /door|tür|türen|dør|døre|deur|porte/, windows: /window|fenster|vindu|raam|ramen|fenêtre/,
  walls: /wall|wänd|wand|væg|muur|muren|mur/, stairs: /stair|treppe|trappe|trap/, columns: /column|stütze|søjle|kolom|poteau/,
  beams: /beam|träger|bjælke|balk|poutre/, slabs: /slab|decke|dæk|vloer|plancher|floor slab/, sanitary: /sanitary|sanitär|toilet|wc/,
  rooms: /room|space|raum|räume|rum|ruimte|pièce|zimmer/,
};

function answerIFC(q) {
  const t = norm(q), Q = quoted(q).map(norm);
  // storey: a quoted storey name, else any storey name mentioned (longest first)
  const storeys = [...S.storeys].sort((a, b) => (b.name ?? "").length - (a.name ?? "").length);
  const st = storeys.find((s) => Q.includes(norm(s.name))) ?? storeys.find((s) => s.name && t.includes(norm(s.name)));
  const els = [...byId.values()].filter((r) => !st || r.el.storey === st.id);
  const where = st ? `on '${st.name}'` : "in the whole model";
  const sum = (xs, k) => Math.round(xs.reduce((a, r) => a + (r.el[k] ?? 0), 0) * 1000) / 1000;
  const of = (g) => els.filter((r) => r.group === g);
  // a named room: quoted label that is not a storey name
  const roomQ = Q.find((x) => !S.storeys.some((s) => norm(s.name) === x));
  if (roomQ && /area|fläche|areal|oppervlak|surface/.test(t)) {
    const hit = [...byId.values()].filter((r) => r.group === "rooms" && (norm(r.el.label) === roomQ || norm(r.el.long_name) === roomQ || norm(r.el.name) === roomQ) && (!st || r.el.storey === st.id));
    if (hit.length === 1) {
      const el = hit[0].el;
      return el.net_floor_area_m2 != null
        ? { text: `${el.net_floor_area_m2.toFixed(2)} m²`, method: `Room ${el.label} on '${el.storey_name}': ${el.area_source === "quantity set" ? "NetFloorArea from the IFC quantity set" : "plan area of the IfcSpace body geometry"}.`, ids: [el.id] }
        : { text: "Area not available", method: "This IfcSpace has neither an area quantity nor readable geometry.", ids: [el.id] };
    }
  }
  if (/how many|count|wie viele|hvor mange|hoeveel|combien|kuinka monta/.test(t)) {
    for (const [g, re] of Object.entries(IFC_WORDS)) if (re.test(t)) {
      const xs = of(g);
      return { text: `${xs.length} ${g}`, method: `Counted ${g === "rooms" ? "IfcSpace" : xs[0]?.el.layer ?? g} entities ${where} (spatial containment in the IFC model).`, ids: xs.map((r) => r.el.id) };
    }
  }
  if (/length|länge|længde|lengte/.test(t) && IFC_WORDS.walls.test(t)) {
    const xs = of("walls"); return { text: `${sum(xs, "length_m")} m`, method: `Sum of the Length quantity of ${xs.length} walls ${where}.`, ids: xs.map((r) => r.el.id) };
  }
  if (/volume|volumen|rumfang/.test(t)) {
    const g = IFC_WORDS.walls.test(t) ? "walls" : "slabs", xs = of(g);
    return { text: `${sum(xs, "net_volume_m3")} m³`, method: `Sum of the NetVolume quantity of ${xs.length} ${g} ${where}.`, ids: xs.map((r) => r.el.id) };
  }
  if (/side area|seitenfläche|sideareal/.test(t)) {
    const xs = of("walls"); return { text: `${sum(xs, "net_side_area_m2")} m²`, method: `Sum of the NetSideArea quantity of ${xs.length} walls ${where}.`, ids: xs.map((r) => r.el.id) };
  }
  if (/area|fläche|areal|oppervlak|surface/.test(t)) {
    const xs = of("rooms"); return { text: `${sum(xs, "net_floor_area_m2").toFixed(2)} m²`, method: `Sum of the net floor area of ${xs.length} rooms ${where}.`, ids: xs.map((r) => r.el.id) };
  }
  return none();
}

function res(els, text, method) { return { text, method, ids: els.map((e) => e.id) }; }
function none() { return { text: "I can't answer that from this drawing yet.", method: "No matching tool: answering would mean guessing, so I'm not.", ids: [] }; }

// Ask the stoa_qa API (multilingual engine, all formats) when the page is served by it
// (uv run uvicorn stoa_qa.api:app); otherwise fall back to the in-browser answer() above.
async function askApi(q) {
  try {
    const r = await fetch("/ask", { method: "POST", headers: { "content-type": "application/json" },
                                    body: JSON.stringify({ path: S.source.file, question: q }) });
    if (!r.ok) return null;
    const d = await r.json();
    const conf = d.confidence >= 0.9 ? "high" : d.confidence >= 0.6 ? "medium" : "low";
    const exp = d.expected == null ? "" :
      ` · Stoa's answer: <b>${d.expected}</b> ${String(d.expected) === String(d.value) ? "✓" : "✗"}`;
    return { text: d.text, ids: d.ids.map(sceneId).filter((i) => i != null),
             method: `${d.method} · confidence ${conf} · ${d.latency_ms} ms${exp}`, engine: "stoa_qa API" };
  } catch { return null; }
}

$("#ask").onsubmit = async (e) => {
  e.preventDefault();
  const q = $("#q").value;
  const a = (await askApi(q)) ?? { ...answer(q), engine: "in-browser" };
  select(a.ids, a.ids.length ? `<b style="color:var(--text)">Evidence:</b> ${a.ids.length} element(s) highlighted in 2D and 3D.<br>${a.ids.slice(0, 40).join(", ")}${a.ids.length > 40 ? " …" : ""}` : "");
  $("#answer").innerHTML = `<div class="big">${a.text}</div><div class="method">${a.method}</div><div class="method">Source: ${S.source.file} · engine: ${a.engine}</div>`;
};

// ------------------------------------------------------------------ toolbar
$("#xray").onclick = (e) => { xray = !xray; e.target.classList.toggle("on", xray); applyXray(); };
$("#isolate").onclick = (e) => { isolate = !isolate; e.target.classList.toggle("on", isolate); applyStorey(); frame(); };
$("#topview").onclick = () => frame(true);
$("#reset").onclick = () => frame(false);
const tab = (orig) => { $("#orig").style.display = orig ? "flex" : "none"; $("#tab-orig").classList.toggle("on", orig); $("#tab-vec").classList.toggle("on", !orig); };
$("#tab-vec").onclick = () => tab(false);
$("#tab-orig").onclick = () => tab(true);
$("#glb").onclick = () => {
  new GLTFExporter().parse(root, (buf) => {
    const a = document.createElement("a");
    a.href = URL.createObjectURL(new Blob([buf], { type: "model/gltf-binary" }));
    a.download = `${$("#scene").value}.glb`; a.click();
  }, (err) => alert(err), { binary: true, onlyVisible: true });
};

boot();

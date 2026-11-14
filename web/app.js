// OQVAR web app: drawing register, the sheet with takeoff marks, the check sheet, linked 3D.
const $ = (s, r = document) => r.querySelector(s);
const SVGNS = "http://www.w3.org/2000/svg";
const h = (tag, attrs = {}, ...kids) => {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v == null || v === false) continue;
    if (k === "class") el.className = v;
    else if (k === "html") el.innerHTML = v;
    else if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
    else if (k === "style" && typeof v === "object") Object.assign(el.style, v);
    else el.setAttribute(k, v === true ? "" : v);
  }
  for (const kid of kids.flat()) if (kid != null && kid !== false) el.append(kid.nodeType ? kid : document.createTextNode(kid));
  return el;
};
const s = (tag, attrs = {}) => { const el = document.createElementNS(SVGNS, tag); for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, v); return el; };
const esc = (t) => String(t).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const fmt = (n) => Number(n).toLocaleString("en-US");
const pad2 = (n) => String(n).padStart(2, "0");
const api = {
  async get(url) { const r = await fetch(url); if (!r.ok) throw new Error(`${r.status} ${url}`); return r.json(); },
  async post(url, body) {
    const r = await fetch(url, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body) });
    if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || `${r.status} ${url}`);
    return r.json();
  },
};

const DEMOS = [
  ["CAD", "data/swiss_dwellings/site1047_bldg2154_floor9740.pdf"],
  ["BIM", "data/ifc_models/kit-fzk-haus_ac20-fzk-haus.ifc"],
  ["Raster", "data/cubicasa5k/colorful-11260.png"],
  ["Schedule", "data/opening_design/Sports_Complex/Sheets - Current Set/A-801 - Door and Finish Schedules & Wall Types.pdf"],
  ["Scan", "data/finna/hka.139877668752900_168422869617700.jpg"],
];
const FREEFORM = {
  swiss_dwellings: ["Which room is the largest?", "Which rooms open onto the corridor?", "How do I get from the kitchen to the stairwell?", "Which rooms have no window?",
    "How many entrance doors are there?", "Which rooms have a toilet?", "What is the widest door?", "What is the average bedroom size?"],
  ifc_models: ["Which room is the largest?", "How do I get from the kitchen to the exit?", "Which rooms have no window?", "How many windows are there on each storey?",
    "What is the narrowest door?", "Which software created this model?"],
  ifc_bench: ["Which room is the largest?", "Which component types make up the plumbing system?", "How well connected is the pipe network?",
    "What is the total length of pipe by storey?", "Which rooms have no window?", "How many exterior doors are there?", "How many doors are there on each storey?", "Which software created this model?"],
  cubicasa5k: ["What total floor area is printed on this plan?"],
  opening_design: ["Which other sheets does this sheet refer to?", "What is the project name in the title block?"],
  default: ["What does this drawing show?", "What scale is this drawing?"],
};
const SHORT = { swiss_dwellings: "Swiss CAD", cubicasa5k: "CubiCasa", opening_design: "Opening Dsn" };
const ENGINE = { geometry: "CAD geometry", bim: "BIM model", detection: "Detectors + vision", planner: "Planned query", graph: "Room graph", mep: "MEP model",
  document: "Sheet text", vision: "Vision model", none: "—" };

const state = {
  source: null, q: "", onlyQ: true, offset: 0, total: 0, busy: false, tab: "stoa",
  drawing: null, sheet: null, view: "sheet", cards: [], active: null, focus: 0, threads: new Map(), seq: 0,
  session: { n: 0, checked: 0, right: 0, tokens: 0, cost: 0 }, pending3d: null, plan: null, planF: 1, items: [],
  graphOn: false, graphs: new Map(), heatOn: false, split: null,
};

// ------------------------------------------------------------------ pan / zoom stage
const stage = {
  root: $("#stage"), canvas: $("#canvas"), x: 0, y: 0, k: 1, W: 1, H: 1,
  set(el, W, H) { this.W = W; this.H = H; this.canvas.replaceChildren(el); this.fit(false); },
  fitK() {  // never <= 0: a hidden or collapsed stage (0 px tall) would otherwise mirror the sheet; the ResizeObserver refits later
    const r = this.root.getBoundingClientRect();
    return Math.max(Math.min((r.width - 76) / this.W, (r.height - 76) / this.H), 0.01);
  },
  apply(anim) {
    this.canvas.classList.toggle("anim", !!anim);
    this.canvas.style.transform = `translate(${this.x}px, ${this.y}px) scale(${this.k})`;
    $("#z-val").textContent = `${Math.round((this.k / this.fitK()) * 100)}%`;
    scaleOverlay();
  },
  fit(anim = true) {
    const r = this.root.getBoundingClientRect();
    this.k = this.fitK();
    this.x = (r.width - this.W * this.k) / 2; this.y = (r.height - this.H * this.k) / 2;
    this.apply(anim);
  },
  zoomAt(cx, cy, f, anim = false) {
    const k2 = Math.min(Math.max(this.k * f, this.fitK() * 0.4), this.fitK() * 40);
    this.x = cx - (cx - this.x) * (k2 / this.k); this.y = cy - (cy - this.y) * (k2 / this.k); this.k = k2;
    this.apply(anim);
  },
  zoomTo([x0, y0, x1, y1]) {
    const r = this.root.getBoundingClientRect(), fitK = this.fitK();
    const bw = Math.max(x1 - x0, 1), bh = Math.max(y1 - y0, 1);
    const cap = state.sheet?.kind === "svg" ? fitK * 16 : Math.max(1.6, fitK * 2.6);  // raster: stay near native pixels
    this.k = Math.max(fitK, Math.min(r.width / (bw * (bw > 5 * bh ? 1.25 : 3.2)), r.height / (bh * 3.2), cap));
    this.x = r.width / 2 - ((x0 + x1) / 2) * this.k; this.y = r.height / 2 - ((y0 + y1) / 2) * this.k;
    this.apply(true);
  },
  init() {
    let drag = null;
    this.root.addEventListener("wheel", (e) => {
      if (state.view !== "sheet") return;
      e.preventDefault();
      const r = this.root.getBoundingClientRect();
      this.zoomAt(e.clientX - r.left, e.clientY - r.top, Math.exp(-e.deltaY * 0.0016));
    }, { passive: false });
    this.root.addEventListener("pointerdown", (e) => {
      if (state.view !== "sheet" || e.button !== 0 || e.target.closest(".stepper")) return;
      drag = { x: e.clientX, y: e.clientY, ox: this.x, oy: this.y, moved: false };
    });
    window.addEventListener("pointermove", (e) => {
      if (!drag) return;
      const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
      if (!drag.moved && Math.hypot(dx, dy) < 4) return;
      drag.moved = true; this.root.classList.add("drag");
      this.x = drag.ox + dx; this.y = drag.oy + dy; this.apply(false);
    });
    window.addEventListener("pointerup", () => { if (drag?.moved) this.dragEnd = Date.now(); drag = null; this.root.classList.remove("drag"); });
    this.root.addEventListener("dblclick", (e) => {
      if (state.view !== "sheet") return;
      const r = this.root.getBoundingClientRect();
      this.zoomAt(e.clientX - r.left, e.clientY - r.top, 2, true);
    });
    const centre = (f) => { const r = this.root.getBoundingClientRect(); this.zoomAt(r.width / 2, r.height / 2, f, true); };
    $("#z-in").onclick = () => centre(1.5); $("#z-out").onclick = () => centre(1 / 1.5); $("#z-fit").onclick = () => this.fit();
    new ResizeObserver(() => { if (state.sheet && !this.canvas.classList.contains("anim")) this.fit(false); }).observe(this.root);
  },
};
const wasDrag = () => Date.now() - (stage.dragEnd || 0) < 250;

// ------------------------------------------------------------------ register
async function loadConfig() {
  const cfg = await api.get("/api/config");
  $("#tb-drawings").textContent = `${fmt(cfg.drawings)} sheets`;
  $("#tb-questions").textContent = fmt(cfg.questions);
  $("#tb-model").textContent = cfg.llm ? cfg.llm_model : "none · geometry only";
  $("#demo").replaceChildren(...DEMOS.map(([label, id]) => h("button", { onclick: () => openDrawing(id), title: id }, label)));
}

async function loadList(reset = true) {
  if (state.busy) return;
  state.busy = true;
  if (reset) { state.offset = 0; $("#list").replaceChildren(); $("#list").scrollTop = 0; }
  const p = new URLSearchParams({ offset: state.offset, limit: 40, only_questions: state.onlyQ });
  if (state.source) p.set("source", state.source);
  if (state.q) p.set("q", state.q);
  let r;
  try { r = await api.get(`/api/drawings?${p}`); } catch (e) { state.busy = false; $("#list").replaceChildren(h("div", { class: "more" }, `Register unavailable: ${e.message}`)); return; }
  state.total = r.total;
  $("#reg-count").textContent = `${fmt(r.total)} sheet${r.total === 1 ? "" : "s"}`;
  if (reset) {
    $("#sources").replaceChildren(
      h("button", { class: "src" + (state.source ? "" : " on"), "aria-pressed": !state.source, onclick: () => { state.source = null; loadList(); } }, "All sources"),
      ...r.facets.map((f) => h("button", { class: "src" + (state.source === f.source ? " on" : ""), "aria-pressed": state.source === f.source, title: `${f.label} — ${f.kind}`,
        onclick: () => { state.source = f.source; loadList(); } }, SHORT[f.source] || f.label, h("small", {}, fmt(f.count)))));
  }
  $("#list .more")?.remove();
  for (const d of r.items) $("#list").append(rowEl(d));
  state.offset += r.items.length;
  if (state.offset < r.total) {
    const more = h("div", { class: "more" }, `${fmt(state.offset)} of ${fmt(r.total)} — scroll for more`);
    $("#list").append(more);
    new IntersectionObserver((es, ob) => { if (es[0].isIntersecting) { ob.disconnect(); loadList(false); } }, { root: $("#list") }).observe(more);
  } else if (!r.total) $("#list").append(h("div", { class: "more" }, "No sheets match."));
  state.busy = false;
}

function rowEl(d) {
  return h("button", { class: "row" + (state.drawing?.id === d.id ? " on" : ""), "data-id": d.id, onclick: () => openDrawing(d.id) },
    h("img", { loading: "lazy", src: `/api/thumb?id=${encodeURIComponent(d.id)}&w=160`, alt: "" }),
    h("div", {}, h("b", { title: d.title }, d.title),
      h("span", {}, d.formats.slice(0, 3).join(" · "), d.has_3d && h("u", {}, "3D"), d.n_questions ? h("i", {}, `${d.n_questions} Q`) : null)));
}

// ------------------------------------------------------------------ drawing + sheets
async function openDrawing(id) {
  if (state.drawing) state.threads.set(state.drawing.id, state.cards);
  setMsg("Opening the sheet", true);
  let d;
  try { d = await api.get(`/api/drawing?id=${encodeURIComponent(id)}`); } catch (e) { setMsg(`<b>Could not open this sheet</b>${esc(e.message)}`); return; }
  state.drawing = d; state.cards = state.threads.get(d.id) || []; state.active = state.cards[0] || null; state.focus = 0; state.items = [];
  state.tab = d.questions.length ? "stoa" : "free";
  history.replaceState(null, "", `#${encodeURIComponent(d.id)}`);
  document.querySelectorAll(".row").forEach((el) => el.classList.toggle("on", el.dataset.id === d.id));
  $("#d-title").textContent = d.title; $("#d-title").title = d.id;
  const tag3d = !d.has_3d ? "" : d.model_kind === "raster" ? " · 3D (approximate)" : d.model_note ? " · 3D (project model)" : " · 3D";
  $("#d-meta").replaceChildren(`${d.source_info.label || d.source} · ${d.formats.join(" · ")}`,
    d.model_note ? h("span", { title: d.model_note }, tag3d) : tag3d,
    d.question_count ? h("i", {}, ` · ${d.question_count} questions`) : "");
  const c = d.credit || {};
  $("#credit").innerHTML = [c.title && `<b>${esc(c.title)}</b>`, c.credit && esc(c.credit), c.licence && esc(c.licence),
    c.url && `<a href="${esc(c.url)}" target="_blank" rel="noopener">source</a>`].filter(Boolean).join(" · ") || "—";
  const sel = $("#sheet-select");
  sel.replaceChildren(...d.sheets.map((sh, i) => h("option", { value: i }, sh.label)));
  sel.classList.toggle("hidden", d.sheets.length < 2);
  $("#btn-3d").disabled = !d.has_3d;
  $("#btn-graph").disabled = d.source !== "swiss_dwellings";  // IFC graphs answer questions but have no sheet coordinates to draw in
  $("#btn-graph").title = d.source === "swiss_dwellings" ? "Show how the rooms connect: doors, open connections and shared walls"
    : d.source.startsWith("ifc") ? "Room graph questions work on this model; the graph is drawn on Swiss CAD sheets only" : "No room graph for raster or scanned sheets";
  $("#three").src = "about:blank"; state.sceneUrl = null;
  renderSuggest(); renderThread();
  if (d.sheets.length) { setView("sheet"); await showSheet(0); }
  else if (d.has_3d) { stage.canvas.replaceChildren(); state.sheet = null; setView("3d"); }
  else { stage.canvas.replaceChildren(); state.sheet = null; setMsg(`<b>No preview for this file type</b>${esc(d.formats.join(", ").toUpperCase())} files are listed in the register but cannot be drawn yet.`); }
}

async function showSheet(i) {
  const sh = state.drawing.sheets[i];
  state.sheet = sh; state.sheetIndex = i; state.plan = null; state.items = []; $("#sheet-select").value = i;
  setMsg("Opening the sheet", true);
  try {
    if (sh.kind === "svg") {
      const text = await (await fetch(sh.src)).text();
      const plan = document.importNode(new DOMParser().parseFromString(text, "image/svg+xml").documentElement, true);
      plan.classList.add("plan"); plan.removeAttribute("width"); plan.removeAttribute("height");
      const vb = (plan.getAttribute("viewBox") || `0 0 ${sh.width} ${sh.height}`).split(/[\s,]+/).map(Number);
      state.planF = 6; state.planVB = vb; state.plan = plan;
      plan.append(s("g", { class: "tg" }));
      plan.addEventListener("click", (e) => {
        if (wasDrag()) return;
        const g = e.target.closest("[data-guid]");
        if (g) tip(`${g.getAttribute("class").replace("cut ", "").replace("pl-ev", "").replace("focus", "").trim()} · ${g.dataset.name || ""}`, e);
      });
      stage.set(h("div", { class: "sheet", style: { width: `${vb[2] * state.planF}px`, height: `${vb[3] * state.planF}px` } }, plan), vb[2] * state.planF, vb[3] * state.planF);
    } else {
      const img = h("img", { src: sh.src, alt: state.drawing.title, draggable: "false" });
      const vb = `0 0 ${sh.width} ${sh.height}`;
      stage.set(h("div", { class: "sheet", style: { width: `${sh.width}px`, height: `${sh.height}px` } },
        img, s("svg", { class: "ov gr", viewBox: vb }), s("svg", { class: "ov hl", viewBox: vb }), s("svg", { class: "ov tg", viewBox: vb })), sh.width, sh.height);
      await new Promise((res, rej) => { if (img.complete) res(); else { img.onload = res; img.onerror = () => rej(new Error("the image did not load")); } });
    }
  } catch (e) { setMsg(`<b>Could not open this sheet</b>${esc(e.message)}`); return; }
  setMsg("");
  drawOverlay(); drawGraph();
}

function setMsg(html, busy = false) {
  $("#stage-msg").innerHTML = !html ? "" : busy ? `<div><span class="mark"></span><b>${html}</b></div>` : `<div>${html}</div>`;
}

function setView(v) {
  state.view = v;
  document.querySelectorAll("#view-seg button").forEach((b) => { b.classList.toggle("on", b.dataset.view === v); b.setAttribute("aria-pressed", b.dataset.view === v); });
  $("#three").classList.toggle("hidden", v !== "3d"); $("#canvas").classList.toggle("hidden", v === "3d");
  document.querySelector(".zoom").classList.toggle("hidden", v === "3d");
  if (v === "3d") open3D(); else { setMsg(""); updateStepper(); }
}

async function open3D() {
  $("#stepper").classList.add("hidden");
  if (!state.sceneUrl) {
    const from = { ifc: "the IFC model", dxf: "the CAD geometry", mesh: "the 3D mesh", raster: "the scan (approximate)" };
    setMsg(state.drawing.upload ? "Tracing walls, rooms, doors and windows from your plan (up to half a minute the first time)"
      : `Building the 3D model from ${from[state.drawing.model_kind] || "the model"}`, true);
    try {
      const r = await api.post("/api/scene", { id: state.drawing.id, question: "" });
      state.sceneUrl = r.url; state.pending3d = state.active?.res?.ids || [];
      $("#three").src = r.url;
    } catch (e) { setMsg(`<b>3D view unavailable</b>${esc(e.message)}`); return; }
  } else highlight3D(state.active?.res?.ids || []);
  setMsg("");
}

function highlight3D(ids) {
  state.pending3d = ids;
  $("#three").contentWindow?.postMessage({ type: "oqvar:highlight", ids }, location.origin);
}
window.addEventListener("message", (e) => {
  if (e.origin === location.origin && e.data?.type === "oqvar:ready" && state.pending3d) highlight3D(state.pending3d);
});

// ------------------------------------------------------------------ takeoff marks on the sheet
// raster sheets: one multiply-blended layer for highlighter shapes, one normal layer for balloons;
// IFC plans: the plan's own elements are recoloured, balloons go in a group inside the plan
const layers = () => (state.sheet?.kind === "svg"
  ? { shapes: null, tags: state.plan?.querySelector("g.tg") }
  : { shapes: $("#canvas svg.ov.hl"), tags: $("#canvas svg.ov.tg") });

function scaleOverlay() {
  const { shapes, tags } = layers(); if (!tags) return;
  const px = 1 / (stage.k * (state.sheet.kind === "svg" ? state.planF : 1));  // one screen pixel in overlay units
  shapes?.style.setProperty("--sw", `${2 * px}px`);
  $("#canvas svg.ov.gr")?.style.setProperty("--sw", `${px}px`);
  for (const c of document.querySelectorAll("#canvas svg.ov.gr .n")) c.setAttribute("r", 4.5 * px);
  for (const c of document.querySelectorAll("#canvas svg.ov.gr .dn")) c.setAttribute("r", 2.6 * px);
  for (const c of document.querySelectorAll("#canvas .route-end")) c.setAttribute("r", 6 * px);
  for (const it of state.items) it.balloon?.setAttribute("transform", `translate(${it.bx} ${it.by}) scale(${px})`);
}

function drawOverlay() {
  const { shapes, tags } = layers(); if (!tags) return;
  shapes?.replaceChildren(); tags.replaceChildren();
  state.plan?.querySelectorAll(".pl-ev").forEach((e) => e.classList.remove("pl-ev", "focus", "b-direct", "b-inferred", "b-review"));
  const card = state.active; state.items = [];
  if (!card?.res || !state.sheet) { updateStepper(); return; }
  const ov = card.res.overlay;
  if (ov.space === "guid") {
    if (state.sheet.kind !== "svg") { updateStepper(); return; }
    for (const it of ov.items) {
      const el = state.plan.querySelector(`[data-guid="${CSS.escape(it.guid)}"]`);
      if (!el) continue;
      el.classList.add("pl-ev", `b-${basisOf(it)}`);
      let b; try { b = el.getBBox(); } catch { continue; }
      const vb = state.planVB, F = state.planF;
      state.items.push({ ...it, el, bx: b.x + b.width / 2, by: b.y + b.height / 2, pb: [b.x, b.y, b.x + b.width, b.y + b.height],
        bbox: [(b.x - vb[0]) * F, (b.y - vb[1]) * F, (b.x + b.width - vb[0]) * F, (b.y + b.height - vb[1]) * F] });
    }
  } else if (state.sheet.kind === "image" && state.sheetIndex === 0) {
    if (ov.path?.length > 1) {  // graph route: the walked line, start to finish
      shapes.append(s("polyline", { class: "route", points: ov.path.map((p) => p.join(",")).join(" ") }));
      const [ex, ey] = ov.path[ov.path.length - 1];
      tags.append(s("circle", { class: "route-end", cx: ex, cy: ey, r: 6 }));
    }
    ov.items.forEach((it, i) => {
      const cls = `shape b-${basisOf(it)}`;
      const shape = it.kind === "point" ? s("circle", { class: cls, cx: it.pts[0][0], cy: it.pts[0][1], r: 26 })
        : s("polygon", { class: cls, points: it.pts.map((p) => p.join(",")).join(" ") });
      shape.style.animationDelay = `${Math.min(i * 14, 420)}ms`;
      shapes.append(shape);
      const [x0, y0, x1, y1] = it.bbox, row = x1 - x0 > 5 * (y1 - y0);
      state.items.push({ ...it, el: shape, row, bx: row ? x0 : (x0 + x1) / 2, by: (y0 + y1) / 2 });  // table rows: balloon at the row start
    });
  }
  state.items.forEach((it, i) => {
    it.i = i + 1;
    // a callout: leader line from the item to a numbered balloon beside it (left of a table row)
    const balloon = s("g", { class: `balloon b-${basisOf(it)}` }), [cx, cy] = it.row ? [-13, 0] : [17, -17];
    if (!it.row) balloon.append(s("line", { x1: 0, y1: 0, x2: 10, y2: -10 }), s("circle", { class: "dot", r: 2 }));
    balloon.append(s("circle", { cx, cy, r: 9.5 }), Object.assign(s("text", { x: cx, y: cy }), { textContent: it.n }));
    tags.append(balloon); it.balloon = balloon;
    const label = `${it.n} · ${it.label || "item"}${it.value != null ? ` · ${it.value}` : ""}`;
    for (const el of [it.el, balloon]) {
      el.addEventListener("click", (e) => { if (wasDrag()) return; e.stopPropagation(); focusItem(it.i); });
      el.addEventListener("mousemove", (e) => tip(label, e)); el.addEventListener("mouseleave", () => tip(null));
    }
  });
  scaleOverlay(); applyFocus(false); updateStepper(); drawHeatKey();
}

// ------------------------------------------------------------------ room graph layer (stoa_qa.graph via /api/graph)
async function toggleGraph() {
  state.graphOn = !state.graphOn;
  $("#btn-graph").setAttribute("aria-pressed", state.graphOn);
  await drawGraph();
}

async function drawGraph() {
  const layer = $("#canvas svg.ov.gr"); if (!layer) return;
  layer.replaceChildren();
  const d = state.drawing;
  if (!state.graphOn || !d || state.sheet?.kind !== "image" || state.sheetIndex !== 0) return;
  let g = state.graphs.get(d.id);
  if (!g) {
    try { g = await api.get(`/api/graph?id=${encodeURIComponent(d.id)}`); } catch (e) { g = { error: e.message }; }
    state.graphs.set(d.id, g);
  }
  if (g.error || g.space !== "px") { setMsg(`<b>No room graph on this sheet</b>${esc(g.error || "it can be queried, but not drawn on this kind of sheet")}`); setTimeout(() => setMsg(""), 2600); return; }
  if (state.drawing !== d) return;
  const at = new Map(g.nodes.map((n) => [n.id, n]));
  for (const e of g.edges) {
    const a = at.get(e.a), b = at.get(e.b);
    const pts = [a?.px, e.px, b?.px].filter(Boolean);
    if (pts.length < 2) continue;
    const line = s("polyline", { class: `e ${e.kind}`, points: pts.map((p) => p.join(",")).join(" ") });
    layer.append(line);
    if (e.kind === "door" && e.px) layer.append(s("circle", { class: "dn", cx: e.px[0], cy: e.px[1], r: 2.6 }));
  }
  for (const n of g.nodes) {
    if (!n.px) continue;
    const links = g.edges.filter((e) => e.a === n.id || e.b === n.id);
    const doors = links.filter((e) => e.kind === "door").length, open = links.filter((e) => e.kind === "open").length;
    const text = `${n.name}${n.area_m2 ? ` · ${n.area_m2} m²` : ""} · ${doors} door${doors === 1 ? "" : "s"}${open ? ` · ${open} open` : ""} · ${n.windows} window${n.windows === 1 ? "" : "s"}`;
    const c = s("circle", { class: `n${n.kind === "outside" ? " out" : ""}`, cx: n.px[0], cy: n.px[1], r: 4.5 });
    c.addEventListener("mousemove", (ev) => tip(text, ev)); c.addEventListener("mouseleave", () => tip(null));
    layer.append(c);
  }
  scaleOverlay();
}

function applyFocus(zoom = true) {
  const { shapes, tags } = layers(); if (!tags) return;
  const f = state.focus;
  shapes?.classList.toggle("dim", f > 0); tags.classList.toggle("dim", f > 0);
  for (const it of state.items) {
    const on = it.i === f;
    it.el.classList.toggle("focus", on); it.balloon.classList.toggle("focus", on);
    if (on) { if (it.el.parentNode === shapes) shapes.append(it.el); tags.append(it.balloon); }
  }
  const n = state.items[f - 1]?.n ?? -1;
  document.querySelectorAll(".rec.on [data-n]").forEach((el) => {
    const on = +el.dataset.n === n;
    el.classList.toggle("focus", on);
    if (on) el.scrollIntoView({ block: "nearest", inline: "nearest" });
  });
  tags.querySelectorAll(".pulse").forEach((e) => e.remove());
  const fit = state.items[f - 1];
  if (fit) {  // a ring that pulses around the focused item, so the eye finds it at any zoom
    const [x0, y0, x1, y1] = fit.pb || fit.bbox, m = Math.max(x1 - x0, y1 - y0) * 0.12 + (fit.pb ? 0.3 : 4);
    for (const cls of ["pulse ring", "pulse wave"])
      tags.prepend(s("rect", { class: cls, x: x0 - m, y: y0 - m, width: x1 - x0 + 2 * m, height: y1 - y0 + 2 * m, rx: Math.min(m, fit.pb ? 0.5 : 8), "vector-effect": "non-scaling-stroke" }));
  }
  updateLegend();
  if (zoom) { const it = state.items[f - 1]; if (it) stage.zoomTo(it.bbox); else stage.fit(); }
}

function focusItem(i) {
  if (state.view !== "sheet") setView("sheet");
  const n = state.items.length; if (!n) return;
  state.focus = i < 1 ? n : i > n ? 1 : i;
  applyFocus(); updateStepper();
}

function updateStepper() {
  const n = state.items.length, st = $("#stepper");
  st.classList.toggle("hidden", !n || state.view !== "sheet");
  if (!n) return;
  const it = state.items[state.focus - 1];
  $("#st-label").innerHTML = it ? `<b>${pad2(state.focus)}</b> / ${pad2(n)} — ${esc(it.label || "item")}${it.value != null ? ` · ${esc(it.value)}` : ""}`
    : `<b>${n}</b> item${n > 1 ? "s" : ""} marked — step through to check each`;
}

function tip(text, e) {
  const t = $("#tip");
  if (!text) { t.classList.add("hidden"); return; }
  t.textContent = text; t.classList.remove("hidden");
  t.style.left = `${Math.min(e.clientX + 14, innerWidth - 350)}px`; t.style.top = `${e.clientY + 16}px`;
  clearTimeout(tip.timer); tip.timer = setTimeout(() => t.classList.add("hidden"), 2600);
}

// ------------------------------------------------------------------ check sheet: questions
function renderSuggest() {
  const d = state.drawing; if (!d) return;
  if (state.tab === "stoa" && !d.questions.length) state.tab = "free";
  $("#tab-stoa").classList.toggle("on", state.tab === "stoa"); $("#tab-free").classList.toggle("on", state.tab === "free");
  $("#tab-stoa").disabled = !d.questions.length;
  $("#tab-stoa").firstChild.textContent = d.upload ? "Counts " : "Stoa's questions ";  // an upload has no dataset questions
  $("#q-count").textContent = d.upload ? "" : d.question_count ? `(${d.question_count})` : "(none)";
  const qs = state.tab === "stoa" ? d.questions : (FREEFORM[d.source] || FREEFORM.default).map((q) => ({ question: q, lang: "any" }));
  $("#suggest").replaceChildren(...qs.map((q) => h("li", {},
    h("button", { title: q.question, onclick: () => ask(q.question) }, h("code", {}, q.lang), h("span", {}, q.question)))));
}

async function ask(question) {
  question = (question || "").trim();
  if (!question || !state.drawing || state.asking) return;
  state.asking = true; $("#send").disabled = true; $("#q").value = "";
  const now = new Date();
  const card = { question, res: null, drawing: state.drawing.id, no: ++state.seq, at: `${pad2(now.getHours())}:${pad2(now.getMinutes())}:${pad2(now.getSeconds())}` };
  state.cards.unshift(card); state.active = card; state.focus = 0; state.items = [];
  renderThread(); drawOverlay(); $("#thread").scrollTop = 0;
  try { card.res = await api.post("/api/ask", { id: state.drawing.id, question, sheet_path: state.sheet?.path || null, sheet_page: state.sheet?.page || 0 }); }
  catch (e) { card.error = e.message; }
  state.asking = false; $("#send").disabled = false;
  if (card.drawing !== state.drawing.id) return;
  if (card.res) {
    const r = card.res, ss = state.session;
    ss.n++; ss.tokens += r.engine.llm_tokens; ss.cost += r.engine.cost_usd;
    if (r.correct != null) { ss.checked++; ss.right += r.correct ? 1 : 0; }
    renderMeter();
  }
  card.fresh = true;
  await activate(card);
  card.fresh = false;
}

async function activate(card) {
  state.active = card; state.focus = 0;
  const ov = card.res?.overlay;
  if (ov?.space === "guid" && ov.sheets?.length && state.drawing.sheets.length) {
    const cur = ov.sheets.find((x) => x.path === state.sheet?.path), best = ov.sheets[0];
    if (!cur || cur.hits < best.hits) {
      const i = state.drawing.sheets.findIndex((x) => x.path === best.path);
      if (i >= 0 && i !== state.sheetIndex) { await showSheet(i); renderThread(); if (state.view === "3d") highlight3D(card.res.ids); return; }
    }
  }
  if (state.view === "sheet") { drawOverlay(); stage.fit(); } else highlight3D(card.res?.ids || []);
  renderThread();
}

function renderMeter() {
  const ss = state.session;
  $("#meter").innerHTML = `<span><b>${ss.n}</b> answered</span>` +
    (ss.checked ? `<span class="${ss.right === ss.checked ? "good" : ""}"><b>${ss.right}/${ss.checked}</b> = Stoa</span>` : "") +
    `<span><b>${fmt(ss.tokens)}</b> tokens</span><span><b>$${ss.cost.toFixed(4)}</b></span>`;
}

// ------------------------------------------------------------------ check sheet: records
function stampOf(r) {
  if (r.expected != null && r.correct === true) return ["ok", "Checked", `= Stoa ${r.expected}`];
  if (r.expected != null && r.correct === false) return ["bad", "Mismatch", `Stoa says ${r.expected}`];
  if (r.value == null) return ["none", "No answer", "not guessed"];
  const k = r.engine.kind;
  if (k === "vision") return ["est", "Estimate", "check it yourself"];
  if (k === "detection") return r.confidence >= 0.85 ? ["calc", "Cross-checked", "sources agree"] : ["est", "Unconfirmed", "sources differ"];
  if (k === "document") return ["calc", "Counted", "rows on the sheet"];
  if (k === "planner") return ["calc", "Computed", "AI-planned query"];
  if (k === "graph") return ["calc", "Traced", "through the room graph"];
  if (k === "mep") return ["calc", "Computed", "from the MEP model"];
  return ["calc", "Computed", k === "bim" ? "from the model" : "from geometry"];
}

function valueEl(r) {
  if (r.value == null) return h("div", { class: "val no" }, r.text);
  if (typeof r.value === "number") {
    const m = r.text.match(/^(-?[\d.,]+)\s*(.*)$/);
    return h("div", { class: "val" }, m ? m[1] : r.text, m && m[2] ? h("small", {}, m[2]) : null);
  }
  const lines = String(r.value).split("\n");
  if (lines.length === 1 && lines[0].length <= 22) return h("div", { class: "val word" }, lines[0]);
  return h("ol", { class: "lines" }, ...lines.map((ln) => {
    const sub = ln.startsWith("  "), t = ln.trim();
    const cut = t.includes(" — ") ? t.indexOf(" — ") : t.includes(": ") ? t.indexOf(": ") : -1;
    return h("li", { class: sub ? "sub" : null }, h("span", {}, cut < 0 ? t : t.slice(0, cut).replace(/:$/, "")),
      cut < 0 ? null : h("em", {}, t.slice(cut).replace(/^( — |: )/, "")));
  }));
}

function specEl(r) {
  const e = r.engine, c = r.confidence >= 0.85 ? [3, "High"] : r.confidence >= 0.6 ? [2, "Medium"] : [1, "Low"];
  const time = r.latency_ms < 1000 ? `${Math.max(1, Math.round(r.latency_ms))} ms` : `${(r.latency_ms / 1000).toFixed(1)} s`;
  const method = e.kind === "planner" ? h("dd", { class: "wide" }, h("code", {}, r.method)) : h("dd", { class: "wide" }, r.method);
  return h("dl", { class: "spec" },
    h("dt", {}, e.kind === "planner" ? "Query" : "Method"), method,
    h("dt", {}, "Engine"), h("dd", { title: e.note }, ENGINE[e.kind] || e.kind), h("dt", {}, "Time"), h("dd", {}, time),
    h("dt", {}, "Cost"), h("dd", { title: e.models.join(", ") }, e.llm_tokens ? `${fmt(e.llm_tokens)} tokens · $${e.cost_usd.toFixed(5)}` : h("span", { class: "zero" }, "0 tokens · $0")),
    h("dt", {}, "Trust"), r.value == null ? h("dd", {}, "—")
      : h("dd", {}, h("span", { class: "conf", "aria-hidden": "true" }, ...[1, 2, 3].map((i) => h("i", { class: i <= c[0] ? "f" : null }))), c[1]));
}

// the crew: who the general contractor asked, in order; then the auditor's independent checks
// ------------------------------------------------------------------ cross-sheet references (stoa_qa.xref)
// each pair: the callout on this sheet beside the detail it points at on the other sheet; "Open" jumps there
function refsEl(r) {
  const refs = r.references || [];
  if (!refs.length) return null;
  const crop = (p, size) => `/api/crop?path=${encodeURIComponent(p.path)}&page=${p.page}&kind=pdf&size=${size}` +
    `&x0=${p.box[0]}&y0=${p.box[1]}&x1=${p.box[2]}&y1=${p.box[3]}`;
  return h("div", { class: "xrefs" },
    h("span", { class: "lbl" }, "References · ", h("b", {}, `${refs.length} followed`)),
    ...refs.slice(0, 12).map((x) => h("div", { class: "xref" },
      x.from.box ? h("figure", {}, h("img", { src: crop(x.from, 180), alt: `callout ${x.label}` }), h("figcaption", {}, `callout ${x.from.text || x.label}`))
        : h("figure", { class: "none" }, h("figcaption", {}, "asked directly")),
      h("span", { class: "arrow", "aria-hidden": "true" }, "→"),
      h("figure", {}, h("img", { src: crop(x.to, 420), alt: `${x.label}: ${x.to.title || x.to.sheet}` }),
        h("figcaption", {}, h("b", {}, x.label), x.to.title ? ` · ${x.to.title}` : "", h("br"), h("i", {}, `found by ${x.to.how}`))),
      h("button", { class: "btn mini", title: `Open sheet ${x.to.sheet} at this detail`, onclick: (ev) => { ev.stopPropagation(); openReference(x.to); } }, `Open ${x.to.sheet}`))));
}

// ------------------------------------------------------------------ when nothing could answer (stoa_qa.explain)
// the reason in plain words, what the drawing does hold, who was asked, and questions that will work here
function whyEl(r) {
  const w = r.unanswered;
  if (!w) return null;
  const box = h("div", { class: `why ${w.category}` },
    w.detail ? h("p", { class: "why-detail" }, w.detail) : null,
    h("p", { class: "why-next" }, w.next));
  if (w.alternatives?.length) box.append(h("div", { class: "why-alts" }, h("span", { class: "lbl" }, "Try instead"),
    ...w.alternatives.map((q) => h("button", { class: "alt", title: "Ask this", onclick: (ev) => { ev.stopPropagation(); ask(q); } }, q))));
  if (w.checked?.length) box.append(h("details", { class: "why-checked", onclick: (ev) => ev.stopPropagation() },
    h("summary", {}, `What I checked (${w.checked.length})`), h("ul", {}, ...w.checked.map((c) => h("li", {}, c)))));
  return box;
}

async function openReference(to) {
  if (state.drawing?.id !== to.path && !state.drawing?.files?.includes(to.path)) await openDrawing(to.path);
  const i = state.drawing.sheets.findIndex((s) => s.path === to.path && (s.page || 0) === (to.page || 0));
  if (i >= 0 && i !== state.sheetIndex) await showSheet(i);
  if (state.view !== "sheet") setView("sheet");
  stage.zoomTo(to.box);
}

function crewEl(r) {
  const crew = r.crew || [], checks = r.audit || [];
  if (!crew.length && !checks.length) return null;
  const MARK = { pass: "✓", warn: "!", fail: "✗", info: "i" };
  const box = h("div", { class: "crew" });
  if (crew.length) box.append(h("ol", { class: "crew-run", "aria-label": "Crew" }, ...crew.map((c) =>
    h("li", { class: c.outcome, title: `${c.role}${c.note ? ` — ${c.note}` : ""} · ${c.ms} ms${c.tokens ? ` · ${c.tokens} tokens` : ""}` },
      h("b", {}, c.agent), h("i", {}, c.outcome)))));
  if (checks.length) box.append(h("ul", { class: "audit", "aria-label": "Auditor checks" }, ...checks.map((c) =>
    h("li", { class: c.status }, h("span", { class: "mk", "aria-hidden": "true" }, MARK[c.status] || "·"),
      h("span", {}, h("b", {}, c.name), " ", c.detail)))));
  return box;
}

function takeoffEl(card, on) {
  const r = card.res, items = r.overlay.items, sh = state.drawing.sheets[0];
  const go = (n) => (ev) => {
    ev.stopPropagation();
    const jump = async () => {
      let i = state.items.findIndex((x) => x.n === n);
      if (i < 0) {  // not on the sheet in view: open the sheet that holds it, then focus it there
        const j = sheetOf(r, n);
        if (j >= 0 && j !== state.sheetIndex) { await showSheet(j); renderThread(); i = state.items.findIndex((x) => x.n === n); }
      }
      if (i >= 0) focusItem(i + 1);
    };
    if (!on) activate(card).then(jump); else jump();
  };
  const box = h("div", { class: "take" }, h("div", { class: "take-head" },
    h("span", { class: "lbl" }, "Takeoff · ", h("b", {}, `${items.length} item${items.length > 1 ? "s" : ""} marked`)),
    h("div", {},
      h("button", { class: "btn mini", onclick: (ev) => { ev.stopPropagation(); if (!on) activate(card).then(() => focusItem(1)); else focusItem(state.focus + 1); } }, "Step →"),
      state.drawing.has_3d && r.ids.length ? h("button", { class: "btn mini", onclick: (ev) => { ev.stopPropagation(); state.active = card; renderThread(); setView("3d"); } }, "3D") : null)));
  const px = r.overlay.space === "px" && sh;
  const wide = px && items.filter((it) => it.bbox[2] - it.bbox[0] > 5 * (it.bbox[3] - it.bbox[1])).length > items.length / 2;
  if (wide) {  // table rows: each one as a readable strip of the sheet
    box.append(h("div", { class: "strips" }, ...items.slice(0, 80).map((it) => {
      const [x0, y0, x1, y1] = it.bbox, k = 22 / Math.max(y1 - y0, 4);
      return h("button", { class: "st", "data-n": it.n, title: it.label || "", onclick: go(it.n),
        style: { width: `${Math.max((x1 - x0) * k + 24, 120)}px`, backgroundImage: `url("${sh.src}")`, backgroundSize: `${sh.width * k}px ${sh.height * k}px`,
          backgroundPosition: `${24 - x0 * k}px ${-y0 * k}px` } }, h("span", { class: "tag" }, it.n));
    })));
  } else if (px) {  // a crop of the sheet around every counted item, with the item highlighted
    box.append(h("div", { class: "grid" }, ...items.slice(0, 96).map((it) => {
      const [x0, y0, x1, y1] = it.bbox, side = Math.min(Math.max(x1 - x0, y1 - y0) * 1.9 + 30, sh.width, sh.height);
      const sx = Math.min(Math.max((x0 + x1) / 2 - side / 2, 0), sh.width - side), sy = Math.min(Math.max((y0 + y1) / 2 - side / 2, 0), sh.height - side);
      const pc = (v) => `${(v / side) * 100}%`;
      return h("button", { class: "th", "data-n": it.n, title: `${it.label || ""}${it.value != null ? ` · ${it.value}` : ""}`, onclick: go(it.n),
        style: { backgroundImage: `url("${sh.src}")`, backgroundSize: `${(sh.width / side) * 100}% ${(sh.height / side) * 100}%`,
          backgroundPosition: `${sh.width > side ? (sx / (sh.width - side)) * 100 : 0}% ${sh.height > side ? (sy / (sh.height - side)) * 100 : 0}%` } },
        h("u", { style: { left: pc(x0 - sx), top: pc(y0 - sy), width: pc(x1 - x0), height: pc(y1 - y0) } }), h("span", { class: "tag" }, it.n));
    })));
  } else {
    box.append(h("div", { class: "elist" }, ...items.slice(0, 150).map((it) =>
      h("button", { class: "el", "data-n": it.n, onclick: go(it.n) }, h("span", { class: "tag" }, it.n), it.label || "element",
        it.value != null ? h("em", {}, typeof it.value === "number" ? it.value.toFixed(2) : it.value) : null))));
  }
  if (on && r.overlay.space === "guid" && state.sheet?.kind === "svg" && state.items.length < items.length)
    box.append(h("p", { class: "note" }, `${state.items.length} of ${items.length} are on this sheet. Pick another storey above the drawing, or open the 3D model.`));
  if (on && r.overlay.space === "guid" && !state.drawing.sheets.length)
    box.append(h("p", { class: "note" }, "This model has no storey plan in the dataset: the elements are highlighted in the 3D model."));
  return box;
}

function recEl(card) {
  const on = card === state.active;
  const el = h("article", { class: "rec" + (on ? " on" : ""), onclick: () => { if (!on) activate(card); } },
    h("div", { class: "rec-head" }, h("span", {}, h("b", {}, `Q.${pad2(card.no)}`)), h("span", {}, card.at)),
    h("div", { class: "rec-q" }, card.question));
  if (card.error) { el.append(h("div", { class: "rec-a" }, h("div", { class: "val no" }, `The request failed: ${card.error}`))); return el; }
  if (!card.res) { el.append(h("div", { class: "wait" }, "Reading the drawing")); return el; }
  const r = card.res, [cls, top, sub] = stampOf(r);
  const stamp = h("div", { class: `stamp ${cls}` }, h("b", {}, top), h("i", {}, sub));
  if (!card.fresh) stamp.style.animation = "none";
  el.append(h("div", { class: "rec-a" }, valueEl(r), stamp), specEl(r));
  const crew = crewEl(r); if (crew) el.append(crew);
  const refs = refsEl(r); if (refs) el.append(refs);
  const why = whyEl(r); if (why) el.append(why);
  if (r.expected != null && r.correct == null)
    el.append(h("details", { class: "ref", onclick: (ev) => ev.stopPropagation() }, h("summary", {}, "Stoa's reference answer (free text, not scored)"), h("pre", {}, String(r.expected))));
  if (r.overlay.items.length) el.append(takeoffEl(card, on));
  if (r.overlay.legends?.length) el.append(legendEl(card));
  else if (r.value != null && !["planner", "none"].includes(r.engine.kind))
    el.append(h("p", { class: "note" }, "Nothing on the sheet could be marked for this answer, so treat it as an estimate."));
  return el;
}

function renderThread() {
  const t = $("#thread");
  if (!state.cards.length) {
    t.replaceChildren(h("div", { class: "blank", html: "<b>Nothing checked yet</b>Pick one of Stoa's questions above or type your own. Each answer is stamped, shows how it was worked out, and marks what it counted on the sheet." }));
    return;
  }
  t.replaceChildren(...state.cards.map(recEl));
}

// ------------------------------------------------------------------ X-ray view: how each mark was established
// basis from pipeline.overlay(): direct (geometry, BIM, sheet text), inferred (detectors), review (vision only).
// The auditor can overrule: a failed check, or a vision-only answer, puts every mark up for sign-off.
const BASIS = { direct: ["Direct", "read from geometry, the model or the sheet text"], inferred: ["Inferred", "detected from symbols / pixels"],
  review: ["Sign-off", "vision only or an auditor fail: a person must check it"] };
function needsSignOff(r) { return !!r && (r.engine?.kind === "vision" || (r.audit || []).some((c) => c.status === "fail")); }
function basisOf(it) {
  const r = state.active?.res;
  return needsSignOff(r) ? "review" : BASIS[it.basis] ? it.basis : "inferred";
}

function toggleHeat() {
  state.heatOn = !state.heatOn;
  $("#btn-heat").setAttribute("aria-pressed", state.heatOn);
  $("#stage").classList.toggle("heat", state.heatOn);
  drawHeatKey();
}

function drawHeatKey() {
  let key = $("#heat-key");
  if (!key) { key = h("div", { class: "heat-key", id: "heat-key", role: "note" }); $("#stage").append(key); }
  key.classList.toggle("hidden", !state.heatOn);
  if (!state.heatOn) return;
  const n = { direct: 0, inferred: 0, review: 0 };
  for (const it of state.items) n[basisOf(it)]++;
  const r = state.active?.res, why = needsSignOff(r) ? (r.engine.kind === "vision" ? "Vision-only answer" : "Auditor check failed") : null;
  key.replaceChildren(h("b", {}, "Confidence"),
    ...Object.entries(BASIS).map(([k, [label, sub]]) => h("span", { class: `k ${k}`, title: sub }, h("i", {}), label, h("em", {}, n[k]))),
    ...[why ? h("small", {}, `${why}: every mark needs sign-off`) : !state.items.length ? h("small", {}, "No marks on this sheet") : null].filter(Boolean));
}

function sheetOf(r, n) {  // index of the sheet that shows evidence item n
  const it = r.overlay.items.find((x) => x.n === n); if (!it) return -1;
  if (r.overlay.space === "guid") return it.sheet ? state.drawing.sheets.findIndex((x) => x.kind === "svg" && x.path === it.sheet) : -1;
  return state.drawing.sheets.findIndex((x) => x.kind === "image");  // pixel marks are drawn on the first image sheet
}

// the legend that defines the focused mark (opening_design sheets print one; other sources have none, so nothing is shown)
function legendPick(r) {
  const f = state.active?.res === r ? state.items[state.focus - 1] : null;
  return f?.legend || r.overlay.legend || [0, -1];
}

function legendEl(card) {
  const r = card.res, [li, ri] = legendPick(r), lg = r.overlay.legends[li];
  if (!lg) return null;
  const [x0, y0, x1, y1] = lg.bbox, row = lg.rows[ri], W = 300, k = W / Math.max(x1 - x0, 1);
  const viewH = Math.min(y1 - y0, 210 / k);
  const wy0 = row ? Math.min(Math.max(row.bbox[1] - viewH / 2, y0), y1 - viewH) : y0;
  const where = lg.own ? "this sheet" : lg.sheet;
  const thumb = h("button", { class: "lg-thumb", title: "Open the legend beside the drawing",
    style: { height: `${viewH * k}px`, backgroundImage: `url("${lg.src}")`, backgroundSize: `${lg.width * k}px ${lg.height * k}px`,
      backgroundPosition: `${-x0 * k}px ${-wy0 * k}px` },
    onclick: (ev) => { ev.stopPropagation(); openSplit({ legend: lg, row }); } },
    row ? h("u", { style: { left: `${(row.bbox[0] - x0) * k - 4}px`, top: `${(row.bbox[1] - wy0) * k - 3}px`,
      width: `${(row.bbox[2] - row.bbox[0]) * k + 8}px`, height: `${(row.bbox[3] - row.bbox[1]) * k + 6}px` } }) : null);
  return h("div", { class: "legend" },
    h("div", { class: "take-head" }, h("span", { class: "lbl" }, "Legend · ", h("b", {}, lg.title), ` · ${where}`),
      h("button", { class: "btn mini", onclick: (ev) => { ev.stopPropagation(); openSplit({ legend: lg, row }); } }, "Beside →")),
    thumb, h("p", { class: "note" }, row ? `Defines ${row.text}` : `${lg.rows.length} entries — no entry names this mark; the whole legend is shown.`));
}

function updateLegend() {
  const box = document.querySelector(".rec.on .legend"), card = state.active;
  if (!box || !card?.res?.overlay.legends?.length) return;
  const el = legendEl(card); if (el) box.replaceWith(el);
}

// split screen: a second sheet beside the drawing (another sheet of the set, a storey, or the legend sheet)
const v2 = {
  x: 0, y: 0, k: 1, W: 1, H: 1,
  root: null, canvas: null,
  fitK() { const r = this.root.getBoundingClientRect(); return Math.min((r.width - 40) / this.W, (r.height - 40) / this.H); },
  apply(anim) { this.canvas.classList.toggle("anim", !!anim); this.canvas.style.transform = `translate(${this.x}px, ${this.y}px) scale(${this.k})`; },
  fit(anim = false) { const r = this.root.getBoundingClientRect(); this.k = this.fitK(); this.x = (r.width - this.W * this.k) / 2; this.y = (r.height - this.H * this.k) / 2; this.apply(anim); },
  zoomTo([x0, y0, x1, y1]) {
    const r = this.root.getBoundingClientRect();
    this.k = Math.max(this.fitK(), Math.min(r.width / ((x1 - x0) * 1.15), r.height / ((y1 - y0) * 1.15), 3));
    this.x = r.width / 2 - ((x0 + x1) / 2) * this.k; this.y = r.height / 2 - ((y0 + y1) / 2) * this.k; this.apply(true);
  },
};

function initSplit() {
  const st = $("#stage"), wrap = h("div", { class: "stages" });
  st.replaceWith(wrap); wrap.append(st);
  const pane = h("section", { class: "stage2 hidden", id: "stage2", "aria-label": "Second sheet" },
    h("header", {}, h("span", { class: "lbl" }, "Beside"), h("select", { id: "s2-select", "aria-label": "Sheet shown beside the drawing" }),
      h("button", { class: "btn mini", id: "s2-fit" }, "Fit"), h("button", { class: "btn mini", id: "s2-close", "aria-label": "Close the second sheet" }, "×")),
    h("div", { class: "s2-view", id: "s2-view" }, h("div", { class: "canvas", id: "s2-canvas" })));
  wrap.append(pane);
  v2.root = $("#s2-view"); v2.canvas = $("#s2-canvas");
  $("#s2-close").onclick = closeSplit; $("#s2-fit").onclick = () => v2.fit(true);
  $("#s2-select").addEventListener("change", (e) => showSplit(state.split.options[+e.target.value]));
  let drag = null;
  v2.root.addEventListener("wheel", (e) => {
    e.preventDefault();
    const r = v2.root.getBoundingClientRect(), cx = e.clientX - r.left, cy = e.clientY - r.top;
    const k2 = Math.min(Math.max(v2.k * Math.exp(-e.deltaY * 0.0016), v2.fitK() * 0.4), v2.fitK() * 30);
    v2.x = cx - (cx - v2.x) * (k2 / v2.k); v2.y = cy - (cy - v2.y) * (k2 / v2.k); v2.k = k2; v2.apply(false);
  }, { passive: false });
  v2.root.addEventListener("pointerdown", (e) => { if (e.button === 0) drag = { x: e.clientX, y: e.clientY, ox: v2.x, oy: v2.y }; });
  window.addEventListener("pointermove", (e) => { if (drag) { v2.x = drag.ox + e.clientX - drag.x; v2.y = drag.oy + e.clientY - drag.y; v2.apply(false); } });
  window.addEventListener("pointerup", () => { drag = null; });
  new ResizeObserver(() => { if (state.split?.shown) v2.fit(false); }).observe(v2.root);
}

async function splitOptions() {
  const d = state.drawing, out = [];
  for (const lg of state.active?.res?.overlay.legends || [])
    out.push({ label: `Legend · ${lg.title} · ${lg.own ? "this sheet" : lg.sheet}`, src: lg.src, width: lg.width, height: lg.height, legend: lg });
  d.sheets.forEach((sh, i) => { if (i !== state.sheetIndex && sh.kind === "image") out.push({ label: `This drawing · ${sh.label}`, ...sh }); });
  // the rest of the set: same folder (real drawing sets) or the other floors of the same building (Swiss)
  const m = d.id.match(/^(data\/swiss_dwellings\/site\d+_bldg\d+)_/), folder = d.id.split("/").slice(0, -1).join("/");
  const q = m ? m[1].split("/").pop() : d.source === "opening_design" ? folder.split("/").slice(2).join(" ") : null;
  if (q) {
    try {
      const r = await api.get(`/api/drawings?${new URLSearchParams({ source: d.source, q, limit: 60 })}`);
      for (const it of r.items) if (it.id !== d.id && it.has_sheet) out.push({ label: `Set · ${it.title}`, drawing: it.id });
    } catch { /* the set list is optional */ }
  }
  return out;
}

async function openSplit(want = {}) {
  if (!state.drawing) return;
  const options = await splitOptions();
  if (!options.length) { setMsg("<b>Nothing to put beside this sheet</b>This drawing has no other sheets, set or legend."); setTimeout(() => setMsg(""), 2400); return; }
  state.split = { options, shown: null };
  const ov = state.active?.res?.overlay;
  if (!want.legend && ov?.legends?.length) {  // default: the legend entry that defines the answer's mark
    const [li, ri] = legendPick(state.active.res);
    want = { legend: ov.legends[li], row: ov.legends[li]?.rows[ri] };
  }
  $("#s2-select").replaceChildren(...options.map((o, i) => h("option", { value: i }, o.label)));
  $("#stage2").classList.remove("hidden"); $("#btn-split").setAttribute("aria-pressed", "true");
  const i = Math.max(0, want.legend ? options.findIndex((o) => o.legend && o.legend.path === want.legend.path && o.legend.title === want.legend.title) : 0);
  $("#s2-select").value = i;
  await showSplit(options[i], want.row);
}

function closeSplit() {
  state.split = null; $("#stage2").classList.add("hidden"); $("#btn-split").setAttribute("aria-pressed", "false");
}

async function showSplit(opt, row) {
  if (!opt || !state.split) return;
  if (opt.drawing && !opt.src) {  // a sheet of another drawing in the set: fetch its first sheet
    try { const d = await api.get(`/api/drawing?id=${encodeURIComponent(opt.drawing)}`); Object.assign(opt, d.sheets.find((x) => x.kind === "image") || {}); } catch { return; }
    if (!opt.src) return;
  }
  state.split.shown = opt; v2.W = opt.width; v2.H = opt.height;
  const vb = `0 0 ${opt.width} ${opt.height}`, mark = s("svg", { class: "ov tg", viewBox: vb });
  const lg = opt.legend;
  if (lg) {
    const [x0, y0, x1, y1] = lg.bbox;
    mark.append(s("rect", { class: "lg-box", x: x0, y: y0, width: x1 - x0, height: y1 - y0, "vector-effect": "non-scaling-stroke" }));
    if (row) {
      const [a, b, c, d] = row.bbox;
      for (const cls of ["pulse ring", "pulse wave"]) mark.append(s("rect", { class: cls, x: a - 6, y: b - 5, width: c - a + 12, height: d - b + 10, rx: 4, "vector-effect": "non-scaling-stroke" }));
    }
  }
  v2.canvas.replaceChildren(h("div", { class: "sheet", style: { width: `${opt.width}px`, height: `${opt.height}px` } },
    h("img", { src: opt.src, alt: opt.label, draggable: "false" }), mark));
  await new Promise((r) => setTimeout(r, 40));  // let the pane lay out before measuring it
  v2.fit(false);
  if (lg) requestAnimationFrame(() => v2.zoomTo(row ? [Math.min(row.bbox[0], lg.bbox[0]), row.bbox[1] - 60, Math.max(row.bbox[2], lg.bbox[2]), row.bbox[3] + 60] : lg.bbox));
}

// ------------------------------------------------------------------ test report
async function openReport() {
  const dlg = $("#bench"), body = $("#bench-body");
  body.replaceChildren(h("div", { class: "wait" }, "Loading the report"));
  dlg.showModal();
  let b;
  try { b = await api.get("/api/benchmark"); } catch (e) { body.replaceChildren(h("div", { class: "group" }, h("p", {}, `The report is unavailable: ${e.message}`))); return; }
  body.replaceChildren(
    h("div", { class: "kpis" }, ...(b.kpis || []).map((k) => h("div", { class: "kpi" }, h("b", {}, k.value), h("span", {}, k.label)))),
    ...(b.groups || []).map((g) => h("section", { class: "group" }, h("h3", {}, g.title), g.note && h("p", {}, g.note),
      ...g.rows.map((r) => h("div", { class: "brow" }, h("div", {}, h("b", {}, r.label), h("small", {}, r.sub || "")),
        h("div", { class: `bar ${r.kind === "ev" ? "ev" : r.kind === "base" ? "" : "top"}`, role: "img", "aria-label": `${r.pct.toFixed(1)} percent` }, h("i", { style: { width: `${r.pct}%` } })),
        h("output", {}, `${r.pct.toFixed(1)}%`))))),
    h("div", { class: "fine" }, ...(b.notes || []).map((n) => h("p", {}, n)), b.generated && h("p", {}, `Generated ${b.generated} from the evaluation runs in this repository.`)));
}

// ------------------------------------------------------------------ wiring
function initVoice() {
  const SR = window.SpeechRecognition || window.webkitSpeechRecognition, mic = $("#mic");
  if (!SR) { mic.title = "Voice input needs Chrome or Edge"; mic.disabled = true; return; }
  const rec = new SR(); rec.interimResults = true; rec.lang = navigator.language || "en-US";
  let live = false;
  rec.onresult = (e) => { $("#q").value = [...e.results].map((r) => r[0].transcript).join(""); if (e.results[e.results.length - 1].isFinal) { rec.stop(); ask($("#q").value); } };
  rec.onend = () => { live = false; mic.classList.remove("live"); };
  rec.onerror = () => { live = false; mic.classList.remove("live"); };
  mic.onclick = () => { if (live) rec.stop(); else { live = true; mic.classList.add("live"); rec.start(); } };
}

// ------------------------------------------------------------------ upload a plan of your own
async function uploadPlan(file) {
  const msg = $("#up-msg");
  if (!file) return;
  if (!/^image\/(png|jpeg)$/.test(file.type)) { msg.textContent = "Use a PNG or JPEG image of a floor plan."; return; }
  msg.textContent = `Uploading ${file.name}…`;
  let d;
  try {
    const r = await fetch("/api/upload", { method: "POST", headers: { "content-type": file.type }, body: file });
    if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || `${r.status}`);
    d = await r.json();
  } catch (e) { msg.textContent = `Upload failed: ${e.message}`; return; }
  const det = d.detector || {};
  msg.textContent = det.available?.length ? `${file.name} · detectors: ${det.available.join(" + ")}`
    : `${file.name} · no detector installed on this server: counts fall back to a vision model or are refused`;
  await openDrawing(d.id);
}

function initUpload() {
  $("#up-file").addEventListener("change", (e) => { uploadPlan(e.target.files[0]); e.target.value = ""; });
  const root = stage.root, files = (e) => [...(e.dataTransfer?.types || [])].includes("Files");
  root.addEventListener("dragover", (e) => { if (!files(e)) return; e.preventDefault(); root.classList.add("drop"); });
  root.addEventListener("dragleave", () => root.classList.remove("drop"));
  root.addEventListener("drop", (e) => { if (!files(e)) return; e.preventDefault(); root.classList.remove("drop"); uploadPlan(e.dataTransfer.files[0]); });
}

function init() {
  stage.init(); initVoice(); initUpload(); renderMeter(); renderThread();
  let timer;
  $("#search").addEventListener("input", (e) => { clearTimeout(timer); timer = setTimeout(() => { state.q = e.target.value.trim(); loadList(); }, 220); });
  $("#only-q").addEventListener("change", (e) => { state.onlyQ = e.target.checked; loadList(); });
  $("#ask").addEventListener("submit", (e) => { e.preventDefault(); ask($("#q").value); });
  $("#q").addEventListener("keydown", (e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); ask($("#q").value); } });
  $("#sheet-select").addEventListener("change", (e) => { state.focus = 0; showSheet(+e.target.value).then(renderThread); });
  $("#tab-stoa").onclick = () => { state.tab = "stoa"; renderSuggest(); }; $("#tab-free").onclick = () => { state.tab = "free"; renderSuggest(); };
  document.querySelectorAll("#view-seg button").forEach((b) => b.addEventListener("click", () => { if (!b.disabled) setView(b.dataset.view); }));
  $("#btn-graph").addEventListener("click", () => { if (!$("#btn-graph").disabled) toggleGraph(); });
  $("#btn-heat").addEventListener("click", toggleHeat);
  $("#btn-split").addEventListener("click", () => (state.split ? closeSplit() : openSplit()));
  initSplit();
  $("#st-prev").onclick = () => focusItem(state.focus - 1); $("#st-next").onclick = () => focusItem(state.focus + 1);
  $("#st-all").onclick = () => { state.focus = 0; applyFocus(); updateStepper(); };
  $("#open-bench").onclick = openReport; $("#close-bench").onclick = () => $("#bench").close();
  $("#bench").addEventListener("click", (e) => { if (e.target.id === "bench") $("#bench").close(); });
  window.addEventListener("keydown", (e) => {
    if ($("#bench").open) return;
    if (e.target.matches("textarea, select, input:not([type=checkbox])")) { if (e.key === "Escape") e.target.blur(); return; }
    if (e.key === "/") { e.preventDefault(); $("#q").focus(); }
    else if (e.key === "ArrowRight") focusItem(state.focus + 1);
    else if (e.key === "ArrowLeft") focusItem(state.focus - 1);
    else if (e.key === "Escape") { state.focus = 0; applyFocus(); updateStepper(); }
  });
  const fromHash = () => (location.hash.length > 1 ? decodeURIComponent(location.hash.slice(1)) : null);
  window.addEventListener("hashchange", () => { const id = fromHash(); if (id && id !== state.drawing?.id) openDrawing(id); });
  loadConfig(); loadList();
  openDrawing(fromHash() || DEMOS[0][1]);
}
init();

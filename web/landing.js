// OQVAR landing: the sample-sheet demo, the live register count and the benchmark chart.
const $ = (s) => document.querySelector(s);
const reduceMotion = matchMedia("(prefers-reduced-motion: reduce)").matches;

// ------------------------------------------------------------------ demo: pins land one by one, the count follows, then the stamp
const DEMO = {
  doors:   { q: "How many doors are on this floor?", n: 7, unit: "Doors", sub: "Level 2 · every door leaf", engine: "Geometry", ev: "7 marks on the sheet", say: "7 doors" },
  sauna:   { q: "Kuinka monta saunaa?", n: 1, unit: "Sauna", sub: "Taso 2 · 4,2 m²", engine: "Detection hybrid", ev: "1 room on the sheet", say: "1 sauna" },
  windows: { q: "Wie viele Fenster?", n: 6, unit: "Fenster", sub: "Ebene 2 · alle Fenster", engine: "Geometry", ev: "6 marks on the sheet", say: "6 windows" },
};
const PIN_STEP = 170, PIN_START = 120;  // keep in sync with the .mk animation delay in landing.css
let timers = [];

function play(key) {
  const d = DEMO[key];
  timers.forEach(clearTimeout); timers = [];
  $("#sheet").dataset.active = key;
  document.querySelectorAll(".asks button").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.q === key)));
  $("#answer-q").textContent = d.q;
  $("#unit").textContent = d.unit;
  $("#unit-sub").textContent = d.sub;
  $("#engine").textContent = d.engine;
  $("#tokens").textContent = "0";
  $("#evid").textContent = d.ev;
  const answer = $("#answer");
  answer.classList.remove("done");
  const finish = () => {
    $("#num").textContent = d.n;
    answer.classList.add("done");
    $("#announce").textContent = `${d.say}. Checked: matches the drawing. ${d.ev}.`;
  };
  if (reduceMotion) return finish();
  $("#num").textContent = "0";
  for (let i = 1; i <= d.n; i++) timers.push(setTimeout(() => { $("#num").textContent = i; }, PIN_START + (i - 1) * PIN_STEP + 180));
  timers.push(setTimeout(finish, PIN_START + d.n * PIN_STEP + 260));
}
document.querySelectorAll(".asks button").forEach((b) => b.addEventListener("click", () => play(b.dataset.q)));
play("doors");

// ------------------------------------------------------------------ live register size from the running app
fetch("/api/config").then((r) => (r.ok ? r.json() : null)).then((c) => {
  if (!c || !c.drawings) return;
  $("#live-d").textContent = c.drawings.toLocaleString("en-US");
  $("#live-q").textContent = c.questions.toLocaleString("en-US");
  $("#live").hidden = false;
}).catch(() => {});

// ------------------------------------------------------------------ benchmark: one series, value at the bar tip, hover/focus tooltip, table view
const el = (tag, cls, text) => { const n = document.createElement(tag); if (cls) n.className = cls; if (text != null) n.textContent = text; return n; };
const pct = (v) => `${v.toFixed(v >= 99 ? 2 : 1)}%`;

function renderBenchmark(data) {
  const plot = $("#bm-plot"), tbody = $("#bm-table tbody"), tip = $("#bm-tip"), fig = $("#bm");
  // OQVAR's own results only; the baseline models stay in the full report in the app
  const groups = data.groups.map((g) => ({ ...g, rows: g.rows.filter((r) => r.kind !== "base") })).filter((g) => g.rows.length);
  if (data.generated) $("#bm-gen").textContent = `by drawing type · measured ${data.generated}`;

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
      const val = el("span", `bm-val${r.pct > 82 ? " inside" : ""}`, pct(r.pct));
      track.append(el("span", "bm-bar"), val);
      row.append(lbl, track);
      row.addEventListener("pointermove", (e) => showTip(r, e.clientX, e.clientY));
      row.addEventListener("pointerleave", hideTip);
      row.addEventListener("focus", () => { const b = track.getBoundingClientRect(); showTip(r, b.left + b.width * r.pct / 100, b.bottom); });
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

  // bars grow from the baseline once, when the chart scrolls into view
  if (reduceMotion || !("IntersectionObserver" in window)) fig.classList.add("in");
  else new IntersectionObserver((entries, obs) => {
    if (entries.some((e) => e.isIntersecting)) { fig.classList.add("in"); obs.disconnect(); }
  }, { threshold: 0.15 }).observe(fig);

  $("#bm-toggle").addEventListener("click", (e) => {
    const table = !$("#bm-table").hidden;
    $("#bm-table").hidden = table;
    plot.hidden = !table;
    hideTip();
    e.currentTarget.setAttribute("aria-expanded", String(!table));
    e.currentTarget.textContent = table ? "Show as table" : "Show as chart";
  });
}

fetch("/web/benchmark.json").then((r) => (r.ok ? r.json() : Promise.reject(r.status))).then(renderBenchmark).catch(() => {
  $("#bm-plot").hidden = true;
  $("#bm-toggle").hidden = true;
  $("#bm-empty").hidden = false;
});

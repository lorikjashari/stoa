"""Generate OQVAR product mockup pages (HTML) to be rendered to PNG by headless Chromium."""
import os, sys

T = sys.argv[1]  # tmp dir holding mark/wordmark path data
OUT = os.path.join(T, "pages"); os.makedirs(OUT, exist_ok=True)
rd = lambda f: open(os.path.join(T, f)).read().strip()
BODY, EYE, WM = rd("mark_body.d"), rd("mark_eye.d"), rd("wordmark.d")
WMW, CAP = map(float, rd("wordmark.meta").split())


def mark(cls="mark", eye="currentColor"):
    return (f'<svg class="{cls}" viewBox="467 465 1114 1117" aria-hidden="true">'
            f'<path fill="currentColor" fill-rule="evenodd" d="{BODY}"/><path fill="{eye}" d="{EYE}"/></svg>')


def wordmark(cls="wm"):
    return f'<svg class="{cls}" viewBox="0 0 {WMW:.0f} {CAP:.0f}" aria-hidden="true"><path fill="currentColor" d="{WM}"/></svg>'


def lockup(cls="lockup", eye="currentColor"):
    return f'<span class="{cls}">{mark(eye=eye)}{wordmark()}</span>'


# ---------------------------------------------------------------- the floor plan (800 x 560 units)
def door(x, y, s, kind, n=None):
    """kind: v+ / v- (door in a vertical wall, swings right/left), h+ / h- (horizontal wall, swings down/up)."""
    o = []
    if kind[0] == "v":
        sx = 1 if kind[1] == "+" else -1
        o.append(f'<rect class="hlr" x="{min(x, x + sx * s) - 4}" y="{y - 4}" width="{s + 8}" height="{s + 8}"/>')
        o.append(f'<rect class="gap" x="{x - 6}" y="{y}" width="12" height="{s}"/>')
        o.append(f'<path class="leaf" d="M{x} {y}L{x + sx * s} {y}"/>')
        o.append(f'<path class="swing" d="M{x + sx * s} {y}A{s} {s} 0 0 {1 if sx > 0 else 0} {x} {y + s}"/>')
        cx, cy = x + sx * (s + 14), y + s / 2
    else:
        sy = 1 if kind[1] == "+" else -1
        o.append(f'<rect class="hlr" x="{x - 4}" y="{min(y, y + sy * s) - 4}" width="{s + 8}" height="{s + 8}"/>')
        o.append(f'<rect class="gap" x="{x}" y="{y - 6}" width="{s}" height="12"/>')
        o.append(f'<path class="leaf" d="M{x} {y}L{x} {y + sy * s}"/>')
        o.append(f'<path class="swing" d="M{x} {y + sy * s}A{s} {s} 0 0 {0 if sy > 0 else 1} {x + s} {y}"/>')
        cx, cy = x + s / 2, y + sy * (s + 14)
    if n is not None:
        o.append(f'<g class="pin"><circle cx="{cx}" cy="{cy}" r="11"/><text x="{cx}" y="{cy + 4}">{n}</text></g>')
    return "".join(o)


def plan(highlight="doors"):
    walls_outer = '<rect class="wall-o" x="40" y="40" width="720" height="480"/>'
    inner = [
        "M420 40V520", "M420 240H760", "M620 240V520", "M620 380H760", "M40 320H420", "M240 320V520",
    ]
    doors = [
        (420, 120, 46, "v-"), (520, 240, 44, "h+"), (620, 290, 40, "v+"), (620, 430, 40, "v+"),
        (300, 320, 44, "h-"), (240, 400, 44, "v-"), (330, 520, 54, "h-"),
    ]
    saunas = highlight == "sauna"
    rooms = [
        (230, 180, "LIVING", "32.4 m²"), (590, 140, "KITCHEN", "18.6 m²"), (520, 400, "BEDROOM 1", "14.8 m²"),
        (708, 310, "BATH", "6.1 m²"), (714, 470, "SAUNA", "4.2 m²"), (140, 420, "BEDROOM 2", "11.9 m²"),
        (330, 430, "HALL", "9.7 m²"),
    ]
    s = ['<svg class="plan" viewBox="0 0 800 560" xmlns="http://www.w3.org/2000/svg">',
         '<defs><pattern id="grid" width="20" height="20" patternUnits="userSpaceOnUse"><path d="M20 0H0V20" fill="none" stroke="rgba(30,60,120,.07)" stroke-width="1"/></pattern></defs>',
         '<rect width="800" height="560" fill="url(#grid)"/>']
    if saunas:
        s.append('<rect class="hlroom" x="625" y="385" width="130" height="130"/>')
    # fixtures
    s.append('<g class="fx">'
             '<rect x="440" y="52" width="300" height="34"/><circle cx="480" cy="69" r="10"/><circle cx="510" cy="69" r="10"/><rect x="560" y="56" width="44" height="26" rx="3"/>'
             '<rect x="704" y="250" width="44" height="80" rx="18"/><rect x="630" y="250" width="40" height="30" rx="10"/>'
             '<path d="M632 396H752M632 420H752M632 444H752"/>'
             '<rect x="440" y="440" width="120" height="70"/><rect x="60" y="440" width="110" height="70"/>'
             '<rect x="70" y="70" width="150" height="70" rx="6"/><rect x="250" y="210" width="110" height="80" rx="40"/>'
             '</g>')
    s.append(walls_outer)
    s.append('<path class="wall-i" d="' + " ".join(inner) + '"/>')
    for i, (x, y, w, k) in enumerate(doors, 1):
        s.append(door(x, y, w, k, i if highlight == "doors" else None))
    # windows
    for x, y, w, h in [(120, 36, 140, 8), (520, 36, 120, 8), (756, 120, 8, 80), (36, 120, 8, 140), (36, 380, 8, 90), (440, 516, 120, 8)]:
        s.append(f'<rect class="win" x="{x}" y="{y}" width="{w}" height="{h}"/>')
    for x, y, a, b in rooms:
        cls = "rl hot" if saunas and a == "SAUNA" else "rl"
        s.append(f'<text class="{cls}" x="{x}" y="{y}">{a}</text><text class="ra" x="{x}" y="{y + 15}">{b}</text>')
    if saunas:
        s.append('<g class="pin"><circle cx="740" cy="400" r="11"/><text x="740" y="404">1</text></g>')
    # dimension line
    s.append('<g class="dim"><path d="M40 22H760M40 16V28M420 16V28M760 16V28"/><text x="230" y="17">7 400</text><text x="590" y="17">6 800</text></g>')
    s.append("</svg>")
    return "".join(s)


FONTS = '<link href="https://fonts.googleapis.com/css2?family=Sora:wght@400;600;700;800&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500;600&display=block" rel="stylesheet">'

BASE = """
:root{--paper:#f2eee4;--desk:#e7e2d5;--card:#fbfaf6;--ink:#15171b;--ink-2:#4b5059;--ink-3:#5d6269;--rule:rgba(21,23,27,.15);
--red:#cf2a1e;--hl:#ffe24a;--green:#156f45;--blue:#1f4fa8;
--brand:"Sora",system-ui,sans-serif;--sans:"IBM Plex Sans",system-ui,sans-serif;--mono:"IBM Plex Mono",ui-monospace,monospace}
*{box-sizing:border-box;margin:0;padding:0}
html,body{width:VW;height:VH;overflow:hidden}
body{background:var(--paper);color:var(--ink);font:15px/1.45 var(--sans);-webkit-font-smoothing:antialiased}
.lockup{display:inline-flex;align-items:center;gap:.42em;color:inherit}
.lockup .mark{height:1.25em;width:auto}.lockup .wm{height:.6em;width:auto;overflow:visible}
.wm{overflow:visible}
.lbl{font:500 10px/1 var(--mono);letter-spacing:.12em;text-transform:uppercase;color:var(--ink-3)}
/* plan */
.plan{display:block;width:100%;height:auto;background:#fdfcf8}
.plan .wall-o{fill:none;stroke:var(--ink);stroke-width:12}
.plan .wall-i{fill:none;stroke:var(--ink);stroke-width:7}
.plan .gap{fill:#fdfcf8}
.plan .leaf{stroke:var(--ink);stroke-width:2.5}
.plan .swing{fill:none;stroke:var(--ink);stroke-width:1;stroke-dasharray:3 3}
.plan .hlr{fill:var(--hl);opacity:.75}
.plan .hlroom{fill:var(--hl);opacity:.55}
.plan .win{fill:#fdfcf8;stroke:var(--ink);stroke-width:1.5}
.plan .fx{fill:none;stroke:rgba(21,23,27,.45);stroke-width:1.2}
.plan .rl{font:600 11px var(--mono);letter-spacing:.08em;fill:var(--ink);text-anchor:middle}
.plan .rl.hot{fill:var(--red)}
.plan .ra{font:400 10px var(--mono);fill:var(--ink-3);text-anchor:middle}
.plan .pin circle{fill:var(--red);stroke:#fff;stroke-width:2.5}
.plan .pin text{font:700 11px var(--mono);fill:#fff;text-anchor:middle}
.plan .dim path{stroke:var(--ink-3);stroke-width:1;fill:none}.plan .dim text{font:400 10px var(--mono);fill:var(--ink-3);text-anchor:middle}
/* app chrome */
.app{background:var(--paper);color:var(--ink);display:flex;flex-direction:column;overflow:hidden}
.tb{height:46px;display:flex;align-items:stretch;background:var(--card);border-bottom:2px solid var(--ink);flex:none}
.tb-brand{display:flex;align-items:center;padding:0 16px;border-right:1px solid var(--ink);font-size:26px}
.tb-cell{display:flex;flex-direction:column;justify-content:center;gap:5px;padding:0 14px;border-right:1px solid var(--ink);white-space:nowrap}
.tb-cell dd{font:500 12px/1 var(--mono)}
.tb-fill{flex:1;background:repeating-linear-gradient(135deg,transparent 0 7px,var(--rule) 7px 8px)}
.tb-btn{background:var(--ink);color:var(--paper);display:flex;align-items:center;padding:0 16px;font:600 10px/1 var(--mono);letter-spacing:.1em;text-transform:uppercase}
.desk{flex:1;display:grid;min-height:0}
.reg{border-right:1px solid var(--ink);display:flex;flex-direction:column}
.ph{display:flex;justify-content:space-between;align-items:baseline;padding:10px 12px;border-bottom:1px solid var(--ink)}
.ph h2{font:600 10px/1 var(--mono);letter-spacing:.12em;text-transform:uppercase}
.ph span{font:400 10px var(--mono);color:var(--ink-3)}
.row{display:flex;gap:10px;align-items:center;padding:9px 12px;border-bottom:1px solid var(--rule)}
.row i{width:38px;height:28px;flex:none;border:1px solid var(--rule);background:#fdfcf8 repeating-linear-gradient(90deg,transparent 0 9px,rgba(21,23,27,.12) 9px 10px)}
.row b{display:block;font:600 11px/1.2 var(--mono)}.row small{display:block;font:400 10px/1.3 var(--mono);color:var(--ink-3)}
.row.on{background:var(--ink);color:var(--paper)}.row.on small{color:rgba(242,238,228,.65)}
.board{background:var(--desk);display:flex;flex-direction:column;min-width:0}
.sheet{margin:18px;flex:1;background:#fdfcf8;border:1px solid var(--ink);box-shadow:4px 4px 0 var(--ink);display:flex;flex-direction:column;overflow:hidden}
.sheet .plan{flex:1}
.titleblk{display:grid;grid-template-columns:1fr auto auto auto;border-top:1.5px solid var(--ink);font:500 10px/1 var(--mono)}
.titleblk div{padding:7px 10px;border-left:1px solid var(--ink)}.titleblk div:first-child{border-left:0}
.titleblk em{display:block;font-style:normal;color:var(--ink-3);font-size:8px;letter-spacing:.12em;margin-bottom:4px}
.chk{border-left:1px solid var(--ink);display:flex;flex-direction:column;background:var(--paper)}
.ask{margin:12px;background:var(--card);border:1.5px solid var(--ink);display:flex;align-items:center;gap:10px;padding:10px 12px;font:400 13px var(--sans)}
.ask .go{margin-left:auto;background:var(--ink);color:var(--paper);font:600 10px var(--mono);letter-spacing:.1em;padding:7px 10px}
.card{margin:0 12px 12px;background:var(--card);border:1.5px solid var(--ink);box-shadow:4px 4px 0 var(--ink)}
.card .q{padding:12px 14px;border-bottom:1px solid var(--rule);font:500 13px/1.35 var(--sans)}
.card .q small{display:block;margin-bottom:5px}
.card .a{display:flex;align-items:center;gap:14px;padding:14px}
.card .num{font:800 64px/.85 var(--brand);letter-spacing:-.02em}
.card .unit{font:600 13px var(--mono);letter-spacing:.06em;text-transform:uppercase}
.card .unit small{display:block;font:400 11px var(--sans);letter-spacing:0;text-transform:none;color:var(--ink-3);margin-top:4px}
.stamp{margin-left:auto;transform:rotate(-7deg);border:2.5px solid var(--green);color:var(--green);padding:6px 9px;font:700 11px/1.1 var(--mono);letter-spacing:.12em;text-align:center;border-radius:3px}
.stamp small{display:block;font-size:8px;letter-spacing:.08em;font-weight:500}
.meta{display:grid;grid-template-columns:repeat(4,1fr);border-top:1px solid var(--rule)}
.meta div{padding:9px 12px;border-left:1px solid var(--rule)}.meta div:first-child{border-left:0}
.meta b{display:block;font:600 12px var(--mono);margin-top:5px}
.ev{padding:4px 14px 12px}
.ev li{list-style:none;display:flex;align-items:center;gap:9px;padding:6px 0;border-bottom:1px dashed var(--rule);font:400 11px var(--mono)}
.ev li span{width:18px;height:18px;border-radius:50%;background:var(--red);color:#fff;font:700 10px/18px var(--mono);text-align:center;flex:none}
.ev li em{margin-left:auto;font-style:normal;color:var(--ink-3)}
"""


def app_ui(q, ql, num, unit, unit_sub, evidence, hl="doors", reg=True, lang="EN"):
    regpane = ""
    if reg:
        rows = [("A-101", "Ground floor plan", False), ("A-102", "Level 2 floor plan", True), ("A-103", "Level 3 floor plan", False),
                ("A-201", "Sections A–A, B–B", False), ("A-501", "Door schedule", False), ("S-100", "Foundation plan", False),
                ("IFC", "Building model · 4 storeys", False)]
        regpane = ('<nav class="reg"><div class="ph"><h2>Drawing register</h2><span>13,943 sheets</span></div>'
                   + "".join(f'<div class="row{" on" if on else ""}"><i></i><div><b>{a}</b><small>{b}</small></div></div>' for a, b, on in rows)
                   + "</nav>")
    ev = "".join(f'<li><span>{i}</span>{a}<em>{b}</em></li>' for i, (a, b) in enumerate(evidence, 1))
    return f"""
<div class="app">
  <header class="tb">
    <div class="tb-brand">{lockup()}</div>
    <dl class="tb-cell"><dt class="lbl">Project</dt><dd>Harbour Block C</dd></dl>
    <dl class="tb-cell"><dt class="lbl">Sheet</dt><dd>A-102 · Level 2</dd></dl>
    <span class="tb-fill"></span>
    <dl class="tb-cell"><dt class="lbl">Language</dt><dd>{lang}</dd></dl>
    <div class="tb-btn">Share check</div>
  </header>
  <main class="desk" style="grid-template-columns:{'210px ' if reg else ''}minmax(0,1fr) 360px">
    {regpane}
    <section class="board"><div class="sheet">{plan(hl)}
      <div class="titleblk"><div><em>DRAWING</em>Level 2 — floor plan</div><div><em>NO.</em>A-102</div><div><em>SCALE</em>1:100</div><div><em>REV</em>C</div></div>
    </div></section>
    <aside class="chk">
      <div class="ask"><span class="lbl" style="color:var(--ink)">Ask</span><span style="color:var(--ink-3)">Ask anything about this sheet…</span><span class="go">ASK ↵</span></div>
      <article class="card">
        <div class="q"><small class="lbl">Question · {ql}</small>{q}</div>
        <div class="a"><div class="num">{num}</div><div class="unit">{unit}<small>{unit_sub}</small></div>
          <div class="stamp">CHECKED<small>MATCHES DRAWING</small></div></div>
        <div class="meta"><div><span class="lbl">Engine</span><b>Geometry</b></div><div><span class="lbl">Confidence</span><b style="color:var(--green)">High</b></div>
          <div><span class="lbl">Time</span><b>0.21 s</b></div><div><span class="lbl">LLM cost</span><b>$0.00</b></div></div>
        <ol class="ev">{ev}</ol>
      </article>
    </aside>
  </main>
</div>"""


DOORS_EV = [("D1 · Living ↔ Kitchen", "900 mm"), ("D2 · Kitchen ↔ Bed 1", "900 mm"), ("D3 · Bed 1 ↔ Bath", "800 mm"),
            ("D4 · Bed 1 ↔ Sauna", "800 mm"), ("D5 · Living ↔ Hall", "900 mm"), ("D6 · Hall ↔ Bed 2", "900 mm"),
            ("D7 · Entrance", "1000 mm")]


def page(name, w, h, css, body):
    html = f"""<!doctype html><html><head><meta charset="utf-8">{FONTS}<style>{BASE.replace('VW', f'{w}px').replace('VH', f'{h}px')}{css}</style></head><body>{body}</body></html>"""
    open(os.path.join(OUT, name + ".html"), "w").write(html)


# ---------------------------------------------------------------- 1. hero
page("oqvar-hero", 1920, 1200, """
body{background:var(--paper) radial-gradient(circle at 1px 1px,rgba(21,23,27,.13) 1px,transparent 0) 0 0/24px 24px}
.nav{position:absolute;left:96px;right:96px;top:44px;display:flex;align-items:center;gap:40px;font:500 15px var(--sans)}
.nav .lockup{font-size:40px;margin-right:auto}
.nav .cta{background:var(--ink);color:var(--paper);padding:12px 20px;font:600 13px var(--mono);letter-spacing:.08em;text-transform:uppercase}
.copy{position:absolute;left:96px;top:250px;width:640px}
.eyebrow{display:inline-flex;gap:10px;align-items:center;border:1.5px solid var(--ink);padding:8px 12px;font:600 12px var(--mono);letter-spacing:.12em;text-transform:uppercase;background:var(--card)}
.eyebrow i{width:8px;height:8px;background:var(--red);border-radius:50%}
h1{font:800 92px/.98 var(--brand);letter-spacing:-.035em;margin:30px 0 28px}
h1 span{background:linear-gradient(transparent 58%,var(--hl) 58% 92%,transparent 92%)}
.sub{font:400 21px/1.5 var(--sans);color:var(--ink-2);max-width:560px}
.btns{display:flex;gap:14px;margin-top:40px}
.b1{background:var(--ink);color:var(--paper);padding:18px 26px;font:600 14px var(--mono);letter-spacing:.08em;text-transform:uppercase;box-shadow:5px 5px 0 var(--red)}
.b2{border:1.5px solid var(--ink);padding:17px 24px;font:600 14px var(--mono);letter-spacing:.08em;text-transform:uppercase;background:var(--card)}
.stats{display:flex;gap:0;margin-top:72px;border-top:1.5px solid var(--ink)}
.stats div{padding:18px 30px 0 0;margin-right:30px;border-right:1px solid var(--rule)}.stats div:last-child{border:0}
.stats b{display:block;font:800 40px/1 var(--brand);letter-spacing:-.02em}
.stats span{display:block;margin-top:8px;font:500 11px var(--mono);letter-spacing:.1em;text-transform:uppercase;color:var(--ink-3)}
.bw{position:absolute;left:790px;top:200px;width:1300px;height:1010px;transform:scale(.8);transform-origin:0 0;border:2px solid var(--ink);background:var(--card);box-shadow:12px 12px 0 var(--ink);display:flex;flex-direction:column}
.bar{height:38px;flex:none;display:flex;align-items:center;gap:8px;padding:0 14px;border-bottom:1.5px solid var(--ink);background:var(--desk)}
.bar i{width:11px;height:11px;border-radius:50%;border:1.5px solid var(--ink)}
.bar .url{margin-left:18px;background:var(--card);border:1px solid var(--rule);padding:4px 14px;font:400 12px var(--mono);color:var(--ink-2);width:420px}
.bw .app{flex:1}
""", f"""
<div class="nav">{lockup()}<span>Product</span><span>Benchmark</span><span>Pricing</span><span>Docs</span><span class="cta">Request access</span></div>
<div class="copy">
  <span class="eyebrow"><i></i>Blueprint intelligence for construction</span>
  <h1>Ask the drawing.<br><span>Check the answer.</span></h1>
  <p class="sub">OQVAR reads construction drawings and BIM models, answers questions in any language, and marks the exact place on the sheet each answer came from.</p>
  <div class="btns"><span class="b1">Try it on your drawings</span><span class="b2">See the benchmark</span></div>
  <div class="stats"><div><b>99.5%</b><span>CAD floor plans</span></div><div><b>99.9%</b><span>IFC models</span></div><div><b>9</b><span>Languages</span></div></div>
</div>
<div class="bw"><div class="bar"><i></i><i></i><i></i><span class="url">app.oqvar.com/harbour-block-c/A-102</span></div>
{app_ui("How many doors are on this floor?", "English", "7", "Doors", "Level 2 · all swing doors", DOORS_EV)}
</div>
""")

# ---------------------------------------------------------------- 2. on site: tablet + phone, Finnish question
page("oqvar-on-site", 1920, 1200, """
body{background:var(--ink);color:var(--paper)}
.wmk{position:absolute;right:-180px;top:-120px;width:1000px;color:rgba(242,238,228,.045)}
.wmk .mark{width:100%;height:auto}
.copy{position:absolute;left:96px;top:96px;width:440px}
.copy .lockup{font-size:38px;color:var(--paper)}
h2{font:800 68px/1.02 var(--brand);letter-spacing:-.03em;margin:84px 0 26px}
h2 em{font-style:normal;color:var(--hl)}
.copy p{font:400 19px/1.55 var(--sans);color:rgba(242,238,228,.72)}
.langs{display:flex;flex-wrap:wrap;gap:8px;margin-top:36px}
.langs span{border:1px solid rgba(242,238,228,.35);padding:8px 11px;font:500 13px var(--mono)}
.langs span.on{background:var(--hl);color:var(--ink);border-color:var(--hl)}
.tab{position:absolute;left:640px;top:170px;width:1180px;height:830px;background:#0b0c0e;border-radius:44px;padding:26px;box-shadow:0 40px 90px rgba(0,0,0,.55),0 0 0 2px #2a2d33 inset}
.tab .scr{width:100%;height:100%;border-radius:20px;overflow:hidden;display:flex}
.tab .app{flex:1}
.phone{position:absolute;left:560px;top:600px;width:330px;height:640px;background:#0b0c0e;border-radius:52px;padding:14px;box-shadow:0 30px 80px rgba(0,0,0,.6),0 0 0 2px #2a2d33 inset}
.phone .scr{width:100%;height:100%;border-radius:40px;overflow:hidden;background:var(--paper);color:var(--ink);display:flex;flex-direction:column}
.phone .top{display:flex;align-items:center;justify-content:space-between;padding:44px 18px 12px;border-bottom:2px solid var(--ink);background:var(--card);font-size:22px}
.phone .top .lbl{font-size:9px}
.phone .mini{margin:12px;border:1px solid var(--ink);background:#fdfcf8;box-shadow:3px 3px 0 var(--ink)}
.phone .card{margin:0 12px}.phone .card .num{font-size:54px}.phone .meta{grid-template-columns:1fr 1fr}
.phone .meta div:nth-child(3){border-left:0}.phone .meta div:nth-child(n+3){border-top:1px solid var(--rule)}
.caption{position:absolute;left:96px;bottom:84px;font:500 12px var(--mono);letter-spacing:.12em;text-transform:uppercase;color:rgba(242,238,228,.5)}
""", f"""
<div class="wmk">{mark()}</div>
<div class="copy">{lockup()}
  <h2>On site.<br>In your language.<br><em>On the sheet.</em></h2>
  <p>Ask from the tablet in the site office or the phone in your pocket. Every answer points back to the drawing, so anyone can check it in seconds.</p>
  <div class="langs"><span>EN</span><span class="on">FI</span><span>SV</span><span>DE</span><span>FR</span><span>IT</span><span>ES</span><span>SQ</span><span>ET</span></div>
</div>
<div class="tab"><div class="scr">{app_ui("Kuinka monta saunaa tässä kerroksessa on?", "Suomi", "1", "Sauna", "Taso 2 · 4.2 m²", [("SAUNA · Taso 2", "4.2 m²")], hl="sauna", reg=False, lang="FI")}</div></div>
<div class="phone"><div class="scr">
  <div class="top">{lockup()}<span class="lbl">A-102</span></div>
  <div class="mini">{plan("doors")}</div>
  <article class="card"><div class="q"><small class="lbl">Question · English</small>How many doors are on this floor?</div>
    <div class="a"><div class="num">7</div><div class="unit">Doors<small>Level 2</small></div><div class="stamp">CHECKED<small>MATCHES</small></div></div>
    <div class="meta"><div><span class="lbl">Engine</span><b>Geometry</b></div><div><span class="lbl">Confidence</span><b style="color:var(--green)">High</b></div><div><span class="lbl">Time</span><b>0.21 s</b></div><div><span class="lbl">LLM cost</span><b>$0.00</b></div></div>
  </article>
</div></div>

""")

# ---------------------------------------------------------------- 3. social / OG card
page("oqvar-social", 1200, 630, """
body{background:var(--ink);color:var(--paper);display:grid;grid-template-columns:600px 1fr}
.l{padding:64px 0 0 72px;position:relative}
.l .lockup{font-size:44px}
h1{font:800 60px/1.02 var(--brand);letter-spacing:-.03em;margin-top:70px}
h1 em{font-style:normal;color:var(--hl)}
.l p{position:absolute;left:72px;bottom:58px;font:500 13px var(--mono);letter-spacing:.12em;text-transform:uppercase;color:rgba(242,238,228,.55)}
.r{padding:56px 56px 56px 0;display:flex;align-items:center}
.r .sh{background:#fdfcf8;border:2px solid var(--paper);box-shadow:10px 10px 0 var(--red);transform:rotate(2deg)}
""", f"""
<div class="l">{lockup()}<h1>Ask the drawing.<br><em>Check the answer.</em></h1><p>Blueprint Q&amp;A you can verify</p></div>
<div class="r"><div class="sh">{plan("doors")}</div></div>
""")

# ---------------------------------------------------------------- 4. app icon (1024)
page("oqvar-app-icon", 1024, 1024, """
body{background:transparent;display:grid;place-items:center}
.ic{width:1024px;height:1024px;border-radius:228px;background:var(--ink);display:grid;place-items:center;color:var(--paper)}
.ic .mark{width:560px;height:auto;transform:translate(-8px,-6px)}
""", f'<div class="ic">{mark(eye="#ffe24a")}</div>')

page("oqvar-app-icon-light", 1024, 1024, """
body{background:transparent;display:grid;place-items:center}
.ic{width:1024px;height:1024px;border-radius:228px;background:var(--paper);display:grid;place-items:center;color:var(--ink)}
.ic .mark{width:560px;height:auto;transform:translate(-8px,-6px)}
""", f'<div class="ic">{mark(eye="#cf2a1e")}</div>')

# ---------------------------------------------------------------- 5. brand sheet
page("oqvar-brand-sheet", 1920, 1340, """
body{padding:72px 96px;background:var(--paper)}
.hd{display:flex;justify-content:space-between;align-items:flex-end;border-bottom:2px solid var(--ink);padding-bottom:22px}
.hd h1{display:flex;align-items:center;gap:18px}.hd h1 .wm{height:34px;width:auto}.hd h1 .kit{font:600 30px/1 var(--brand);color:var(--ink-3);letter-spacing:-.01em;text-transform:none}.hd span{font:500 12px var(--mono);letter-spacing:.12em;text-transform:uppercase;color:var(--ink-3)}
.g{display:grid;grid-template-columns:repeat(4,1fr);gap:24px;margin-top:32px}
.t{height:300px;border:1.5px solid var(--ink);display:grid;place-items:center;position:relative}
.t .lbl{position:absolute;left:14px;top:14px}
.t.wide{grid-column:span 2}
.t.ink{background:var(--ink);color:var(--paper)}.t.ink .lbl{color:rgba(242,238,228,.55)}
.t.red{background:var(--red);color:var(--paper)}.t.red .lbl{color:rgba(255,255,255,.7)}
.t.yel{background:var(--hl)}
.t .lockup{font-size:84px}
.t .mark.big{height:170px;width:auto}
.stack{display:flex;flex-direction:column;align-items:center;gap:22px}.stack .mark{height:120px;width:auto}.stack .wm{height:40px;width:auto}
.sw{display:grid;grid-template-columns:repeat(6,1fr);gap:0;margin-top:24px;border:1.5px solid var(--ink)}
.sw div{height:150px;padding:14px;display:flex;flex-direction:column;justify-content:flex-end;font:500 12px/1.5 var(--mono);border-left:1.5px solid var(--ink)}
.sw div:first-child{border-left:0}.sw b{font:700 15px var(--brand)}
.row2{display:grid;grid-template-columns:1.1fr 1fr 1fr;gap:24px;margin-top:24px}
.type{border:1.5px solid var(--ink);padding:22px;height:300px;background:var(--card)}
.type .A{font:800 120px/1 var(--brand);letter-spacing:-.04em;margin:18px 0 6px}
.type p{font:400 14px/1.5 var(--sans);color:var(--ink-2)}
.type .m{font:500 14px var(--mono);margin-top:10px}
.bc{height:300px;display:grid;place-items:center;background:var(--desk);border:1.5px solid var(--ink);position:relative}
.bc .lbl{position:absolute;left:14px;top:14px}
.card2{width:300px;height:172px;box-shadow:0 18px 30px rgba(21,23,27,.25);position:absolute}
.front{background:var(--ink);color:var(--paper);display:grid;place-items:center;transform:translate(-80px,-28px) rotate(-6deg)}
.front .lockup{font-size:40px}
.back{background:var(--card);transform:translate(80px,50px) rotate(4deg);padding:24px;display:flex;flex-direction:column;justify-content:space-between}
.back .lockup{font-size:20px}
.back b{font:700 17px var(--brand)}.back small{display:block;font:400 11px/1.6 var(--mono);color:var(--ink-3)}
.back i{position:absolute;right:24px;bottom:24px;width:46px;height:8px;background:var(--red)}
.st{height:300px;border:1.5px solid var(--ink);background:#c9c3b3;display:grid;place-items:center;position:relative;overflow:hidden}
.st .lbl{position:absolute;left:14px;top:14px}
.hat{width:300px;height:170px;background:#f4f1ea;border-radius:150px 150px 22px 22px;position:relative;box-shadow:inset -30px -20px 40px rgba(0,0,0,.12),0 20px 30px rgba(0,0,0,.25);display:grid;place-items:center;margin-top:40px}
.hat:after{content:"";position:absolute;left:-30px;right:-30px;bottom:-14px;height:26px;background:#ebe7de;border-radius:14px;box-shadow:0 10px 18px rgba(0,0,0,.2)}
.hat .stk{width:86px;height:86px;border-radius:50%;background:var(--ink);display:grid;place-items:center;color:var(--paper);transform:translateY(-6px)}
.hat .stk .mark{width:46px;height:auto}
""", f"""
<div class="hd"><h1>{wordmark()} <span class="kit">Brand kit</span></h1><span>v1 · November 2026</span></div>
<div class="g">
  <div class="t wide"><span class="lbl">Primary lockup</span>{lockup()}</div>
  <div class="t ink"><span class="lbl">Mark · reversed</span>{mark("mark big", eye="#ffe24a")}</div>
  <div class="t"><span class="lbl">Stacked</span><div class="stack">{mark()}{wordmark()}</div></div>
  <div class="t ink wide"><span class="lbl">Lockup · on ink</span>{lockup()}</div>
  <div class="t red"><span class="lbl">Mark · on signal red</span>{mark("mark big")}</div>
  <div class="t yel"><span class="lbl">Mark · on highlighter</span>{mark("mark big")}</div>
</div>
<div class="sw">
  <div style="background:#15171b;color:#f2eee4"><b>Ink</b>#15171B</div>
  <div style="background:#f2eee4"><b>Paper</b>#F2EEE4</div>
  <div style="background:#e7e2d5"><b>Drafting mat</b>#E7E2D5</div>
  <div style="background:#cf2a1e;color:#fff"><b>Checker red</b>#CF2A1E</div>
  <div style="background:#ffe24a"><b>Highlighter</b>#FFE24A</div>
  <div style="background:#156f45;color:#fff"><b>Stamp green</b>#156F45</div>
</div>
<div class="row2">
  <div class="type"><span class="lbl">Type</span><div class="A">Aa</div><p><b>Unbounded</b> — wordmark only. <b>Sora</b> — headlines. <b>IBM Plex Sans</b> — interface and body.</p><div class="m">IBM Plex Mono — labels, numbers, sheet refs</div></div>
  <div class="bc"><span class="lbl">Business card</span>
    <div class="card2 front">{lockup()}</div>
    <div class="card2 back"><span class="lockup" style="font-size:20px">{mark()}{wordmark()}</span><div><b>Your Name</b><small>Site engineering · Helsinki<br>name@oqvar.com</small></div><i></i></div></div>
  <div class="st"><span class="lbl">Site sticker</span><div class="hat"><div class="stk">{mark(eye="#ffe24a")}</div></div></div>
</div>
""")
print("ok", sorted(os.listdir(OUT)))

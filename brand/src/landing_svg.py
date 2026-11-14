"""Emit the inline SVGs for web/landing.html: the logo sprite, three plan variants and an isometric BIM model.

usage: python landing_svg.py <dir with mark_body.d, mark_eye.d, wordmark.d, wordmark.meta> <out dir>
"""
import math
import sys

T, OUT = sys.argv[1], sys.argv[2]
rd = lambda f: open(f"{T}/{f}").read().strip()
BODY, EYE, WM = rd("mark_body.d"), rd("mark_eye.d"), rd("wordmark.d")
W, CAP = map(float, rd("wordmark.meta").split())

sprite = (f'<svg width="0" height="0" style="position:absolute" aria-hidden="true">'
          f'<symbol id="oq-mark" viewBox="467 465 1114 1117"><path fill="currentColor" fill-rule="evenodd" d="{BODY}"/>'
          f'<path fill="var(--eye, currentColor)" d="{EYE}"/></symbol>'
          f'<symbol id="oq-wm" viewBox="0 0 {W:.0f} {CAP:.0f}" overflow="visible"><path fill="currentColor" d="{WM}"/></symbol></svg>')


def pin(cx, cy, n, r=11):
    return f'<g class="pin"><circle cx="{cx}" cy="{cy}" r="{r}"/><text x="{cx}" y="{cy + 4}">{n}</text></g>'


def door(x, y, s, kind):
    if kind[0] == "v":
        sx = 1 if kind[1] == "+" else -1
        geo = (f'<rect class="gap" x="{x-6}" y="{y}" width="12" height="{s}"/><path class="leaf" d="M{x} {y}H{x+sx*s}"/>'
               f'<path class="swing" d="M{x+sx*s} {y}A{s} {s} 0 0 {1 if sx > 0 else 0} {x} {y+s}"/>')
        return geo, (min(x, x + sx * s) - 4, y - 4, s + 8, s + 8), (x + sx * (s + 14), y + s / 2)
    sy = 1 if kind[1] == "+" else -1
    geo = (f'<rect class="gap" x="{x}" y="{y-6}" width="{s}" height="12"/><path class="leaf" d="M{x} {y}V{y+sy*s}"/>'
           f'<path class="swing" d="M{x} {y+sy*s}A{s} {s} 0 0 {0 if sy > 0 else 1} {x+s} {y}"/>')
    return geo, (x - 4, min(y, y + sy * s) - 4, s + 8, s + 8), (x + s / 2, y + sy * (s + 14))


DOORS = [(420, 120, 46, "v-"), (520, 240, 44, "h+"), (620, 290, 40, "v+"), (620, 430, 40, "v+"), (300, 320, 44, "h-"), (240, 400, 44, "v-"), (330, 520, 54, "h-")]
WINDOWS = [(120, 36, 140, 8), (520, 36, 120, 8), (756, 120, 8, 80), (36, 120, 8, 140), (36, 380, 8, 90), (440, 516, 120, 8)]
ROOMS = [(230, 180, "LIVING", "32.4 m²"), (590, 150, "KITCHEN", "18.6 m²"), (520, 400, "BEDROOM 1", "14.8 m²"), (708, 310, "BATH", "6.1 m²"),
         (714, 470, "SAUNA", "4.2 m²"), (140, 420, "BEDROOM 2", "11.9 m²"), (330, 430, "HALL", "9.7 m²")]


def plan(prefix, layer, label, scan=False):
    """layer: 'doors' (numbered door marks), 'kitchen' (room polygon + area) or 'sauna' (room on a scanned sheet)."""
    s = [f'<svg class="plan{" plan-scan" if scan else ""}" viewBox="0 0 800 560" role="img" aria-label="{label}">',
         f'<defs><pattern id="{prefix}-pg" width="20" height="20" patternUnits="userSpaceOnUse"><path d="M20 0H0V20" fill="none" stroke="rgba(31,79,168,.08)"/></pattern>']
    if scan:
        s.append(f'<filter id="{prefix}-grain" x="0" y="0" width="100%" height="100%"><feTurbulence type="fractalNoise" baseFrequency=".9" numOctaves="2" seed="7"/>'
                 '<feColorMatrix values="0 0 0 0 .3  0 0 0 0 .28  0 0 0 0 .24  0 0 0 .22 0"/><feComposite in2="SourceGraphic" operator="in"/>'
                 '<feBlend in="SourceGraphic" mode="multiply"/></filter>')
    s.append('</defs>')
    s.append(f'<rect width="800" height="560" class="paper"/>' + ('' if scan else f'<rect width="800" height="560" fill="url(#{prefix}-pg)"/>'))
    # markups that sit under the linework
    under = []
    if layer == "kitchen":
        under.append('<g class="mk" style="--i:0"><rect class="hl room" x="426" y="46" width="328" height="188"/></g>')
    if layer == "sauna":
        under.append('<g class="mk" style="--i:0"><rect class="hl room" x="625" y="385" width="130" height="130"/></g>')
    s += under
    g = [f'<g{f" filter=\"url(#{prefix}-grain)\"" if scan else ""}>',
         '<g class="fx"><rect x="440" y="52" width="300" height="34"/><circle cx="480" cy="69" r="10"/><circle cx="510" cy="69" r="10"/>'
         '<rect x="560" y="56" width="44" height="26" rx="3"/><rect x="704" y="250" width="44" height="80" rx="18"/>'
         '<rect x="630" y="250" width="40" height="30" rx="10"/><path d="M632 396H752M632 420H752M632 444H752"/>'
         '<rect x="440" y="440" width="120" height="70"/><rect x="60" y="440" width="110" height="70"/>'
         '<rect x="70" y="70" width="150" height="70" rx="6"/><rect x="250" y="210" width="110" height="80" rx="40"/></g>',
         '<rect class="wall-o" x="40" y="40" width="720" height="480"/>',
         '<path class="wall-i" d="M420 40V520M420 240H760M620 240V520M620 380H760M40 320H420M240 320V520"/>']
    marks = []
    for i, d in enumerate(DOORS):
        geo, hl, p = door(*d)
        g.append(geo)
        if layer == "doors":
            x, y, w, h = hl
            marks.append(f'<g class="mk" style="--i:{i}"><rect class="hl" x="{x}" y="{y}" width="{w}" height="{h}"/>{pin(*p, i + 1)}</g>')
    for x, y, w, h in WINDOWS:
        g.append(f'<rect class="win" x="{x}" y="{y}" width="{w}" height="{h}"/>')
    for x, y, a, b in ROOMS:
        hot = (layer == "kitchen" and a == "KITCHEN") or (layer == "sauna" and a == "SAUNA")
        g.append(f'<text class="rl{" hot" if hot else ""}" x="{x}" y="{y}">{a}</text><text class="ra" x="{x}" y="{y + 15}">{b}</text>')
    g.append('</g>')
    s += g
    if layer == "kitchen":  # the room polygon the area was computed from: outline, vertices, area tag
        pts = [(426, 46), (754, 46), (754, 234), (426, 234)]
        s.append('<g class="mk" style="--i:1"><path class="poly" d="M' + "L".join(f"{x} {y}" for x, y in pts) + 'Z"/>'
                 + "".join(f'<circle class="vtx" cx="{x}" cy="{y}" r="5"/>' for x, y in pts) + '</g>')
        s.append('<g class="mk" style="--i:3"><rect class="tag" x="530" y="168" width="120" height="34" rx="8"/>'
                 '<text class="tag-t" x="590" y="190">18.6 m²</text></g>')
    if layer == "sauna":
        s.append(f'<g class="mk" style="--i:1">{pin(740, 400, 1)}</g>')
    s += marks
    s.append('<g class="dim"><path d="M40 22H760M40 16V28M420 16V28M760 16V28"/><text x="230" y="17">7 400</text><text x="590" y="17">6 800</text></g>')
    s.append('</svg>')
    return "".join(s)


def bim(prefix):
    """An isometric three-storey block, every window marked and numbered."""
    k, c, sn = 17, math.cos(math.radians(30)), math.sin(math.radians(30))
    ox, oy = 200, 64
    iso = lambda x, y, z: (ox + (x - y) * c * k, oy + (x + y) * sn * k - z * k)
    P = lambda pts: " ".join(f"{a:.1f},{b:.1f}" for a, b in (iso(*p) for p in pts))
    Wd, D, St, Hs = 12, 8, 3, 3.1
    H = St * Hs
    s = [f'<svg class="bim" viewBox="0 0 400 320" role="img" aria-label="Isometric view of a three-storey building model with every window marked">',
         f'<ellipse cx="{iso(Wd/2, D/2, 0)[0]:.0f}" cy="{iso(Wd, D, 0)[1] - 8:.0f}" rx="170" ry="30" class="shadow"/>',
         f'<polygon class="face-l" points="{P([(0, D, 0), (Wd, D, 0), (Wd, D, H), (0, D, H)])}"/>',
         f'<polygon class="face-r" points="{P([(Wd, 0, 0), (Wd, D, 0), (Wd, D, H), (Wd, 0, H)])}"/>',
         f'<polygon class="face-t" points="{P([(0, 0, H), (Wd, 0, H), (Wd, D, H), (0, D, H)])}"/>',
         f'<polygon class="parapet" points="{P([(.4, .4, H), (Wd - .4, .4, H), (Wd - .4, D - .4, H), (.4, D - .4, H)])}"/>']
    for z in [Hs * i for i in range(1, St)]:  # slab lines
        s.append(f'<polyline class="slab" points="{P([(0, D, z), (Wd, D, z), (Wd, 0, z)])}"/>')
    windows, n = [], 0
    for st in range(St):
        z0 = st * Hs + 1.0
        xs = [1.6, 8.6] if st == 0 else [1.6, 5.1, 8.6]
        for x0 in xs:  # long face (y = D)
            windows.append([(x0, D, z0), (x0 + 1.8, D, z0), (x0 + 1.8, D, z0 + 1.5), (x0, D, z0 + 1.5)])
        if st > 0:  # short face (x = Wd)
            windows.append([(Wd, 3.1, z0), (Wd, 4.9, z0), (Wd, 4.9, z0 + 1.5), (Wd, 3.1, z0 + 1.5)])
    s.append(f'<polygon class="door" points="{P([(5.3, D, 0), (6.9, D, 0), (6.9, D, 2.2), (5.3, D, 2.2)])}"/>')
    marks = []
    for i, w in enumerate(windows):
        s.append(f'<polygon class="glass" points="{P(w)}"/>')
        cx = sum(iso(*p)[0] for p in w) / 4
        cy = sum(iso(*p)[1] for p in w) / 4
        marks.append(f'<g class="mk" style="--i:{i}"><polygon class="hl" points="{P(w)}"/>{pin(round(cx, 1), round(cy - 15, 1), i + 1, r=8)}</g>')
        n += 1
    s += marks
    s.append('</svg>')
    return "".join(s), n


open(f"{OUT}/sprite.svg", "w").write(sprite)
open(f"{OUT}/plan-doors.svg", "w").write(plan("pd", "doors", "Sample floor plan with all seven doors outlined and numbered"))
open(f"{OUT}/plan-cad.svg", "w").write(plan("pc", "kitchen", "Sample floor plan with the kitchen outline traced and its area, 18.6 square metres, labelled"))
open(f"{OUT}/plan-scan.svg", "w").write(plan("ps", "sauna", "Scanned floor plan with the sauna highlighted", scan=True))
svg, n = bim("bm")
open(f"{OUT}/bim.svg", "w").write(svg)
print("windows in BIM sample:", n)

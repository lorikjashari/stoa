"""Emit the logo sprite and the layered demo floor plan used by web/landing.html."""
import sys
T, OUT = sys.argv[1], sys.argv[2]
rd = lambda f: open(f"{T}/{f}").read().strip()
BODY, EYE, WM = rd("mark_body.d"), rd("mark_eye.d"), rd("wordmark.d")
W, CAP = map(float, rd("wordmark.meta").split())

sprite = (f'<svg width="0" height="0" style="position:absolute" aria-hidden="true">'
          f'<symbol id="oq-mark" viewBox="467 465 1114 1117"><path fill="currentColor" fill-rule="evenodd" d="{BODY}"/>'
          f'<path fill="var(--eye, currentColor)" d="{EYE}"/></symbol>'
          f'<symbol id="oq-wm" viewBox="0 0 {W:.0f} {CAP:.0f}" overflow="visible"><path fill="currentColor" d="{WM}"/></symbol></svg>')

def door(x, y, s, kind):
    if kind[0] == "v":
        sx = 1 if kind[1] == "+" else -1
        geo = (f'<rect class="gap" x="{x-6}" y="{y}" width="12" height="{s}"/><path class="leaf" d="M{x} {y}H{x+sx*s}"/>'
               f'<path class="swing" d="M{x+sx*s} {y}A{s} {s} 0 0 {1 if sx>0 else 0} {x} {y+s}"/>')
        hl = (min(x, x+sx*s)-4, y-4, s+8, s+8); pin = (x+sx*(s+14), y+s/2)
    else:
        sy = 1 if kind[1] == "+" else -1
        geo = (f'<rect class="gap" x="{x}" y="{y-6}" width="{s}" height="12"/><path class="leaf" d="M{x} {y}V{y+sy*s}"/>'
               f'<path class="swing" d="M{x} {y+sy*s}A{s} {s} 0 0 {0 if sy>0 else 1} {x+s} {y}"/>')
        hl = (x-4, min(y, y+sy*s)-4, s+8, s+8); pin = (x+s/2, y+sy*(s+14))
    return geo, hl, pin

def mark(i, hl, pin):
    x, y, w, h = hl; cx, cy = pin
    return (f'<g class="mk" style="--i:{i}"><rect class="hl" x="{x}" y="{y}" width="{w}" height="{h}"/>'
            f'<g class="pin"><circle cx="{cx}" cy="{cy}" r="11"/><text x="{cx}" y="{cy+4}">{i+1}</text></g></g>')

doors = [(420,120,46,"v-"),(520,240,44,"h+"),(620,290,40,"v+"),(620,430,40,"v+"),(300,320,44,"h-"),(240,400,44,"v-"),(330,520,54,"h-")]
windows = [(120,36,140,8),(520,36,120,8),(756,120,8,80),(36,120,8,140),(36,380,8,90),(440,516,120,8)]
win_pins = [(190,62),(580,62),(734,160),(62,190),(62,425),(500,494)]
rooms = [(230,180,"LIVING","32.4 m²",""),(590,140,"KITCHEN","18.6 m²",""),(520,400,"BEDROOM 1","14.8 m²",""),
         (708,310,"BATH","6.1 m²",""),(714,470,"SAUNA","4.2 m²"," sauna"),(140,420,"BEDROOM 2","11.9 m²",""),(330,430,"HALL","9.7 m²","")]

s = ['<svg class="plan" viewBox="0 0 800 560" role="img" aria-labelledby="plan-title">',
     '<title id="plan-title">Illustrative floor plan, level 2: living room, kitchen, two bedrooms, hall, bath and sauna</title>',
     '<defs><pattern id="pg" width="20" height="20" patternUnits="userSpaceOnUse"><path d="M20 0H0V20" fill="none" stroke="rgba(31,79,168,.08)"/></pattern></defs>',
     '<rect width="800" height="560" fill="url(#pg)"/>',
     '<g data-set="sauna"><g class="mk" style="--i:0"><rect class="hl room" x="625" y="385" width="130" height="130"/>'
     '<g class="pin"><circle cx="740" cy="400" r="11"/><text x="740" y="404">1</text></g></g></g>',
     '<g class="fx"><rect x="440" y="52" width="300" height="34"/><circle cx="480" cy="69" r="10"/><circle cx="510" cy="69" r="10"/>'
     '<rect x="560" y="56" width="44" height="26" rx="3"/><rect x="704" y="250" width="44" height="80" rx="18"/>'
     '<rect x="630" y="250" width="40" height="30" rx="10"/><path d="M632 396H752M632 420H752M632 444H752"/>'
     '<rect x="440" y="440" width="120" height="70"/><rect x="60" y="440" width="110" height="70"/>'
     '<rect x="70" y="70" width="150" height="70" rx="6"/><rect x="250" y="210" width="110" height="80" rx="40"/></g>',
     '<rect class="wall-o" x="40" y="40" width="720" height="480"/>',
     '<path class="wall-i" d="M420 40V520M420 240H760M620 240V520M620 380H760M40 320H420M240 320V520"/>']
dmarks = []
for i, d in enumerate(doors):
    geo, hl, pin = door(*d); s.append(geo); dmarks.append(mark(i, hl, pin))
for x, y, w, h in windows:
    s.append(f'<rect class="win" x="{x}" y="{y}" width="{w}" height="{h}"/>')
wmarks = [mark(i, (x-4, y-4, w+8, h+8), win_pins[i]) for i, (x, y, w, h) in enumerate(windows)]
for x, y, a, b, extra in rooms:
    s.append(f'<text class="rl{extra}" x="{x}" y="{y}">{a}</text><text class="ra" x="{x}" y="{y+15}">{b}</text>')
s.append('<g data-set="doors">' + "".join(dmarks) + '</g>')
s.append('<g data-set="windows">' + "".join(wmarks) + '</g>')
s.append('<g class="dim"><path d="M40 22H760M40 16V28M420 16V28M760 16V28"/><text x="230" y="17">7 400</text><text x="590" y="17">6 800</text></g>')
s.append('</svg>')
open(f"{OUT}/sprite.svg", "w").write(sprite)
open(f"{OUT}/plan.svg", "w").write("".join(s))
print(len(sprite), len("".join(s)))

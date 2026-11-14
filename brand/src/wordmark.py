import re, sys
from fontTools.ttLib import TTFont
from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.pens.transformPen import TransformPen
font = TTFont(sys.argv[1]); out = sys.argv[2]
gs = font.getGlyphSet(); cmap = font.getBestCmap(); upm = font["head"].unitsPerEm
cap = font["OS/2"].sCapHeight
track = 0.03*upm
x = 0; pen = SVGPathPen(gs); kern=0
for ch in "OQVAR":
    g = cmap[ord(ch)]
    gs[g].draw(TransformPen(pen, (1,0,0,-1,x,cap)))
    x += gs[g].width + track
w = x - track
d = pen.getCommands()
open(f"{out}/wordmark.d","w").write(d)
open(f"{out}/wordmark.meta","w").write(f"{w} {cap}")
mark = open(f"{out}/oqvar-mark.svg").read()
md = re.search(r' d="([^"]*)', mark).group(1)
open(f"{out}/mark.d","w").write(md)
ink="#15171b"
open(f"{out}/oqvar-wordmark.svg","w").write(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="-20 -20 {w+40:.0f} {cap+200}"><path fill="{ink}" d="{d}"/></svg>\n')
# lockup: mark height = 1.5*cap, gap = 0.45*cap
mh = cap*1.5; s = mh/1117; gap = cap*0.5
mw = 1114*s; W = mw+gap+w; H = mh
ty = (mh-cap)/2 + cap*0.06
lock = f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W:.0f} {H:.0f}"><g transform="scale({s:.5f}) translate(-467 -465)"><path fill="{ink}" fill-rule="evenodd" d="{md}"/></g><g transform="translate({mw+gap:.1f} {ty:.1f})"><path fill="{ink}" d="{d}"/></g></svg>\n'
open(f"{out}/oqvar-lockup.svg","w").write(lock)
print(w, cap, W, H)

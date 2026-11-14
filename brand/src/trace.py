import sys, numpy as np, potrace
from PIL import Image
src, out = sys.argv[1], sys.argv[2]
im = Image.open(src).convert("L")
a = np.array(im) >= 128
bm = potrace.Bitmap(a)
plist = bm.trace(turdsize=50, alphamax=1.0, opticurve=True, opttolerance=0.2)
ys, xs = np.where(a); x0,x1,y0,y1 = xs.min(), xs.max(), ys.min(), ys.max()
print("bbox", x0,x1,y0,y1)
parts=[]
for c in plist:
    s = c.start_point; d=[f"M{s.x:.1f} {s.y:.1f}"]
    for seg in c.segments:
        if seg.is_corner: d.append(f"L{seg.c.x:.1f} {seg.c.y:.1f}L{seg.end_point.x:.1f} {seg.end_point.y:.1f}")
        else: d.append(f"C{seg.c1.x:.1f} {seg.c1.y:.1f} {seg.c2.x:.1f} {seg.c2.y:.1f} {seg.end_point.x:.1f} {seg.end_point.y:.1f}")
    d.append("Z"); parts.append("".join(d))
pad=0
vb=f"{x0} {y0} {x1-x0+1} {y1-y0+1}"
path=" ".join(parts)
open(out,"w").write(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{vb}"><path fill="#15171b" fill-rule="evenodd" d="{path}"/></svg>\n')
print(len(plist),"paths", vb)

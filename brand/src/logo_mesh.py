"""Turn the traced OQVAR mark into a light 2D mesh for the WebGL logo: outlines plus cap triangles, per material.

usage: python logo_mesh.py <mark_body.d> <mark_eye.d> <out.json>
The .d files hold SVG path data (M/C/L/Z, absolute) from the trace. Coordinates are mapped to the same square as
logo_sdf.py (mark centre at 0, the 15%-padded canvas spans -1..1), so the mesh and the distance texture line up.
The browser extrudes the outlines into front/back caps and side walls.
"""
import json
import re
import sys

import mapbox_earcut as earcut
import numpy as np

CX, CY, E = 467 + 1114 / 2, 465 + 1117 / 2, 1117 * 1.3   # keep in sync with the SDF canvas
STEPS = 10                                                  # segments per cubic curve


def rings(d: str) -> list[np.ndarray]:
    out = []
    for sub in [s for s in d.split("Z") if s.strip()]:
        toks = re.findall(r"[MCL]|-?\d+(?:\.\d+)?", sub)
        pts, i, cur = [], 0, None
        while i < len(toks):
            t = toks[i]
            if t in "ML":
                cur = (float(toks[i + 1]), float(toks[i + 2])); pts.append(cur); i += 3
            elif t == "C":
                c1, c2, e = [(float(toks[i + k]), float(toks[i + k + 1])) for k in (1, 3, 5)]
                for s in np.linspace(0, 1, STEPS + 1)[1:]:
                    u = 1 - s
                    pts.append(tuple(u**3 * np.array(cur) + 3 * u * u * s * np.array(c1) + 3 * u * s * s * np.array(c2) + s**3 * np.array(e)))
                cur = e; i += 7
            else:
                i += 1
        p = np.array(pts)
        keep = np.r_[True, np.linalg.norm(np.diff(p, axis=0), axis=1) > 0.5]  # drop repeated points
        p = p[keep]
        if np.linalg.norm(p[0] - p[-1]) < 0.5:
            p = p[:-1]
        out.append(np.c_[(p[:, 0] - CX) / (E / 2), -(p[:, 1] - CY) / (E / 2)])
    return out


def part(d: str) -> dict:
    rs = rings(d)
    pts = np.concatenate(rs)
    ends = np.cumsum([len(r) for r in rs]).astype(np.uint32)
    tris = earcut.triangulate_float32(pts.astype(np.float32), ends)
    return {"pts": [round(float(v), 5) for v in pts.ravel()], "rings": [int(e) for e in ends], "tris": [int(t) for t in tris]}


body, eye = open(sys.argv[1]).read().strip(), open(sys.argv[2]).read().strip()
mesh = {"body": part(body), "eye": part(eye)}
json.dump(mesh, open(sys.argv[3], "w"), separators=(",", ":"))
for k, v in mesh.items():
    print(k, "points:", len(v["pts"]) // 2, "rings:", v["rings"], "cap triangles:", len(v["tris"]) // 3)

"""Turn a DXF that already holds a 3D mesh (3DFACE entities, e.g. a Blender export) into a viewer scene.

Nothing is inferred: the faces are the model. The scene JSON carries the metadata; the triangles go into a
compact binary next to it (<name>.mesh.bin: float32 xyz positions, then uint32 triangle indices), because a
few hundred thousand faces as JSON would be tens of megabytes.

    python blueprint3d/mesh_to_scene.py "<model>.dxf" [-o out_dir]

Coordinates are written in the viewer's frame: plan (x, y), height z  ->  (x, z, -y).
"""
import argparse, json, os, re, sys

import ezdxf
import numpy as np


INSUNITS = {1: (0.0254, "inches"), 2: (0.3048, "feet"), 4: (0.001, "millimetres"), 5: (0.01, "centimetres"), 6: (1.0, "metres")}


def read_faces(path):
    """(n, 4, 3) array of 3DFACE corners in drawing units, and $INSUNITS. Triangles repeat their last corner."""
    doc = ezdxf.readfile(path)
    quads = [[tuple(e.dxf.get(k)) for k in ("vtx0", "vtx1", "vtx2", "vtx3")] for e in doc.modelspace() if e.dxftype() == "3DFACE"]
    return np.asarray(quads, dtype=np.float64).reshape(-1, 4, 3), doc.header.get("$INSUNITS")


def unit_scale(insunits, height, extent):
    """(metres per unit, how we know). Without $INSUNITS, pick the first unit that gives a plausible building."""
    if insunits in INSUNITS:
        s, name = INSUNITS[insunits]
        return s, f"{name} ($INSUNITS)"
    for s, name in ((1.0, "metres"), (0.3048, "feet"), (0.0254, "inches"), (0.001, "millimetres")):
        if 2.2 <= height * s <= 20.0 and 3.0 <= extent * s <= 300.0:
            return s, (f"{name} (not stored in the file; assumed because the model is then "
                       f"{extent * s:.1f} m across and {height * s:.1f} m tall)")
    return 1.0, "unknown (not stored in the file)"


def triangulate(quads):
    """Split every face into (0,1,2) + (0,2,3), drop degenerate triangles, merge shared corners."""
    tris = np.concatenate([quads[:, [0, 1, 2]], quads[:, [0, 2, 3]]])
    area2 = np.linalg.norm(np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0]), axis=1)
    tris = tris[area2 > 1e-12]
    pts = np.round(tris.reshape(-1, 3), 6)
    verts, idx = np.unique(pts, axis=0, return_inverse=True)
    return verts, idx.reshape(-1, 3)


def build_scene(path, bin_path):
    quads, insunits = read_faces(path)
    if not len(quads):
        raise ValueError("no 3DFACE entities: this DXF holds no 3D mesh")
    verts, tris = triangulate(quads)
    scale, units = unit_scale(insunits, float(np.ptp(verts[:, 2])), float(np.ptp(verts[:, :2], axis=0).max()))
    verts = (verts - verts.min(axis=0) * [0, 0, 1]) * scale  # metres; ground at z = 0
    world = np.stack([verts[:, 0], verts[:, 2], -verts[:, 1]], axis=1).astype(np.float32)
    lo, hi = world.min(axis=0), world.max(axis=0)
    with open(bin_path, "wb") as f:
        f.write(world.tobytes())
        f.write(tris.astype(np.uint32).tobytes())
    rel = os.path.relpath(path).replace("\\", "/")
    m = re.search(r"data/.*", rel)
    size = (hi - lo).round(2).tolist()
    return {
        "schema": "blueprint3d/scene@1",
        "source": {"file": m.group(0) if m else os.path.basename(path), "format": "mesh", "units": "m",
                   "file_units": units, "scale_to_m": scale},
        "assumptions": {"note": f"The 3D faces stored in the file, shown as they are; no element types. File units: {units}."},
        "mesh": {"file": os.path.basename(bin_path), "vertices": int(len(world)), "triangles": int(len(tris)), "faces": int(len(quads))},
        "size": {"x": size[0], "height": size[1], "y": size[2]},
        "bbox": [float(lo[0]), float(-hi[2]), float(hi[0]), float(-lo[2])],
        "counts": {"rooms": 0, "faces": int(len(quads))},
        "units": [], "rooms": [], "walls": [], "doors": [], "windows": [], "fixtures": [],
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dxf", nargs="+")
    ap.add_argument("-o", "--out", default=os.path.join(os.path.dirname(__file__), "viewer", "scenes"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for p in a.dxf:
        name = re.sub(r"[^A-Za-z0-9_.-]+", "-", os.path.splitext(os.path.basename(p))[0]).strip("-")
        scene = build_scene(p, os.path.join(a.out, name + ".mesh.bin"))
        with open(os.path.join(a.out, name + ".json"), "w", encoding="utf-8") as f:
            json.dump(scene, f, ensure_ascii=False, separators=(",", ":"))
        print(f"{name}: {scene['mesh']['faces']} faces -> {scene['mesh']['triangles']} triangles, size {scene['size']}")


if __name__ == "__main__":
    sys.exit(main())

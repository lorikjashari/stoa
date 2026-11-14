"""Run the pretrained CubiCasa5K multi-task model (Kalervo et al., SCIA 2019) on a raster floor plan.

Model code: third_party/CubiCasa5k (github.com/CubiCasa/CubiCasa5k, CC BY-NC 4.0)
Weights:    third_party/weights/model_best_val_loss_var.pkl (link in that repo's README)

Outputs per pixel: 21 junction heatmaps, 12 room classes, 11 icon classes. The repo's post-processing turns
these into wall polygons, door/window/icon polygons and room polygons, which we convert to our scene.json.
"""
import importlib.util, os, sys, time

import cv2
import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(HERE, "..", "third_party", "CubiCasa5k")
WEIGHTS = os.path.join(HERE, "..", "third_party", "weights", "model_best_val_loss_var.pkl")
sys.path.insert(0, REPO)

ROOM_CLASSES = ["Background", "Outdoor", "Wall", "Kitchen", "Living Room", "Bed Room", "Bath", "Entry", "Railing",
                "Storage", "Garage", "Undefined"]
ICON_CLASSES = ["No Icon", "Window", "Door", "Closet", "Electrical Appliance", "Toilet", "Sink", "Sauna Bench",
                "Fire Place", "Bathtub", "Chimney"]
SPLIT = [21, 12, 11]

_model = None


_mods = {}


class _StatsCompat:
    """2019 code expects scipy.stats.mode(...).mode[0]; modern SciPy returns scalars unless keepdims=True."""

    def __getattr__(self, name):
        import scipy.stats
        return getattr(scipy.stats, name)

    @staticmethod
    def mode(a, *args, **kwargs):
        import scipy.stats
        kwargs.setdefault("keepdims", True)
        return scipy.stats.mode(a, *args, **kwargs)


def _load_module(name, rel):
    """Import one file from the repo without running its package __init__ (which pulls in lmdb, svg tools...)."""
    if name not in _mods:
        spec = importlib.util.spec_from_file_location(name, os.path.join(REPO, rel))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        if hasattr(mod, "stats"):
            mod.stats = _StatsCompat()
        _mods[name] = mod
    return _mods[name]


def load_model():
    global _model
    if _model is None:
        from floortrans.models.hg_furukawa_original import hg_furukawa_original
        m = hg_furukawa_original(n_classes=51)
        m.conv4_ = torch.nn.Conv2d(256, 44, bias=True, kernel_size=1)
        m.upsample = torch.nn.ConvTranspose2d(44, 44, kernel_size=4, stride=4)
        ckpt = torch.load(WEIGHTS, map_location="cpu", weights_only=True)  # tensors only, no pickled code
        m.load_state_dict(ckpt["model_state"])
        m.eval()
        _model = m
    return _model


def predict(path, max_side=1024, tta=False):
    """Return (heatmaps, rooms, icons, scale) at the (possibly downscaled) working resolution."""
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"cannot read {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h0, w0 = img.shape[:2]
    scale = min(1.0, max_side / max(h0, w0))
    if scale < 1:
        img = cv2.resize(img, (round(w0 * scale), round(h0 * scale)), interpolation=cv2.INTER_AREA)
    h, w = img.shape[:2]
    x = torch.from_numpy(np.moveaxis(img, -1, 0)).float().unsqueeze(0)
    x = 2 * (x / 255.0) - 1
    model = load_model()
    with torch.no_grad():
        if tta:
            rot = _load_module("cc_aug", "floortrans/loaders/augmentations.py").RotateNTurns()
            preds = []
            for fwd, back in [(0, 0), (1, -1), (2, 2), (-1, 1)]:
                p = model(rot(x, "tensor", fwd))
                p = rot(rot(p, "tensor", back), "points", back)
                preds.append(F.interpolate(p, size=(h, w), mode="bilinear", align_corners=True))
            pred = torch.stack(preds).mean(0)
        else:
            pred = F.interpolate(model(x), size=(h, w), mode="bilinear", align_corners=True)
    pp = _load_module("cc_post", "floortrans/post_prosessing.py")
    heatmaps, rooms, icons = pp.split_prediction(pred, (h, w), SPLIT)
    return heatmaps, rooms, icons, scale


def vectorize(heatmaps, rooms, icons, threshold=0.2):
    """Repo post-processing: wall/icon polygons (+types) and room polygons (+types)."""
    pp = _load_module("cc_post", "floortrans/post_prosessing.py")
    return pp.get_polygons((heatmaps, rooms, icons), threshold, [1, 2])


if __name__ == "__main__":
    t = time.time(); load_model(); print(f"model loaded in {time.time() - t:.1f}s")
    for p in sys.argv[1:]:
        t = time.time()
        hm, rm, ic, s = predict(p)
        t1 = time.time() - t
        polygons, types, room_polygons, room_types = vectorize(hm, rm, ic)
        from collections import Counter
        c = Counter(ICON_CLASSES[ty["class"]] if ty["type"] == "icon" else ty["type"] for ty in types)
        r = Counter(ROOM_CLASSES[ty["class"]] for ty in room_types)
        print(f"{os.path.basename(p)}: net {t1:.1f}s, total {time.time() - t:.1f}s | {dict(c)} | rooms {dict(r)}")

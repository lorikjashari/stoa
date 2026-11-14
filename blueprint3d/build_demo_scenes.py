"""Build the demo scenes for the viewer in one go, skipping whatever isn't available on this machine.

    python blueprint3d/build_demo_scenes.py                    # DXF + IFC (same readers as stoa_qa)
    .venv/Scripts/python -W ignore blueprint3d/build_demo_scenes.py   # + raster plans (needs .venv + model)

Uses the drawings shown in the team demo when present; otherwise falls back to the first files of each source
(e.g. when only the 0.5 GB preview dataset is downloaded). Writes blueprint3d/viewer/scenes/*.json + index.json.
"""
import glob, json, os, sys, traceback

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "junction-dataset", "junction-dataset", "data")
OUT = os.path.join(HERE, "viewer", "scenes")
sys.path.insert(0, HERE)

DEMO = {
    "dxf": ("swiss_dwellings", "*.dxf", ["site1000_bldg2082_floor9385", "site1047_bldg2154_floor9740",
                                         "site1689_bldg2841_floor13117", "site4114_bldg6635_floor26960"]),
    "ifc": ("ifc_models", "*.ifc", ["kit-fzk-haus_ac20-fzk-haus", "bsc-duplex_duplex-a-20110907",
                                    "kit-institute_ac20-institute-var-2", "bimspeed-warmond_ar"]),
    "raster": ("cubicasa5k", "*.png", ["colorful-10052", "high_quality-10004", "high_quality_architectural-1",
                                       "high_quality_architectural-1191"]),
}


def pick(source, pattern, wanted, n=4):
    folder = os.path.join(DATA, source)
    ext = pattern[1:]
    found = [os.path.join(folder, w + ext) for w in wanted if os.path.exists(os.path.join(folder, w + ext))]
    if len(found) < n:  # preview dataset: top up with whatever exists, smallest files first (fast to load)
        extra = sorted(glob.glob(os.path.join(folder, pattern)), key=os.path.getsize)
        if source == "ifc_models":
            extra = [f for f in extra if os.path.getsize(f) > 200_000]  # skip near-empty models
        found += [f for f in extra if f not in found][: n - len(found)]
    return found


def builders():
    out = {}
    import dxf_to_scene, ifc_to_scene
    out["dxf"] = lambda p: dxf_to_scene.build_scene(p, dxf_to_scene.DEFAULTS)
    out["ifc"] = ifc_to_scene.build_scene
    try:
        import raster_to_scene, cubicasa_model  # noqa: F401  (needs numpy, cv2, torch)
        if not os.path.exists(cubicasa_model.WEIGHTS):
            raise FileNotFoundError(cubicasa_model.WEIGHTS)
        out["raster"] = raster_to_scene.build_scene_model
    except Exception as e:
        print(f"[raster] skipped: {type(e).__name__}: {e}. See SETUP.md step 4 (run with .venv python).")
    return out


def main():
    if not os.path.isdir(DATA):
        sys.exit(f"Dataset not found at {os.path.normpath(DATA)}. See SETUP.md step 2.")
    os.makedirs(OUT, exist_ok=True)
    index_path = os.path.join(OUT, "index.json")
    index = json.load(open(index_path, encoding="utf-8")) if os.path.exists(index_path) else []
    made = 0
    for kind, build in builders().items():
        source, pattern, wanted = DEMO[kind]
        files = pick(source, pattern, wanted)
        if not files:
            print(f"[{kind}] no files in data/{source}/")
        for p in files:
            name = os.path.splitext(os.path.basename(p))[0]
            try:
                scene = build(p)
            except Exception:
                print(f"[{kind}] FAILED {name}:\n{traceback.format_exc(limit=2)}")
                continue
            with open(os.path.join(OUT, name + ".json"), "w", encoding="utf-8") as f:
                json.dump(scene, f, ensure_ascii=False, separators=(",", ":"))
            index = [x for x in index if x["name"] != name] + [{"name": name, "source": scene["source"]["file"], "counts": scene["counts"]}]
            c = scene["counts"]
            print(f"[{kind}] {name}: {c.get('rooms', 0)} rooms, {c.get('doors', 0)} doors, {c.get('windows', 0)} windows")
            made += 1
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(sorted(index, key=lambda x: x["name"]), f, ensure_ascii=False, indent=1)
    print(f"\n{made} scenes ready. Start the viewer:  py -m http.server 8765  ->  http://localhost:8765/blueprint3d/viewer/")


if __name__ == "__main__":
    main()

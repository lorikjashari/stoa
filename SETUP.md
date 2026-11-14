# Setup: Blueprint 3D (Stoa · Make Blueprints Talk)

Use this guide to get from a fresh clone to the running **2D ⇄ 3D viewer** that answers questions about construction drawings. You can follow it yourself, or tell your coding agent: *"Follow SETUP.md and stop at the first step that fails."*

| You want | Do steps | Time | Needs |
| --- | --- | --- | --- |
| DXF floor plans + IFC building models in 3D | 1 → 2 → 3 → 5 | ~10 min | Python 3.10+, a browser |
| + raster plans (scans/images) via the CubiCasa5K network | + step 4 | +10 min | ~1.5 GB disk |
| Reproduce the accuracy numbers | + step 6 | 5–15 min | — |

Commands are shown for **Windows** (Git Bash or PowerShell). On **macOS/Linux**, replace `py` with `python3` and `.venv/Scripts/python` with `.venv/bin/python`.

---

## 1. Clone

```bash
git clone https://github.com/lorikjashari/stoa.git
cd stoa
```

Run every command below from the repo root (`stoa/`).

## 2. Get Stoa's dataset (not in git)

Download the dataset zip. The preview (0.5 GB) is linked in the challenge brief; the full set comes from Stoa at the event:

```text
https://fwjydgxpttajiqpfmqtj.supabase.co/storage/v1/object/public/stoa-junction-2026/junction-dataset-preview.zip
```

Unzip it so the folders end up exactly here:

```text
stoa/junction-dataset/junction-dataset/data/swiss_dwellings/...
stoa/junction-dataset/junction-dataset/data/ifc_models/...
stoa/junction-dataset/junction-dataset/data/cubicasa5k/...
stoa/junction-dataset/junction-dataset/questions.jsonl
```

To check:

```bash
py -c "import os; print(os.path.isfile('junction-dataset/junction-dataset/questions.jsonl'))"
```

It must print `True`. If it doesn't, you probably have one `junction-dataset` level too many or too few.

`junction-dataset/` is gitignored. **Never commit it.** It's large and licensed per file (some files are non-commercial or share-alike).

## 3. Build the demo scenes (DXF + IFC)

These use the Python standard library only, so there's nothing to install:

```bash
py -I blueprint3d/build_demo_scenes.py
```

Expected output: `[dxf] …` and `[ifc] …` lines, and `8 scenes ready`. If the venv from step 4 isn't set up yet, you'll also see `[raster] skipped …`, which is fine.

The scenes go to `blueprint3d/viewer/scenes/` (gitignored, so each person builds their own). With only the preview dataset, the script picks whatever files exist.

## 4. Optional: raster plans with the CubiCasa5K network

This turns scanned or image floor plans into 3D, with typed rooms, doors, windows, toilets, sinks and saunas.

```bash
# 4a. Python environment (OpenCV, SciPy, Shapely + CPU PyTorch)
py -m venv .venv
.venv/Scripts/python -m pip install -r blueprint3d/requirements.txt
.venv/Scripts/python -m pip install torch --index-url https://download.pytorch.org/whl/cpu

# 4b. Model code (official repo, 2019)
git clone --depth 1 https://github.com/CubiCasa/CubiCasa5k.git third_party/CubiCasa5k

# 4c. Model weights (208 MB, from the link in the CubiCasa5k README)
#     In Windows PowerShell, type curl.exe instead of curl (plain curl is an alias for Invoke-WebRequest there)
mkdir -p third_party/weights
curl -L -o third_party/weights/model_best_val_loss_var.pkl "https://drive.usercontent.google.com/download?id=1gRB7ez1e4H7a9Y09lLqRuna0luZO5VRK&export=download&confirm=t"

# 4d. Build all demo scenes, raster included
.venv/Scripts/python -W ignore blueprint3d/build_demo_scenes.py
```

Expected: `12 scenes ready`, including four `[raster]` lines.

Check the weights: the file should be about 208 MB. If it's a few KB, Google Drive sent an HTML page instead. Download it in a browser from the CubiCasa5k README link and put it in `third_party/weights/`.

`.venv/` and `third_party/` are gitignored. **Never commit them.** The model and its data are licensed CC BY-NC(-SA) 4.0: non-commercial use, with credit to CubiCasa and Kalervo et al. 2019.

### 4e. Optional: count on your own plans with the trained YOLO detectors

The app answers raster plans from boxes predicted offline for CubiCasa's val/test plans. To count on any other plan,
including one you upload in the web app ("Upload a plan image", or drop the file on the sheet), install the detectors:

```bash
uv sync --extra detect                       # ultralytics + CPU torch (not needed for CI or the base app)
modal token new                              # once, if this machine has no Modal login
uv run modal volume get stoa-cubicasa runs/m13-1280/weights/best.pt third_party/weights/m13-1280.pt
# the second model of the fused run, if you have it (it was trained on a laptop, not on Modal):
#   copy its best.pt to third_party/weights/local-y11s-13c.pt
```

`stoa_qa/cubicasa_live.py` runs whichever of the two models is present (both → fused like the offline run; one →
used alone, and the answer says the count is less certain). Boxes are cached in `eval_out/yolo/live/`; uploads are
stored in `eval_out/uploads/` (both gitignored). `STOA_YOLO_WEIGHTS` points to another weights folder,
`STOA_NO_LIVE_YOLO=1` turns live detection off. The app also starts without the dataset, for uploads only.

## 5. Launch the viewer

```bash
py -m http.server 8765
```

Open **http://localhost:8765/blueprint3d/viewer/** and keep the terminal running.

- **Pick a drawing:** use the top-left dropdown. Entries are labelled `[DXF]`, `[IFC]` or `[RASTER]`.
- **Ask a question:** for example *How many doors?*, *Area of room 2.10?*, *What is the net floor area of the room 'Living Room (A102)' on 'Level 1'?* or *How many toilet bowls are drawn on this floor plan?* The answer is highlighted in both views.
- **Click** any room, wall, door or window in either view to see its ID, source and quantities.
- **3D controls:** drag to orbit, scroll to zoom, right-drag to pan. Buttons: **X-ray** (see-through walls), **Top**, **Reset**. IFC models also get a storey picker and an **Isolate storey** button.
- **Original drawing** (2D tab): shows the source image so you can check the result against it.
- **Export .glb:** downloads the 3D model. Open it in Blender with *File → Import → glTF 2.0*.

If you use the Claude desktop app, `.claude/launch.json` already has a `blueprint3d` server entry that does the same.

**Add any other drawing:**

```bash
py -I blueprint3d/dxf_to_scene.py  junction-dataset/junction-dataset/data/swiss_dwellings/<file>.dxf
py -I blueprint3d/ifc_to_scene.py  junction-dataset/junction-dataset/data/ifc_models/<file>.ifc
.venv/Scripts/python -W ignore blueprint3d/raster_to_scene.py junction-dataset/junction-dataset/data/cubicasa5k/<file>.png
```

Then refresh the browser. The raster converter takes `--engine opencv` for the hand-tuned baseline.

## 6. Optional: reproduce the accuracy numbers

```bash
py -I blueprint3d/eval_swiss.py      # DXF: ~15.5k questions, ~1 min   -> 99.2%
py -I blueprint3d/eval_ifc.py        # IFC: ~1.6k questions, ~15 s     -> 98.9%

# raster (needs step 4): fetch CubiCasa's train/test split lists (~4 MB read from the Zenodo zip),
# then score the network ONLY on the 400 plans it never trained on
py -I blueprint3d/fetch_cubicasa_split.py
.venv/Scripts/python -W ignore blueprint3d/eval_cubicasa_model.py --split test
```

Model predictions are cached in `third_party/cubicasa_preds/`, so re-runs are fast.

---

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| `Dataset not found at …` | Step 2: the unzip location is wrong. `questions.jsonl` must be at `junction-dataset/junction-dataset/questions.jsonl`. |
| Viewer page is empty or the dropdown is empty | No scenes yet: run step 3. Also check that the server runs from the **repo root**, not `blueprint3d/`. |
| "Original drawing" tab is blank | The server isn't running from the repo root, or that drawing isn't in your (preview) dataset. |
| IFC model never finishes loading | It needs internet once: web-ifc and three.js load from the jsdelivr CDN. Very large IFCs (40–85 MB) are slow. Try the smaller demo ones first. |
| `[raster] skipped: No module named 'torch'` | You ran with `py`. Use `.venv/Scripts/python` after step 4a. |
| `[raster] skipped: FileNotFoundError … weights` | Step 4c: the weights file is missing or in the wrong folder. |
| `UnicodeEncodeError` printing results on Windows | Prefix the command with `PYTHONIOENCODING=utf-8` (Git Bash), or run `$env:PYTHONIOENCODING="utf-8"` first (PowerShell). |
| `py` not found (macOS/Linux) | Use `python3`, and `.venv/bin/python` instead of `.venv/Scripts/python`. |

## Map of the code

```text
blueprint3d/
  dxf_to_scene.py        DXF floor plan  -> scene.json   (stdlib)
  ifc_to_scene.py        IFC model       -> scene.json   (stdlib; own STEP reader)
  raster_to_scene.py     image plan      -> scene.json   (CubiCasa5K network, or --engine opencv)
  cubicasa_model.py      loads + runs the CubiCasa5K network (weights_only, CPU)
  build_demo_scenes.py   builds the demo set in one command
  eval_*.py              accuracy against Stoa's questions.jsonl
  viewer/                index.html + app.js: linked 2D/3D viewer, Q&A, .glb export (three.js, web-ifc)
README.md                project overview, approach, results
SETUP.md                 this file
```

All three converters write the same `scene.json` format, where every element has an ID, a source layer or class, and geometry in metres. The viewer renders that format the same way whatever its source.

## Ground rules for teammates and agents

- **Never commit** `junction-dataset/`, `.venv/`, `third_party/` or `blueprint3d/viewer/scenes/`. They're gitignored; keep it that way.
- **Keep credits** for any drawing you show or share (`data/<source>/LICENSE.md`), and respect NC/SA/ND licences.
- **Answers need evidence.** Every number the viewer shows must be traceable to highlighted elements. If a tool can't answer, it says so instead of guessing.
- **Measure before you claim.** Run the matching `eval_*.py` after changing a converter. For raster plans, only the `--split test` number is a fair accuracy figure.
- Work on a branch and open small PRs (see the README's team agreement).

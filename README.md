# Stoa — Make Blueprints Talk

> Ask a question about a construction drawing. Get an answer you can verify in seconds — in 2D *and* in 3D.

Our Junction 2026 project for the **Stoa** challenge **Make Blueprints Talk**: a program that answers natural-language questions about construction drawings and building information models (BIM), **accurately and verifiably**.

Two things we are building:

1. **Blueprint Q&A (the core challenge).** Ask in any language ("How many doors are on this floor?", "Kuinka monta saunaa?", "What is the net area of the kitchen?"). You get the answer *and* the exact place on the drawing it came from, highlighted, so an engineer can check it in seconds.
2. **Blueprint → 3D (our addition).** We turn 2D plans into an interactive 3D model of the same building. Clicking a room or wall in 3D highlights its source on the 2D sheet, and every answer is highlighted in both views. The 3D model is a second way to check an answer, not a separate feature.

## Plumbline: the app ✅

```bash
uv sync
uv run uvicorn stoa_qa.api:app --port 8765     # then open http://localhost:8765
```

The dataset goes in `junction-dataset/junction-dataset/` (see [SETUP.md](SETUP.md) step 2). Put `FEATHERLESS_API_KEY=...` in `.env` to enable free-form questions; without it the app still runs, geometry only.

What you get:

- **Every drawing in the dataset** (13,943) in one browser: search, filter by source, thumbnails, and the licence credit under each sheet.
- **Ask in any language.** Each answer card shows the value, whether it matches Stoa's answer (when the question is from the dataset), the confidence, which engine answered, the response time, and the LLM tokens and cost (zero for most answers).
- **Evidence on the original sheet.** Every counted door, room, fixture or schedule row is outlined and numbered where it is on the drawing. Each one also gets a thumbnail, and ← → steps through them one at a time.
- **IFC answers light up the real elements** on the storey plan (matched by IFC GlobalId) and in the linked 3D model.
- **Free-form questions** ("Which room is the largest?", "How many windows per storey?"): Gemma 4 writes one query, shown on the card, and the engine computes the result from the drawing.
- **A benchmark view** with the measured numbers below.

How a question is answered, cheapest trustworthy engine first (`stoa_qa/pipeline.py`):

| Step | Used for | LLM use | Evidence shown |
| --- | --- | --- | --- |
| 1. Geometry / BIM | dataset-style questions on CAD and IFC drawings | none | polygons on the sheet, IFC elements |
| 1. Detection hybrid | counts on raster plans | none at question time for evaluated plans (network and Gemma counts are precomputed); one Gemma call for a new plan | the network's detected shapes |
| 2. Query planner | free-form questions on CAD and IFC | Gemma 4 plans one query: about 1,200 tokens, $0.00015 | the rows the query returned |
| 3. Sheet text | PDF sheets with real text, e.g. schedules | Gemma picks the rows; the count is done by us | each picked row, highlighted |
| 4. Vision | scans and anything without structure | Gemma reads the image | the model's own boxes, flagged low confidence |

Measured results (see the Benchmark button in the app, built by `python -m stoa_qa.report`):

| What | Questions | Accuracy |
| --- | --- | --- |
| Swiss CAD floor plans, 4 languages | 37,584 | **99.49%** |
| IFC models, 9 languages | 3,062 | **99.90%** |
| Raster plans, CubiCasa5K held-out test split: hybrid | 3,398 | **65.9%** (network alone 50.2%, Gemma 4 alone 63.4%) |
| Raster plans when two independent sources agree (34% of answers) | | **86.9%** — this is what the confidence badge reports |
| Door and finish schedules on real PDF sheets: row counts (first version) | 39 | 51.3% (20 of 39). Misses are mostly questions that need a quantity column summed rather than rows counted |

Gemini 3.5 Flash alone scored 83.4% on raster plans, but on only 25 plans before the free-tier quota ran out, so it is not part of the hybrid yet.

Tests: `uv run pytest` runs 26 tests on synthetic drawings (no dataset or API key needed); `.github/workflows/ci.yml` runs them on every push.

Deploy with a public URL (CPU only): `modal run modal_jobs/serve.py::load_data`, then `modal deploy modal_jobs/serve.py`.

## The problem

Site engineers spend hours every day looking for information in dense drawing sets. General-purpose AI often gets technical drawings wrong: it miscounts symbols, misreads scales and invents dimensions. In construction an answer that only *sounds* right is useless. People need to see where it came from.

## Our approach: compute, don't guess

We don't ask a vision model to "look at the picture and answer". We pull **structured geometry and text** out of the drawing first, then compute the answer from that structure:

- **Counts** (doors, windows, rooms, apartments, saunas…) come from detected and labelled objects, and each counted item gets its own highlight.
- **Areas, lengths and volumes** come from polygons and the drawing scale, not from the model's estimate.
- **Text/lookup questions** (schedules, title blocks, IFC metadata) come from OCR or IFC properties, cited by region or element ID.
- A **frontier LLM** handles the language side: reading the question in any language, picking the right tool, and writing the answer. It never invents a number.

This is a hybrid of LLM, computer vision and geometry/3D, which is the kind of solution Stoa asked for because better frontier models next year won't make it obsolete.

```text
Question (any language)
        ↓
LLM router → question type + target (room, element, sheet)
        ↓
Structured drawing index: rooms, walls, doors, windows, text, scale
        ↓
Deterministic tool computes the answer (count / area / length / volume / lookup)
        ↓
Answer + confidence + evidence (2D region highlight + 3D element highlight)
```

## The dataset (what we're working with)

Stoa's dataset has about 14,000 drawings in about 30 languages and **about 127,000 questions** in `questions.jsonl` (`path, question, answer, unit, type`).

| Question type | Count | Scoring |
| --- | --- | --- |
| count | ~111,800 | exact |
| area | ~14,200 | within 2% |
| text | ~840 | free-form |
| volume | ~140 | within 2% |
| length | ~60 | within 2% |

| Source | Format | Questions | What it means for us |
| --- | --- | --- | --- |
| `cubicasa5k` | PNG raster plans (Finnish) | ~85,500 | Most of the questions. Needs CV: detect doors/windows/rooms, OCR Finnish labels (MH, KPH, S…) |
| `swiss_dwellings` | PDF + **DXF** + PNG | ~37,600 | Vector geometry is available, so exact areas/counts and a direct 2D→3D extrusion |
| `ifc_models` | **IFC** (+ rendered PDF/PNG/SVG) | ~3,100 | Already 3D. Read rooms, areas and quantities straight from the model |
| `ifc_bench` | IFC | ~820 | Text/metadata questions about the model |
| `opening_design` | PDF sheets, DWG, DXF, IFC | ~60 | Real drawing sets with schedules and title blocks |
| `habs`, `finna`, `europeana`, `wikimedia` | scans (JPG/SVG) | 0 | No questions. Useful for robustness demos and multilingual OCR |

Our plan follows these numbers: **counts on raster plans are where the score is**, areas come second, and DXF/IFC give us accurate ground truth for both answers and 3D.

## Q&A engine: `stoa_qa` (all languages) ✅

`stoa_qa/` answers Stoa's questions **in every language they're asked in** and returns each answer with its evidence: the polygons or IFC GlobalIds that the viewer highlights. It has no LLM and no network calls, and runs on a laptop CPU, so these answers cost €0 per question.

```text
question (any language) ─▶ lang.py: object class + quantity + scope (multilingual keyword stems)
                         ─▶ router.py: pick the solver by source
                               ├─ swiss.py  DXF: rooms, apartments, doors, windows, fixture symbols
                               └─ ifc.py    IFC: rooms, storeys, elements, quantity sets / geometry
                         ─▶ Answer(value, unit, confidence, method, evidence[polygon | IFC GlobalId])
```

**Full question set, all languages** (counts exact, areas/lengths/volumes within 2%):

| Source | Languages | Questions | Accuracy | Speed |
| --- | --- | --- | --- | --- |
| `swiss_dwellings` (DXF) | EN, DE, FR, IT | 37,584 | **99.49%** | 6 ms/question |
| `ifc_models` (IFC, and the PDF plans rendered from it) | EN, DE, NL, DA, ET, ES, PL, RO, IT | 3,062 | **99.90%** | 63 ms/question |

Swiss by question type: room-label counts, room areas, label areas, doors, windows, apartments and showers are 100%; WC 99.9%, bathtubs 99.6%, washbasins 99.5%, floor area 97.3%, apartment area 96.9%. **Fixtures (WC, washbasin, bathtub, shower) are classified from the raw `FIXTURE` geometry**: each symbol is a rectangle, identified by what is inside it (ellipse = WC, ellipse + dot = washbasin, 69-point rounded inner outline = bathtub, 5-point inner outline = shower).

IFC by question type: every category is 100% except named-room area (99.9%) and storey total area (98.7%). Lengths are converted from the model's units (some models store millimetres), IFC2X3 `IfcFlowTerminal`s are typed through their type object (lights, sanitary), and stairs split into riser parts are counted once.

Remaining misses: some unlabelled numbered rooms in Swiss plans are shafts that should be excluded. The PDF fill colour can tell them apart (`_analysis/swiss_pdf_fill.py`).

```bash
uv sync                                                        # Python 3.12 env with ezdxf, shapely, ifcopenshell, pymupdf
uv run python -m stoa_qa.eval --source swiss_dwellings         # full Swiss set, ~4 min
uv run python -m stoa_qa.eval --source ifc_models              # full IFC set, ~3 min
uv run python -m stoa_qa.eval --source swiss_dwellings --limit 50   # quick check on 50 random floors
uv run python -m stoa_qa.eval --sample 2000                    # stratified sample across sources × types
```

Predictions and per-intent failures are written to `eval_out/` (gitignored). The dataset is found at `junction-dataset/junction-dataset/` (same as SETUP.md); `STOA_DATASET=/path` overrides it.

## Blueprint → 3D: working prototype ✅

`blueprint3d/` turns **DXF floor plans**, **IFC building models** and **raster scans** into a **linked 2D + 3D model** you can ask questions about.

```text
plan.dxf  ──dxf_to_scene.py────▶ scene.json ─┐
plan.png  ──raster_to_scene.py─▶ scene.json ─┼─ viewer (three.js) ─▶ 2D plan ⇄ 3D model ─▶ .glb (Blender)
model.ifc ──ifc_to_scene.py────▶ scene.json ─┘    click / ask → same element highlighted in both views
     └──────── web-ifc (browser) ─▶ real BIM geometry, keyed by the same IFC entity IDs
```

| | DXF (`swiss_dwellings`) | IFC (`ifc_models`) | Raster (`cubicasa5k`) |
| --- | --- | --- | --- |
| 3D comes from | extruding plan polygons (heights assumed, shown on screen) | the BIM model itself, via `web-ifc` (nothing assumed) | walls, rooms, doors, windows and fixtures predicted by the CubiCasa5K network, then extruded; labelled **APPROXIMATE** |
| 2D comes from | the DXF vector drawing | a plan projection of the selected storey's 3D geometry, plus the dataset's rendered storey plan | the traced vectors, plus the original scan |
| Facts come from | DXF layers + polygon geometry | IFC entities, spatial containment, quantity sets; room geometry when quantities are missing | network detections (rooms by type, icons by class) |
| Accuracy on Stoa's questions | **99.2%** English (15,371 / 15,493); **99.49%** all languages via `stoa_qa` (37,392 / 37,584) | **98.9%** English (1,607 / 1,625); **99.90%** all languages via `stoa_qa` (3,059 / 3,062) | **50.2%** exact on held-out test plans (2,842 questions) |

### DXF accuracy

**Measured on Stoa's own ground truth.** The 3D model is built from the same scene that answers the questions, so this accuracy is also how accurate the 3D model is:

| `swiss_dwellings`, 1,040 floor plans, English questions | Correct | Accuracy |
| --- | --- | --- |
| Rooms by label ("How many 'Zimmer'?") | 6,365 / 6,365 | 100% |
| Room net area | 2,076 / 2,076 | 100% |
| Doors | 1,040 / 1,040 | 100% |
| Windows | 1,040 / 1,040 | 100% |
| Total area of a room label | 1,017 / 1,017 | 100% |
| Apartments | 1,026 / 1,040 | 98.7% |
| Total indoor floor area | 1,009 / 1,040 | 97.0% |
| Apartment net area | 1,798 / 1,875 | 95.9% |
| **All** | **15,371 / 15,493** | **99.2%** |

Fixture counts (WC, washbasin, bathtub, shower) and the German/French/Italian questions are answered by `stoa_qa/swiss.py`; see [Q&A engine](#qa-engine-stoa_qa-all-languages-).

### IFC accuracy

`ifc_models`, 29 models, English questions (counts exact, quantities within 2%):

| Question | Correct | Accuracy |
| --- | --- | --- |
| Net floor area of a named room on a storey | 1,179 / 1,183 | 99.7% |
| Doors / windows / walls / columns per storey | 121 / 121 | 100% |
| Wall length, wall net side area, wall and slab net volume per storey | 133 / 133 | 100% |
| Doors / windows / walls / beams / columns in the whole model | 85 / 85 | 100% |
| Total room area per storey | 38 / 40 | 95.0% |
| Stairs (per storey / whole model) | 47 / 50 | 94.0% |
| Sanitary fixtures / unsupported wording | 4 / 13 | — |
| **All** | **1,607 / 1,625** | **98.9%** |

About 730 of those rooms have **no stored area** in the IFC file. Their areas come from the room geometry: the profile of an extruded solid, or the plan-projected floor faces of a Brep. All of this uses only the Python standard library, no geometry kernel.

### Raster (scanned plans): pretrained CubiCasa5K network

`raster_to_scene.py` runs the **CubiCasa5K multi-task network** (Kalervo et al., SCIA 2019; code in `third_party/CubiCasa5k`, CC BY-NC 4.0). It predicts wall junctions, 12 room classes and 11 icon classes, which become walls, typed rooms (bedroom, kitchen, bath…), doors, windows and fixtures (toilet, sink, bathtub, sauna bench) in the same `scene.json`. The scale comes from the detected doors (0.9 m). The 3D model is approximate, and the viewer says so.

**Fair evaluation.** Stoa's cubicasa5k images are byte-identical to CubiCasa5K's `F1_original.png` files, and the network was trained on CubiCasa's train split. So we score **only the 387 test-split plans it never saw** (`eval_cubicasa_model.py --split test`, 2,842 questions):

| Question | Exact | ±1 | OpenCV baseline (exact) |
| --- | --- | --- | --- |
| Saunas | 85.3% | 100% | — |
| Kitchens | 83.2% | 99.2% | — |
| Toilets | 70.7% | 92.0% | — |
| Bathrooms | 61.2% | 93.3% | — |
| Balconies | 60.0% | 84.0% | — |
| Bedrooms | 54.4% | 84.5% | — |
| Sinks | 47.4% | 79.1% | — |
| Doors | 33.5% | 68.0% | 4–10% |
| Windows | 30.2% | 54.8% | 0–20% |
| Showers + bathtubs | 6.3% | 80.3% | — |
| **All** | **50.2%** | | |

That's a large step up from the hand-tuned OpenCV baseline (`--engine opencv`), but far from the ~99% we get on DXF and IFC. Known gaps:
- **Showers:** the network has no shower class, so "showers and bathtubs" is almost always undercounted.
- **Doors and windows:** neighbouring openings merge or split during vectorisation.
- **Room types:** the Finnish labels on the drawings (MH, KPH, WC, S) aren't read yet. OCR would both fix room types and separate WC rooms from bathrooms.
- **Floors per plan:** not attempted.

**Run it**

```bash
# 1. convert drawings/models to scenes (DXF/IFC: Python standard library only)
py -I blueprint3d/dxf_to_scene.py junction-dataset/junction-dataset/data/swiss_dwellings/site1000_bldg2082_floor9385.dxf
py -I blueprint3d/ifc_to_scene.py junction-dataset/junction-dataset/data/ifc_models/kit-fzk-haus_ac20-fzk-haus.ifc
#    raster needs the venv + CubiCasa5K weights (see "Local setup" below); --engine opencv for the baseline
.venv/Scripts/python -W ignore blueprint3d/raster_to_scene.py junction-dataset/junction-dataset/data/cubicasa5k/high_quality-10004.png

# 2. serve the repo root and open the viewer (or use the "blueprint3d" entry in .claude/launch.json)
py -m http.server 8765
#    → http://localhost:8765/blueprint3d/viewer/

# 3. reproduce the accuracy tables
py -I blueprint3d/eval_swiss.py
py -I blueprint3d/eval_ifc.py
.venv/Scripts/python -W ignore blueprint3d/eval_cubicasa_model.py --split test   # network, held-out plans
.venv/Scripts/python blueprint3d/eval_raster.py --per-style 40                     # OpenCV baseline
```

**Viewer features**

- 2D plan and 3D model side by side, plus an "Original drawing" tab showing the source PNG (DXF) or the rendered storey plan (IFC).
- Click any room, wall, door or window in either view to highlight it in both. The panel shows its ID, source layer or IFC class, storey, and computed quantities.
- Ask in English, German or French (IFC also takes Danish and Dutch keywords). Answers highlight the evidence in both views and say how they were computed. Questions it can't answer get "I can't answer that" rather than a guess.
- IFC: a storey picker, an **Isolate storey** toggle, and auto-jump to the storey that holds the evidence.
- X-ray mode (see-through walls), top view, and **Export .glb** to open the model in Blender.
- DXF only: assumed heights (storey 2.6 m, door head 2.1 m, window 0.9–2.1 m) are always shown on screen. IFC needs no assumptions.

### Routes by format

How much work 3D takes depends on the format:

| Input | 2D → 3D route | Accuracy | Effort |
| --- | --- | --- | --- |
| **IFC** | ✅ **Built**: `web-ifc` streams the real geometry into three.js; `ifc_to_scene.py` supplies storeys, rooms, quantities | Exact (98.9% on ground truth) | Done |
| **DXF** (swiss_dwellings, opening_design) | ✅ **Built**: read the `WALL` / `ROOM_OUTLINE` / `DOOR` / `WINDOW` / `UNIT_OUTLINE` layers, extrude in three.js, export `.glb` | Exact (99.2% on ground truth) | Done |
| **Raster PNG/JPG** (cubicasa5k, scans) | ✅ **Built**: CubiCasa5K network → walls, typed rooms, doors, windows, fixtures → extrude (`raster_to_scene.py`; `--engine opencv` baseline) | Approximate, labelled as such; 50.2% on held-out test plans | Next: OCR of room labels, shower class |

Rules for 3D:

- **Every 3D object keeps the ID of its 2D source** (room polygon, wall segment, IFC GlobalId). That shared ID is what links the two views.
- **Heights we assume are shown in the UI** (e.g. "storey height assumed 2.7 m"). A 2D plan alone doesn't give you volume, so we never present an assumed height as fact.
- **Blender is optional.** The live pipeline is plain Python (no dependencies) plus three.js in the browser, so it runs anywhere. Click **Export .glb**, then in Blender use *File → Import → glTF* for polished renders and the demo video. Element IDs are kept as object names.
- **Browser viewer:** three.js for all three formats, with web-ifc streaming native IFC geometry.

Next steps for 3D, in priority order:

1. **Raster accuracy** (`cubicasa5k`, which has 85k questions): add OCR of the Finnish room labels (MH, KPH, WC, S) to fix room types and split WC rooms from bathrooms, handle showers, and tune opening vectorisation. Same `scene.json`, same viewer.
2. **`ifc_bench`** (820 free-text questions on 30 more IFC models): the same reader, plus an LLM that writes the answer from the extracted facts.
3. **opening_design DXF/DWG:** their layer names differ from the Swiss ones, so they need a layer-mapping table.

## Supported questions (first version)

Chosen to match the dataset's question distribution:

1. **Counts:** doors, windows, rooms by type (bedroom, bathroom, sauna, toilet), apartments per floor.
2. **Areas:** net floor area of a named room or the total for a floor.
3. **Lookups:** schedule entries (door types on sheet A-801), IFC metadata (authoring tool, schema, storeys).
4. **Volumes/lengths:** from IFC quantities first, then from DXF geometry.
5. **Multilingual:** the same question in English and Finnish (and other dataset languages) gives the same answer.

If the system isn't sure, it says so and shows what it found, instead of guessing.

## Architecture

```text
Drawings (PDF / PNG / JPG / DXF / DWG / IFC / SVG)
        ↓
Ingest per format
  ├─ raster: render → CV detection (doors, windows, walls, rooms) + OCR
  ├─ vector: DXF/SVG/PDF → polygons, layers, text, scale
  └─ BIM: IFC → spaces, elements, quantities, properties
        ↓
Unified drawing index (rooms, elements, text, bbox/polygon, source IDs)
        ├──────────────→ 3D builder (extrude / IFC → glTF)
        ↓
Q&A agent: LLM router + deterministic tools (count, area, length, volume, lookup)
        ↓
Answer + confidence + evidence
        ↓
UI: question box │ 2D viewer with highlights │ 3D viewer with linked highlights
```

## Planned stack

| Area | Choice | Why |
| --- | --- | --- |
| UI | Next.js / React | Question interface and 2D/3D viewers |
| 2D viewer | PDF.js / OpenSeadragon | Deep zoom and highlight overlays on large sheets |
| 3D viewer | three.js + web-ifc (built) | Extruded DXF plans and native IFC in the browser |
| API & processing | Python + FastAPI | PDF, CAD, CV and geometry tooling |
| Vector/CAD | `ezdxf`, `shapely`, PyMuPDF | Exact geometry, text and scale from DXF/PDF |
| BIM | Own stdlib IFC/STEP reader (`blueprint3d`, viewer); IfcOpenShell (`stoa_qa`, answers) | IFC storeys, spaces, quantities, room geometry areas, unit scaling |
| 3D generation | `trimesh` (+ Blender headless for renders) | 2D polygons → 3D `.glb` |
| CV | YOLO-style detector / segmentation model fine-tuned on CubiCasa labels | Door/window/room counts on raster plans |
| OCR | PaddleOCR / Tesseract (+ LLM vision fallback) | Multilingual labels, schedules, title blocks |
| Reasoning | Frontier LLM (Gemini, Claude, or what the hackerpacks provide) | Question parsing, tool routing, answer writing |

## Junction hackerpack tools

Checked against the actual hackerpacks and tech challenges on 2026-11-14. These are the side challenges we plan to enter, in priority order:

| Tech challenge (prize) | Its role in Make Blueprints Talk | Status |
| --- | --- | --- |
| **Gemma** (€1k, top 3) | Gemma 4 31B (via Featherless) plans queries for free-form questions, picks schedule rows on PDF sheets, counts on raster plans and reads scans. Numbers are always computed by us | **In the app** |
| **Modal** ($2k + credits) | Ran the CubiCasa network on 769 plans in parallel (minutes instead of hours on a laptop); detector dataset and training jobs; deploy script for a public URL | Batch jobs done; training needs a payment method for GPU; deploy script ready |
| **Gemini API** (€4k + GCP credits) | "Frontier model only" baseline and a boxes mode on raster plans (`stoa_qa/cubicasa_gemini.py`) | Baseline on 25 plans; waiting for hackathon credits |
| **CRACI** (LEGO + interview) | CI runs the test suite on every push; set the repo variable `CI_RUNNER=craci` to run it on CRACI | Workflow ready |
| **Cloudinary** ($500 per member) | Evidence crops and sheet thumbnails are generated by our own server for now | Not integrated |
| **Aiven** (€1k) | Postgres + PostGIS store for the drawing graph (map-style queries: "doors inside room X") | Not integrated |
| Nemotron VL, ElevenLabs | Optional: a second vision model for the agreement-based confidence check; voice questions on site | If time |

Credits: Gemini (one form per team, 18+), Modal (code `VVX-2QX-8HW`, $100 per person), Anthropic ($20 per person), Aiven ($50), Cloudinary (survey form), CRACI (invite code in #craci), IronShard (`JUNCTION26`, dataset and model storage), Featherless (`JUNCTION26`, hosted Gemma).

Rule: we only enter a side prize if the tool is actually part of the working demo.

## Evaluation

The dataset comes with an official question set, so we measure ourselves on it from day one:

- Run a fixed, stratified sample (by source × type × language) after every pipeline change.
- Report exact-match rate for counts and the share of areas/lengths/volumes within 2%.
- Track **evidence coverage**: how many answers come with a correct highlighted source.
- Compare against a baseline of "send the image to a frontier LLM and ask" to show what our structured approach adds.

## Milestones

- [x] Data loader for `questions.jsonl` and an evaluation harness (`stoa_qa/eval.py`: full, per-source, stratified sample).
- [x] IFC path: spaces, areas and quantities → answers (99.90% all languages); real BIM geometry in the linked 3D viewer.
- [x] DXF path (swiss_dwellings): room polygons → counts/areas (99.49% all languages); extrusion → linked 3D viewer.
- [x] Fixture symbol classification (WC, washbasin, bathtub, shower) on the DXF `FIXTURE` layer (99.5–100%).
- [x] Multilingual questions without an LLM (`stoa_qa/lang.py`, 10 languages).
- [x] Raster path (cubicasa5k), first version: pretrained network + Gemma 4 hybrid, 65.9% on held-out plans (network alone 50.2%).
- [ ] Raster path, next: detector trained on CubiCasa annotations for doors, windows and fixtures (label extraction and training jobs are in `modal_jobs/`; training needs a GPU).
- [x] Vision baselines on held-out plans: Gemma 4 on all 387 plans; Gemini 3.5 Flash on 25 (free-tier quota).
- [x] One answer API (`/api/ask`): every format shares the same answers and evidence; the 3D viewer uses it too.
- [x] Free-form questions: Gemma 4 plans a query over the drawing's fact tables (`stoa_qa/planner.py`); covers many `ifc_bench` questions.
- [x] UI: question → answer → evidence on the original sheet → linked 3D highlight (`web/`).
- [x] Uncertainty: confidence from source agreement, "I can't answer that" instead of a guess, vision answers flagged as estimates.
- [x] Schedules on PDF sheets: rows selected by Gemma, counted and highlighted by us (`stoa_qa/docqa.py`).
- [x] Unit tests on synthetic drawings and CI.
- [ ] Public deployment (`modal_jobs/serve.py` is ready).
- [ ] Two-minute live demo + accuracy slide against the baseline.

## Team roles

| Role | Focus |
| --- | --- |
| CV / document AI | Raster detection, OCR, PDF/DXF parsing, scale detection |
| BIM / 3D | IFC parsing, DXF extrusion, glTF export, 3D viewer |
| AI / agent | LLM router, tools, prompts, evaluation harness |
| Frontend | 2D/3D viewers, linked highlights, Q&A interface |
| Product / demo | Question selection, pitch, demo script, licence credits |

## Definition of a winning demo

In under two minutes a judge asks a real question (in English or Finnish) and:

1. gets the correct answer in seconds;
2. sees exactly which doors, rooms or labels it counted, highlighted on the original drawing;
3. switches to 3D, sees the same elements highlighted in the model, and can walk around them;
4. sees our measured accuracy on Stoa's own question set compared with a frontier-model-only baseline.

## Principles

- Trust is the product. No answer without evidence, and uncertainty is always shown.
- Compute numbers from geometry. Don't let the LLM estimate them.
- 2D and 3D share IDs. A 3D view you can't trace back to the drawing is just decoration.
- Keep credits for every drawing we show (see `data/<source>/LICENSE.md`); respect NC/SA/ND licences.

## Local setup

**Full step-by-step guide: [SETUP.md](SETUP.md).** It's written so a teammate, or their coding agent, can follow it from a fresh clone. Quick version:

```bash
# dataset unzipped to junction-dataset/junction-dataset/... (see SETUP.md step 2), then:
py -I blueprint3d/build_demo_scenes.py      # DXF + IFC scenes, standard library only
py -m http.server 8765                      # open http://localhost:8765/blueprint3d/viewer/
```

Raster plans (CubiCasa5K network) need a venv, PyTorch and the model weights: see SETUP.md step 4.

The app and the multilingual Q&A engine use [uv](https://docs.astral.sh/uv/): `uv sync`, then `uv run uvicorn stoa_qa.api:app --port 8765` (see [Plumbline: the app](#plumbline-the-app-)). To score the engine: `uv run python -m stoa_qa.eval --source swiss_dwellings --limit 50`. Secrets go in `.env` (gitignored): `FEATHERLESS_API_KEY` (Gemma), `GEMINI_API_KEY`.

## Useful links

- [Junction 2026](https://hackjunction.com/)
- [web-ifc](https://github.com/ThatOpen/engine_web-ifc) · [three.js](https://threejs.org/) · [IfcOpenShell](https://ifcopenshell.org/) · [ezdxf](https://ezdxf.mozman.de/)

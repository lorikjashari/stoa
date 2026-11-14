# Stoa — Make Blueprints Talk

> Ask a question about a construction drawing. Get an answer you can verify in seconds — in 2D *and* in 3D.

Our Junction 2026 project for the **Stoa** challenge **Make Blueprints Talk**: a program that answers natural-language questions about construction drawings and building information models (BIM), **accurately and verifiably**.

Two things we are building:

1. **Blueprint Q&A (the core challenge).** Ask in any language ("How many doors are on this floor?", "Kuinka monta saunaa?", "What is the net area of the kitchen?"). You get the answer *and* the exact place on the drawing it came from, highlighted, so an engineer can check it in seconds.
2. **Blueprint → 3D (our addition).** We turn 2D plans into an interactive 3D model of the same building. Clicking a room or wall in 3D highlights its source on the 2D sheet, and every answer is highlighted in both views. The 3D model is a second way to check an answer, not a separate feature.

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

## Blueprint → 3D: is it possible?

**Yes, and part of it is close to free.** How much work it takes depends on the format:

| Input | 2D → 3D route | Accuracy | Effort |
| --- | --- | --- | --- |
| **IFC** | Already a 3D model. Convert to glTF with IfcOpenShell / `IfcConvert`, or load in the browser with `web-ifc` | Exact | Low |
| **DXF** (swiss_dwellings, opening_design) | `ezdxf` reads the wall/room layers, `shapely` builds polygons, then we extrude to storey height with `trimesh` (or Blender) and export `.glb` | High (real geometry) | Medium |
| **Raster PNG/PDF** (cubicasa5k, scans) | CV wall/door/window segmentation, vectorise to polygons, extrude | Approximate, labelled as such | High |

Rules for 3D:

- **Every 3D object keeps the ID of its 2D source** (room polygon, wall segment, IFC GlobalId). That shared ID is what links the two views.
- **Heights we assume are shown in the UI** (e.g. "storey height assumed 2.7 m"). A 2D plan alone doesn't give you volume, so we never present an assumed height as fact.
- **Blender is optional.** Headless Blender (`blender -b -P extrude.py`) is good for polished renders and a demo video, but the live pipeline uses plain Python (`ezdxf` + `shapely` + `trimesh` → glTF) so it runs on a server without a GPU or Blender install.
- **Browser viewer:** Three.js / react-three-fiber for the extruded plans, and That Open Engine (`@thatopen/components`, formerly IFC.js) for native IFC.

Scope: 3D only needs to work well for DXF and IFC drawings. Raster-to-3D is a stretch goal; if we get there it's labelled "approximate".

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
| 3D viewer | Three.js / react-three-fiber, That Open Engine | Extruded plans and native IFC in the browser |
| API & processing | Python + FastAPI | PDF, CAD, CV and geometry tooling |
| Vector/CAD | `ezdxf`, `shapely`, PyMuPDF | Exact geometry, text and scale from DXF/PDF |
| BIM | IfcOpenShell | IFC spaces, quantities, glTF export |
| 3D generation | `trimesh` (+ Blender headless for renders) | 2D polygons → 3D `.glb` |
| CV | YOLO-style detector / segmentation model fine-tuned on CubiCasa labels | Door/window/room counts on raster plans |
| OCR | PaddleOCR / Tesseract (+ LLM vision fallback) | Multilingual labels, schedules, title blocks |
| Reasoning | Frontier LLM (Gemini, Claude, or what the hackerpacks provide) | Question parsing, tool routing, answer writing |

## Junction hackerpack tools

> ⚠️ **To confirm:** the hackerpacks page (`hackjunction.app/participate/45/hackerpacks`) needs a sign-in and hasn't been checked yet. The list below is our current shortlist from the Junction 2026 partner/prize list. Update it once someone has read the actual hackerpacks.

| Tool | How we'd use it in Make Blueprints Talk |
| --- | --- |
| **Gemini** | Multimodal model for routing questions, multilingual understanding, and vision fallback on hard sheets |
| **Modal / Verda** | GPU jobs: batch CV inference over ~14k drawings, fine-tuning the door/window detector, IFC/DXF → glTF conversion |
| **Cloudinary** | Serving sheet tiles and thumbnails, cropping and annotating evidence regions |
| **Aiven / MariaDB** | Drawing index: elements, polygons, text, source IDs, plus evaluation results |
| **IronShard** | Storing processed drawings, generated `.glb` models and model checkpoints |
| **Codex** | Coding assistant during the build |
| **CRACI** | CI with reproducible builds and supply-chain visibility |

Rule: we only enter a side prize if the tool is actually part of the working demo.

## Evaluation

The dataset comes with an official question set, so we measure ourselves on it from day one:

- Run a fixed, stratified sample (by source × type × language) after every pipeline change.
- Report exact-match rate for counts and the share of areas/lengths/volumes within 2%.
- Track **evidence coverage**: how many answers come with a correct highlighted source.
- Compare against a baseline of "send the image to a frontier LLM and ask" to show what our structured approach adds.

## Milestones

- [ ] Data loader for `questions.jsonl` and a stratified evaluation harness with baseline numbers.
- [ ] IFC path: spaces, areas and quantities → answers; IFC → glTF in the 3D viewer.
- [ ] DXF path (swiss_dwellings): room polygons → counts/areas; extrusion → 3D.
- [ ] Raster path (cubicasa5k): door/window/room detection + Finnish OCR → counts.
- [ ] LLM router with tools; multilingual question handling.
- [ ] UI: question → answer → 2D highlight; linked 3D highlight.
- [ ] Uncertainty handling and "no answer" responses.
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

The repo is still at kickoff. Setup commands will be added once the frontend/backend structure exists. The dataset (`junction-dataset/`) isn't committed. Download the preview from Stoa's link in the challenge brief.

## Useful links

- [Junction 2026](https://hackjunction.com/)
- [IfcOpenShell](https://ifcopenshell.org/) · [That Open Engine](https://thatopen.com/) · [ezdxf](https://ezdxf.mozman.de/) · [trimesh](https://trimesh.org/)

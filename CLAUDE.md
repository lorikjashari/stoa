# CLAUDE.md — Stoa / Plumbline

Context for Claude Code sessions (and teammates' agents). Read this first; README.md has the full results tables,
SETUP.md the step-by-step install. Keep this file short and current: when a fact here goes stale, fix it here.

## What this is

Junction 2026 hackathon project for Stoa's **"Make Blueprints Talk"** challenge. **Plumbline** answers natural-language
questions (any language) about construction drawings and BIM models, and every answer comes with **evidence highlighted
on the original sheet** (and in a linked 3D model). Remote: `github.com/lorikjashari/stoa`, branch `main`.

Core principle — **compute, don't guess**: numbers come from geometry, BIM data or detector boxes, never from an LLM's
estimate. The LLM (Gemma 4 31B via Featherless) only plans queries, picks rows, or — as a last resort — reads an image,
and those answers are flagged as estimates. "I can't answer that" beats a guess.

## Layout

```text
stoa_qa/        Python package: the answer engine + FastAPI app (the product)
blueprint3d/    2D -> 3D scene converters (DXF, IFC, raster, mesh) + the three.js viewer (viewer/)
modal_jobs/     Training / batch / deploy jobs (Modal cloud, plus local-GPU equivalents)
web/            Plumbline web app front end (vanilla JS, no build step), served by stoa_qa.api
tests/          pytest on a synthetic dataset built in conftest.py (no real data, no LLM)
_analysis/      One-off research scripts (rule fitting, sweeps, probes). Not imported by the app.
```

Not in git (gitignored, must exist locally for most real work): `junction-dataset/` (Stoa data, ~12 GB, licensed
per file), `third_party/` (CubiCasa5k repo, weights, network prediction cache, cubicasa5k.zip), `eval_out/`
(eval outputs, **YOLO prediction boxes**, Gemma/Gemini caches), `.venv/`, `.env`, `blueprint3d/viewer/scenes/`, `*.pt`, `*.pkl`.

## The dataset (`junction-dataset/junction-dataset/`)

`questions.jsonl` rows: `path, question, answer, unit, type` (~127k questions, ~14k drawings, ~30 languages).
Scoring: counts exact; area/length/volume within 2%; `text` is free-form.
Dataset root resolves in `stoa_qa/core.py:DATASET`; override with env `STOA_DATASET`.

| Source | Format | Answered by |
| --- | --- | --- |
| `swiss_dwellings` | DXF (+PDF/PNG) | `stoa_qa/swiss.py` — ezdxf geometry, layers `WALL/ROOM_OUTLINE/DOOR/WINDOW/UNIT_OUTLINE/FIXTURE` |
| `ifc_models`, `ifc_bench` | IFC | `stoa_qa/ifc.py` (+ `ifc_props.py`, `mep.py`) — IfcOpenShell |
| `cubicasa5k` | PNG raster (Finnish) | `stoa_qa/cubicasa.py` — reads **precomputed** detector/network/LLM predictions |
| `opening_design` | real PDF sets | `docqa.py`, `xref.py`, `legend.py` (PDF text layer via PyMuPDF) |
| `habs`, `finna`, `europeana`, `wikimedia` | archive scans | no questions; `blueprint3d/archive_plan.py` finds a plan on the sheet for 3D |

## How a question is answered

Entry: `stoa_qa/api.py` `POST /api/ask` → `pipeline.respond()` → `pipeline.solve()`. `solve` is a "crew", cheapest
trustworthy engine first; each one may decline and pass on:

0. `mep` (IFC services wording) and `graph` (room topology: routes, adjacency, doors between rooms) — no LLM
1. template engines via `router.SOLVERS` (`swiss` / `ifc` / `cubicasa`) — no LLM, for dataset-shaped questions.
   `lang.py` parses object/quantity/scope from multilingual keyword stems; `scope.py` refuses to answer a
   room/apartment-scoped question with a floor-wide number.
2. `planner` — Gemma writes ONE JSON query over fact tables; we execute it (`planner.execute`) and show the plan
3. `xref` (callouts like `6A/A-401`) and `docqa` (Gemma picks PDF text rows; we count them)
4. `vision` — Gemma reads the image; flagged as an estimate

Then `audit.run()` re-checks the answer against independent evidence and sets confidence; `explain.py` turns a refusal
into reasons + clickable alternative questions; `pipeline.overlay()` maps evidence into the shown sheet's coordinates
(`sheets.py`). Every `Answer` (`core.py`) carries `Evidence` items (polygon / IFC GlobalId / box). Each ask is logged to
`eval_out/ask_log.jsonl`.

`llm.py`: OpenAI-compatible client to Featherless. Key `FEATHERLESS_API_KEY` from env or `.env`; model
`STOA_LLM_MODEL` (default `google/gemma-4-31B-it`). `STOA_NO_LLM=1` forces geometry-only mode (tests use it).

## How the AI is trained (raster plans only)

DXF and IFC need no training — they are read exactly (~99.5% / ~99.9%). Training exists **only for `cubicasa5k`
raster plans**, which are ~85k of the 127k questions.

**Key insight:** Stoa's cubicasa5k images are byte-identical to CubiCasa5K's `F1_original.png`, and Stoa's answers are
counts of CubiCasa5K's own `model.svg` annotations. So a detector trained on those annotations learns exactly what Stoa
counts, and each counted item has a checkable box.

Pipeline (Modal cloud version / local-GPU version do the same steps and write the same files):

1. **Extract** — download CubiCasa5K zip (Zenodo 2613548, 5.5 GB); parse every `model.svg` into objects with bounding
   boxes, scaled to original image pixels (`modal_jobs/cubicasa_detector.py::extract` or `cubicasa_local.py extract`).
2. **Label rules** — `stoa_qa/cubicasa_rules.py` maps SVG classes to **13 detector classes**
   (`door, window, toilet, sink, shower, bathtub, floor, sauna, kitchen, bedroom, wc_room, bathroom, balcony`; order =
   YOLO class id, keep stable). Rules were fitted on the TRAIN split (`_analysis/raster_svg_rules.py`) and reproduce
   Stoa's answers on 98.4–100% of train plans. `QUESTION_CLASSES` maps the 12 questions to classes.
3. **YOLO labels** — `::labels` / `cubicasa_local.py labels` writes YOLO txt labels + `data.yaml`.
4. **Train** Ultralytics YOLO11 on the train split, flips as augmentation:
   - `local-y11s-13c`: YOLO11s, 1024 px, laptop GPU (`cubicasa_local.py train`)
   - `m13-1280`: YOLO11m, 1280 px, 60 epochs, Modal H100 (`cubicasa_cloud.py::fit_full`; test mAP50 0.901)
   - `tiles-dw-*`: door/window-only model on 640 px tiles of 2048 px plans (`cubicasa_tiles.py`) — higher mAP but
     counted worse (tile-border merging), so val does **not** pick it
   - `cubicasa_train.py` is the older first Modal trainer (7→13 classes); `cubicasa_cloud.py` supersedes it.
5. **Predict** low-confidence (0.05) boxes for val + test → `eval_out/yolo/<run>/{val,test}.json`.
6. **Fuse** two runs with weighted-box fusion: `python -m stoa_qa.cubicasa_ensemble m13-1280 local-y11s-13c --out ens-m13-s13`
   (agreed boxes averaged; a box only one model finds keeps half its score).
7. **Tune + select** — `python -m stoa_qa.cubicasa_hybrid --yolo-run ens-m13-s13` picks, **on val**, per question: the
   source (yolo / tiles / network / gemma / gemini), the score threshold and the duplicate-IoU cut. Writes
   `stoa_qa/cubicasa_picks.json` (committed; current: all 12 questions → yolo run `ens-m13-s13`, test 84.2% on 3,398 Qs).

Other raster sources the hybrid can use: the **pretrained CubiCasa5K network** (`blueprint3d/cubicasa_model.py`, run in
bulk by `modal_jobs/cubicasa_network.py` / `cubicasa_local_preds.py` → `third_party/cubicasa_preds/`, post-processed by
`cubicasa_refine.py`), and Gemma/Gemini image baselines (`stoa_qa/cubicasa_gemini.py` → `eval_out/gemini/`).
Confidence = agreement between sources (agree → 0.9, ~92% accurate; disagree → 0.5).

**User's own plans:** `modal_jobs/cubicasa_userplans.py` (local) / `cubicasa_userplans_cloud.py` (Modal) pseudo-label
→ correct in `tools/plan_labeler.html` → fine-tune `best.pt` with a CubiCasa replay mix → evaluate before/after on the
user's held-out test plans + CubiCasa test; logic in `stoa_qa/cubicasa_userplans.py` (README "Train on your own plans").

**Honesty rule (do not break):** train on CubiCasa's train split, choose thresholds/sources on val, report only on the
held-out **test** split (387 plans). Never tune on test. Current weak spots: windows (~49%), doors (~74%).

**At runtime the app does not run YOLO.** `cubicasa.answer` only looks up precomputed boxes in `eval_out/yolo/<run>/`
and `third_party/cubicasa_preds/`. A plan with no precomputed predictions falls back to one live Gemma call
(`pipeline._live_cubicasa_counts`). Licence: CubiCasa5K is CC BY-NC 4.0 — non-commercial, credit Kalervo et al. 2019.

## Commands

Package manager is **uv** (Python 3.11–3.13; `pyproject.toml`, `uv.lock`). Optional raster deps: `uv sync --extra raster`.

```bash
uv sync
uv run pytest                                          # 65 tests, synthetic data, no dataset/key needed (CI runs this)
uv run uvicorn stoa_qa.api:app --port 8765             # app at http://localhost:8765 (also serves /blueprint3d/)
uv run python -m stoa_qa.eval --source swiss_dwellings --limit 50    # quick accuracy check (needs dataset)
uv run python -m stoa_qa.eval --source ifc_models
uv run python -m stoa_qa.eval_text --limit 40          # free-form questions, LLM-graded (needs key)
uv run python -m stoa_qa.cubicasa_hybrid               # raster: select on val, report on test
uv run python -m stoa_qa.report                        # rebuild web/benchmark.json from eval outputs
modal run modal_jobs/serve.py::load_data && modal deploy modal_jobs/serve.py   # public CPU deploy
```

Windows notes: set `PYTHONIOENCODING=utf-8` if printing crashes. `.claude/launch.json` has preview entries
(`plumbline` on 8766, `plumbline-app` on 8765, `blueprint3d` static server on 8765); the plumbline entries expect a
`venv/` (not `.venv/`) interpreter, so adjust if your env lives elsewhere.

## Rules for changes

- **Evidence or nothing.** Any new answer path must return `Evidence` that can be highlighted; uncertain → decline.
- **Numbers never come from the LLM.** LLM output is a plan/selection that our code executes.
- **Measure before you claim.** After changing a solver or converter, run the matching eval; update README numbers
  only with measured results (`report.py` builds the benchmark view from real runs).
- **2D and 3D share IDs** (DXF handle / IFC GlobalId / element id) — keep `viewer_id` wiring intact.
- `swiss.py` and `ifc.py` are shared by the answers **and** the 3D converters — changes affect both.
- `cubicasa_rules.py` is stdlib-only (imported inside Modal jobs); keep `CLASSES` order stable.
- Never commit dataset, weights, `eval_out/`, `third_party/`, `.env`, or images in `_analysis/` (licensed drawings).
- Tests must stay offline: no dataset, no network, `STOA_NO_LLM=1` (see `tests/conftest.py`).
- Commit only when asked; work on a branch for non-trivial changes.

## Known gaps / next steps

Raster windows and doors; free-form text questions (~37% correct on `ifc_bench` + `opening_design`); PDF schedule
quantity sums (51%); public deploy not yet live. README's "Raster (scanned plans)" table mixes old network-only and new
detector columns — trust the top-of-README tables and `cubicasa_picks.json`.

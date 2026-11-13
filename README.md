# Blueprint Evidence Copilot

> Ask a question about a construction drawing. Get an answer you can verify in seconds.

Our Junction 2026 project for **Stoa — Make Blueprints Talk**. We are building an AI assistant that understands technical construction drawings and gives evidence-grounded answers: not only *what* it found, but the exact sheet and area where it found it.

## The problem

Construction engineers lose time searching through large, dense drawing sets. General AI can give plausible answers, but in construction a plausible answer is not enough: people need to see the source and verify it quickly.

## Our solution

Blueprint Evidence Copilot turns a natural-language question into a short, trustworthy answer with visual proof.

```text
Question: “Where is the fire exit on Level 2?”

Answer: “East stairwell, Corridor B.”
Evidence: the relevant sheet opens, zooms to the exit, and highlights the
symbol/label used for the answer.
```

## What makes this different

- **Evidence first:** every answer must cite a sheet and highlighted drawing region.
- **No confident guessing:** ambiguous or missing information returns an honest limitation.
- **Built for drawings:** combines OCR, document retrieval, coordinates, geometry, and multimodal AI instead of treating a whole blueprint as one image.
- **Focused scope:** we will solve a small set of high-value questions extremely well.

## Initial supported questions

We will refine these after seeing Stoa's dataset and example questions.

1. **Location:** Where is the fire exit, lift, electrical room, or stairwell?
2. **Dimensions:** What is the width/length of a marked corridor or room?
3. **Relationships:** Which rooms connect to a named stairwell or corridor?
4. **Counts:** How many doors, exits, or labelled rooms are on a floor?

## Proposed architecture

```text
Technical drawings (PDF / images / BIM)
        ↓
High-resolution sheet rendering + metadata extraction
        ↓
OCR, symbol/label detection, coordinates, sheet indexing
        ↓
MariaDB: text + regions + coordinates + source metadata
        ↓
Retrieve only relevant evidence for a question
        ↓
Gemini / vision model reasons over the retrieved evidence
        ↓
Answer + confidence + highlighted source region in the UI
```

## Planned stack

| Area | Choice | Why |
| --- | --- | --- |
| UI | Next.js / React | Fast, clear drawing viewer and question interface |
| API & processing | Python + FastAPI | Strong PDF, OCR, CV, and data tooling |
| Multimodal reasoning | Gemini | Interpret retrieved visual/textual drawing evidence |
| Compute | Modal or Verda | Run OCR, embeddings, and vision workloads reliably |
| Database | MariaDB | Store drawing text, regions, coordinates, and query evidence |
| Image delivery | Cloudinary | Efficient sheet previews, crops, zooming, and annotations |
| CI | CRACI | Reproducible builds and supply-chain visibility |

## First milestones

- [ ] Obtain dataset, formats, sample questions, and evaluation expectations from Stoa.
- [ ] Select one drawing format and 3 high-value question types.
- [ ] Render a drawing sheet and build a pan/zoom viewer.
- [ ] Extract OCR text with bounding-box coordinates.
- [ ] Build the first full flow: question → evidence retrieval → highlight → answer.
- [ ] Create a test set of 10–20 questions and measure accuracy.
- [ ] Add uncertainty handling and source links for every answer.
- [ ] Prepare a two-minute live demo.

## Team working agreement

- Keep the main branch deployable.
- Make small, focused pull requests with a clear before/after note.
- Do not add features that cannot be demonstrated with evidence.
- Record every tested question, expected answer, actual answer, source sheet, and result.
- If the system is uncertain, surface that uncertainty. Trust is the product.

## Suggested roles

| Role | Focus |
| --- | --- |
| Product/demo | User flow, question selection, pitch, demo script |
| Document AI | PDF rendering, OCR, drawing metadata, coordinate extraction |
| AI/retrieval | Retrieval pipeline, Gemini prompts, answer/evidence validation |
| Frontend | Viewer, highlighted regions, question/answer UI |
| Platform | Database, deploy, CI, dataset/model storage |

One person can cover more than one role; this is a focus map, not a rigid team structure.

## Success criteria

The demo succeeds when a judge can ask a real question and, in under 10 seconds:

1. receive a useful answer;
2. see the exact supporting region of the original drawing;
3. understand why the system is confident—or why it refuses to answer.

## Prize entries

Main entry: **Stoa — Make Blueprints Talk**.

If the tools are genuinely part of the working solution, we can also enter Best Use of Gemini, Modal or Verda, Cloudinary, Codex, CRACI, and Aiven/MariaDB.

## Local setup

The repository is currently a project kickoff. Setup commands will be added once the frontend/backend structure is created.

## Useful links

- [Junction 2026](https://hackjunction.com/)

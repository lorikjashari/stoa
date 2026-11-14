"""Can an open-vocabulary detector find beds on drawings it was never trained on? (feasibility probe, Modal GPU)

Runs OWLv2 and Grounding DINO with text prompts on the images in /vol/probe of the stoa-cubicasa volume and writes
each image with its boxes drawn to eval_out/scratch/zeroshot/ (plus the raw boxes as JSON).

    modal run modal_jobs/zero_shot_probe.py
"""
from __future__ import annotations

import io
import json
from pathlib import Path

import modal

REPO = Path(__file__).resolve().parent.parent
vol = modal.Volume.from_name("stoa-cubicasa")
app = modal.App("stoa-zero-shot-probe")
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("torch", "torchvision", "transformers>=4.44", "pillow", "scipy", "timm"))
PROMPTS = ["a bed", "a double bed", "a single bed"]


@app.function(image=image, volumes={"/vol": vol}, gpu="L40S", timeout=1800)
def run() -> dict:
    import torch
    from PIL import Image, ImageDraw
    from transformers import (AutoModelForZeroShotObjectDetection, AutoProcessor, Owlv2ForObjectDetection,
                              Owlv2Processor)

    out = {}
    owl_p = Owlv2Processor.from_pretrained("google/owlv2-large-patch14-ensemble")
    owl = Owlv2ForObjectDetection.from_pretrained("google/owlv2-large-patch14-ensemble").cuda().eval()
    gd_p = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-base")
    gd = AutoModelForZeroShotObjectDetection.from_pretrained("IDEA-Research/grounding-dino-base").cuda().eval()
    for f in sorted(Path("/vol/probe").glob("*.png")):
        im = Image.open(f).convert("RGB")
        res = {}
        with torch.no_grad():
            inp = owl_p(text=[PROMPTS], images=im, return_tensors="pt").to("cuda")
            o = owl(**inp)
            side = max(im.size)  # OWLv2 pads to a square
            pp = getattr(owl_p, "post_process_object_detection", None) or owl_p.post_process_grounded_object_detection
            r = pp(outputs=o, threshold=0.1, target_sizes=torch.tensor([[side, side]]).cuda())[0]
            res["owlv2"] = [[PROMPTS[int(l)], round(float(s), 3), *[round(float(v), 1) for v in b]]
                            for s, l, b in zip(r["scores"], r["labels"], r["boxes"])]
            inp = gd_p(images=im, text="bed.", return_tensors="pt").to("cuda")
            o = gd(**inp)
            r = gd_p.post_process_grounded_object_detection(o, inp.input_ids, threshold=0.25, text_threshold=0.25,
                                                            target_sizes=[im.size[::-1]])[0]
            res["gdino"] = [["bed", round(float(s), 3), *[round(float(v), 1) for v in b]] for s, b in zip(r["scores"], r["boxes"])]
        imgs = {}
        for name, boxes in res.items():
            c = im.copy()
            d = ImageDraw.Draw(c)
            for lab, s, *b in boxes:
                d.rectangle(b, outline="red", width=3)
                d.text((b[0] + 2, b[1] + 2), f"{s:.2f}", fill="red")
            buf = io.BytesIO()
            c.save(buf, format="PNG")
            imgs[name] = buf.getvalue()
        out[f.stem] = {"boxes": res, "images": imgs}
    return out


@app.local_entrypoint()
def main():
    res = run.remote()
    d = REPO / "eval_out/scratch/zeroshot"
    d.mkdir(parents=True, exist_ok=True)
    for stem, r in res.items():
        for name, png in r["images"].items():
            (d / f"{stem}.{name}.png").write_bytes(png)
        (d / f"{stem}.json").write_text(json.dumps(r["boxes"]))
        print(stem, {k: len([b for b in v if b[1] >= 0.2]) for k, v in r["boxes"].items()})

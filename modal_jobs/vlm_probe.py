"""Can a grounding vision-language model (Qwen2.5-VL) box every bed on a floor plan? (feasibility probe, Modal GPU)

Images: /vol/probe on the stoa-cubicasa volume. Output: eval_out/scratch/vlm/<image>.<model>.png with the boxes drawn,
plus the raw replies as JSON.

    modal run modal_jobs/vlm_probe.py --model Qwen/Qwen2.5-VL-7B-Instruct
"""
from __future__ import annotations

import io
import json
import re
from pathlib import Path

import modal

REPO = Path(__file__).resolve().parent.parent
vol = modal.Volume.from_name("stoa-cubicasa")
hf = modal.Volume.from_name("stoa-hf-cache", create_if_missing=True)
app = modal.App("stoa-vlm-probe")
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("torch", "torchvision", "transformers>=4.51", "accelerate", "pillow", "qwen-vl-utils")
         .env({"HF_HOME": "/hf"}))
PROMPT = ("This is an architectural floor plan. Locate every bed drawn on it (single, double or bunk beds; not sofas, "
          "tables or desks). Output a JSON list, one entry per bed: {\"bbox_2d\": [x1, y1, x2, y2], \"label\": \"bed\"}.")


@app.function(image=image, volumes={"/vol": vol, "/hf": hf}, gpu="H100", timeout=3600)
def run(model_id: str) -> dict:
    import torch
    from PIL import Image, ImageDraw
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="cuda")
    proc = AutoProcessor.from_pretrained(model_id, max_pixels=1600 * 1600)
    hf.commit()
    out = {}
    for f in sorted(Path("/vol/probe").glob("*.png")):
        im = Image.open(f).convert("RGB")
        msgs = [{"role": "user", "content": [{"type": "image", "image": im}, {"type": "text", "text": PROMPT}]}]
        text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inp = proc(text=[text], images=[im], return_tensors="pt").to("cuda")
        with torch.no_grad():
            gen = model.generate(**inp, max_new_tokens=2048, do_sample=False)
        reply = proc.batch_decode(gen[:, inp.input_ids.shape[1]:], skip_special_tokens=True)[0]
        # Qwen2.5-VL answers in the pixels of the image it was fed (resized to a multiple of 28): map back
        gh, gw = inp["image_grid_thw"][0][1].item() * 14, inp["image_grid_thw"][0][2].item() * 14
        sx, sy = im.width / gw, im.height / gh
        boxes = []
        for m in re.finditer(r"\[\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*\]", reply):
            x1, y1, x2, y2 = map(float, m.groups())
            boxes.append([x1 * sx, y1 * sy, x2 * sx, y2 * sy])
        c = im.copy()
        d = ImageDraw.Draw(c)
        for b in boxes:
            d.rectangle(b, outline="red", width=3)
        buf = io.BytesIO()
        c.save(buf, format="PNG")
        out[f.stem] = {"reply": reply, "boxes": boxes, "png": buf.getvalue()}
    return out


@app.local_entrypoint()
def main(model: str = "Qwen/Qwen2.5-VL-7B-Instruct"):
    res = run.remote(model)
    d = REPO / "eval_out/scratch/vlm"
    d.mkdir(parents=True, exist_ok=True)
    tag = model.split("/")[-1]
    for stem, r in res.items():
        (d / f"{stem}.{tag}.png").write_bytes(r["png"])
        (d / f"{stem}.{tag}.json").write_text(json.dumps({"reply": r["reply"], "boxes": r["boxes"]}))
        print(stem, tag, len(r["boxes"]), "beds")

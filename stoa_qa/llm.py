"""Small LLM client: Gemma 4 through Featherless (OpenAI-compatible), JSON in / JSON out, with cost accounting.

The LLM never produces numbers that end up in an answer: it only plans a query over extracted facts
(stoa_qa.planner) or reads an image when no structured data exists (stoa_qa.vision).
"""
from __future__ import annotations

import base64
import io
import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from dotenv import dotenv_values

REPO = Path(__file__).resolve().parent.parent
MODEL = os.environ.get("STOA_LLM_MODEL", "google/gemma-4-31B-it")
URL = os.environ.get("STOA_LLM_URL", "https://api.featherless.ai/v1/chat/completions")


@dataclass
class Usage:
    model: str = ""
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    seconds: float = 0.0

    def to_dict(self) -> dict:
        return {"model": self.model, "tokens_in": self.tokens_in, "tokens_out": self.tokens_out,
                "cost_usd": round(self.cost_usd, 6), "seconds": round(self.seconds, 2)}


class LLMError(RuntimeError):
    pass


def api_key() -> str | None:
    if os.environ.get("STOA_NO_LLM"):  # geometry-only mode (also used by the unit tests)
        return None
    return os.environ.get("FEATHERLESS_API_KEY") or dotenv_values(REPO / ".env").get("FEATHERLESS_API_KEY")


def available() -> bool:
    return bool(api_key())


def image_part(path: Path, max_side: int = 1400) -> dict:
    from PIL import Image

    im = Image.open(path).convert("RGB")
    im.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=88)
    return {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()}}


def extract_json(text: str):
    """First JSON object in a model reply (tolerates code fences and prose around it)."""
    start = text.find("{")
    if start < 0:
        raise LLMError(f"no JSON in reply: {text[:160]!r}")
    depth, in_str, esc = 0, False, False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            esc = (ch == "\\") and not esc
            if ch == '"' and not esc:
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start: i + 1])
    raise LLMError(f"unterminated JSON in reply: {text[:160]!r}")


def chat_json(system: str, user: str, image: Path | None = None, max_tokens: int = 700, timeout: int = 120) -> tuple[dict, Usage]:
    key = api_key()
    if not key:
        raise LLMError("FEATHERLESS_API_KEY is not set")
    content: list | str = user
    if image is not None:
        content = [image_part(image), {"type": "text", "text": user}]
    body = {"model": MODEL, "max_tokens": max_tokens, "temperature": 0,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": content}]}
    t0 = time.time()
    last = ""
    for attempt in range(3):
        req = urllib.request.Request(URL, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                resp = json.loads(r.read())
            text = resp["choices"][0]["message"]["content"]
            u = resp.get("usage", {})
            usage = Usage(MODEL, u.get("prompt_tokens", 0), u.get("completion_tokens", 0), u.get("cost") or 0.0, time.time() - t0)
            return extract_json(text), usage
        except urllib.error.HTTPError as e:
            last = f"HTTP {e.code}: {e.read()[:160]!r}"
            if e.code not in (429, 500, 502, 503):
                break
            time.sleep(3 * (attempt + 1))
        except (LLMError, KeyError, json.JSONDecodeError) as e:
            last = f"bad reply: {e}"
            if "unterminated" in str(e):  # the reply was cut off at max_tokens (long row lists): give it more room
                body["max_tokens"] = min(int(body["max_tokens"] * 2.5), 4000)
        except Exception as e:  # network
            last = repr(e)[:160]
            time.sleep(2)
    raise LLMError(last)

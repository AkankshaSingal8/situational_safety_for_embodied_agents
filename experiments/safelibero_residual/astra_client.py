"""GPT-6 Astra (OpenAI Chat Completions) client for the Astra arms (ASTRA.md).

One call per chunk: two images + a text block in, a JSON array out (structured output, strict schema).
Every call is appended to <log_dir>/part_<pid>.jsonl: request id, model, latency, tokens, raw output, error.
The key is read from OPENAI_API_KEY (source ~/.envfile).
"""
import base64
import json
import os
import random
import time

import cv2
import numpy as np
import requests

URL = "https://api.openai.com/v1/chat/completions"
MODEL = os.environ.get("ASTRA_MODEL", "gpt-6-astra")


def png_b64(img):
    ok, buf = cv2.imencode(".png", cv2.cvtColor(np.ascontiguousarray(img), cv2.COLOR_RGB2BGR))
    assert ok
    return base64.b64encode(buf.tobytes()).decode()


def matrix_schema(name, rows, cols):
    """{"<name>": rows x cols numbers}, strict."""
    return {"type": "json_schema", "json_schema": {"name": f"astra_{name}", "strict": True, "schema": {
        "type": "object", "additionalProperties": False, "required": [name],
        "properties": {name: {"type": "array", "minItems": rows, "maxItems": rows,
                              "items": {"type": "array", "minItems": cols, "maxItems": cols,
                                        "items": {"type": "number"}}}}}}}


class AstraClient:
    def __init__(self, log_dir, effort=None, detail="auto", retries=4, timeout=120, max_tokens=None):
        self.key = os.environ["OPENAI_API_KEY"]
        self.effort, self.detail, self.retries, self.timeout = effort, detail, retries, timeout
        self.max_tokens = max_tokens
        os.makedirs(log_dir, exist_ok=True)
        self.log = open(os.path.join(log_dir, f"part_{os.getpid()}.jsonl"), "a")
        self.s = requests.Session()
        self.tag = {}  # episode identifiers written into each log line (set by the controller)

    def call(self, system, text, images, name, rows, cols):
        """Returns a (rows, cols) float array, or None after all retries fail."""
        content = [{"type": "text", "text": text}] + [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{png_b64(im)}", "detail": self.detail}}
            for im in images]
        body = {"model": MODEL, "messages": [{"role": "system", "content": system}, {"role": "user", "content": content}],
                "response_format": matrix_schema(name, rows, cols)}
        if self.effort:
            body["reasoning_effort"] = self.effort
        if self.max_tokens:
            body["max_completion_tokens"] = self.max_tokens
        for attempt in range(self.retries + 1):
            t0, rec = time.time(), dict(self.tag, attempt=attempt)
            try:
                r = self.s.post(URL, headers={"Authorization": f"Bearer {self.key}"}, json=body, timeout=self.timeout)
                rec.update(status=r.status_code, latency=round(time.time() - t0, 3))
                d = r.json()
                if r.status_code != 200:
                    rec["error"] = str(d.get("error", d))[:400]
                    self._write(rec)
                    if r.status_code in (400, 401, 403, 404):
                        return None  # not retryable
                    time.sleep(min(60, 2 ** attempt + random.random() * 2))
                    continue
                u = d.get("usage", {})
                msg = d["choices"][0]["message"]
                rec.update(id=d.get("id"), model=d.get("model"), finish=d["choices"][0].get("finish_reason"),
                           prompt_tokens=u.get("prompt_tokens"), completion_tokens=u.get("completion_tokens"),
                           reasoning_tokens=(u.get("completion_tokens_details") or {}).get("reasoning_tokens"),
                           cached_tokens=(u.get("prompt_tokens_details") or {}).get("cached_tokens"),
                           raw=msg.get("content"), refusal=msg.get("refusal"))
                a = np.asarray(json.loads(msg["content"])[name], np.float64)
                if a.shape != (rows, cols) or not np.all(np.isfinite(a)):
                    raise ValueError(f"bad output shape/values {a.shape}")
                self._write(rec)
                return a.astype(np.float32)
            except Exception as ex:  # network, JSON, shape: retry
                rec.setdefault("latency", round(time.time() - t0, 3))
                rec["error"] = repr(ex)[:400]
                self._write(rec)
                time.sleep(min(60, 2 ** attempt + random.random() * 2))
        return None

    def _write(self, rec):
        self.log.write(json.dumps(rec) + "\n")
        self.log.flush()


FABLE_URL = "https://api.anthropic.com/v1/messages"
FABLE_MODEL = os.environ.get("FABLE_MODEL", "claude-fable-5-2")


class FableClient(AstraClient):
    """Same interface as AstraClient, Claude Messages API. Thinking is always on for Fable 5.2 (no `thinking`
    field); depth is set by output_config.effort. No server-side fallback, so every answer comes from Fable;
    a refusal counts as a failed call."""

    def __init__(self, log_dir, effort=None, retries=4, timeout=300, max_tokens=16000, **_):
        self.key = os.environ["ANTHROPIC_API_KEY"]
        self.effort, self.retries, self.timeout, self.max_tokens = effort, retries, timeout, max_tokens
        os.makedirs(log_dir, exist_ok=True)
        self.log = open(os.path.join(log_dir, f"part_{os.getpid()}.jsonl"), "a")
        self.s = requests.Session()
        self.tag = {}

    def call(self, system, text, images, name, rows, cols):
        content = [{"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": png_b64(im)}}
                   for im in images] + [{"type": "text", "text": text}]
        # Claude's structured outputs reject minItems/maxItems > 1, so the shape is stated in the prompt and
        # checked below (a wrong shape is a failed attempt and is retried).
        schema = {"type": "object", "additionalProperties": False, "required": [name],
                  "properties": {name: {"type": "array", "items": {"type": "array", "items": {"type": "number"}}}}}
        text = text + f"\nReturn \"{name}\" as exactly {rows} rows of {cols} numbers."
        oc = {"format": {"type": "json_schema", "schema": schema}}
        if self.effort:
            oc["effort"] = self.effort
        body = {"model": FABLE_MODEL, "max_tokens": self.max_tokens, "system": system, "output_config": oc,
                "messages": [{"role": "user", "content": content}]}
        hdr = {"x-api-key": self.key, "anthropic-version": "2023-06-01", "content-type": "application/json"}
        for attempt in range(self.retries + 1):
            t0, rec = time.time(), dict(self.tag, attempt=attempt)
            try:
                r = self.s.post(FABLE_URL, headers=hdr, json=body, timeout=self.timeout)
                rec.update(status=r.status_code, latency=round(time.time() - t0, 3))
                d = r.json()
                if r.status_code != 200:
                    rec["error"] = str(d.get("error", d))[:400]
                    self._write(rec)
                    if r.status_code in (400, 401, 403, 404):
                        return None
                    time.sleep(min(60, 2 ** attempt + random.random() * 2))
                    continue
                u = d.get("usage", {})
                txt = "".join(b.get("text", "") for b in d.get("content", []) if b.get("type") == "text")
                rec.update(id=d.get("id"), model=d.get("model"), finish=d.get("stop_reason"),
                           prompt_tokens=u.get("input_tokens"), completion_tokens=u.get("output_tokens"),
                           reasoning_tokens=(u.get("output_tokens_details") or {}).get("thinking_tokens"),
                           cached_tokens=u.get("cache_read_input_tokens"), raw=txt)
                if d.get("stop_reason") == "refusal":
                    rec["error"] = "refusal"
                    self._write(rec)
                    return None
                a = np.asarray(json.loads(txt)[name], np.float64)
                if a.shape != (rows, cols) or not np.all(np.isfinite(a)):
                    raise ValueError(f"bad output shape/values {a.shape}")
                self._write(rec)
                return a.astype(np.float32)
            except Exception as ex:
                rec.setdefault("latency", round(time.time() - t0, 3))
                rec["error"] = repr(ex)[:400]
                self._write(rec)
                time.sleep(min(60, 2 ** attempt + random.random() * 2))
        return None

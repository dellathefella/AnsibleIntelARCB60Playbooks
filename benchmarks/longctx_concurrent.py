#!/usr/bin/env python3
"""Concurrent long-context session probe.

Fires N simultaneous requests with a large nonce'd prompt and reports per-request
TTFT, wall time, and aggregate prefill throughput. Used to validate that the KV
pool supports multiple ~262K sessions.

Usage:
  python3 longctx_concurrent.py --base http://127.0.0.1:8081 --model qwen3.8-27b \
      --n 3 --units 1083 --max-tokens 16
"""
import argparse, json, time, urllib.request, uuid, concurrent.futures

_UNIT = """class Processor_{i}:
    def handle_{i}(self, items, *, strict=False, retries=3):
        acc = {{}}
        for idx, it in enumerate(items):
            key = getattr(it, 'id', idx)
            try:
                val = self.transform_{i}(it)
            except ValueError as exc:
                if strict:
                    raise RuntimeError(f'bad item {{key}}: {{exc}}') from exc
                continue
            if key in acc and retries > 0:
                acc[key] = self.merge_{i}(acc[key], val)
            else:
                acc[key] = val
        return self.finalize_{i}(acc)

    def transform_{i}(self, it):
        return [x * (i + 1) for x in getattr(it, 'values', []) if x is not None]

    def merge_{i}(self, a, b):
        return sorted(set(list(a) + list(b)))

    def finalize_{i}(self, acc):
        return {{k: tuple(v) for k, v in acc.items() if v}}
"""


def make_prompt(n_units):
    body = "\n".join(_UNIT.format(i=i) for i in range(n_units))
    return ("Review this Python module for correctness, duplication, and performance. "
            "Identify the top 5 issues and propose a refactor that removes the per-index "
            "boilerplate. Be concrete.\n\n```python\n" + body + "\n```")


def stream(base, model, prompt, max_tokens, timeout=3600):
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    data = json.dumps(body).encode()
    req = urllib.request.Request(base + "/v1/chat/completions", data=data,
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    ttft = None
    ptok = ctok = None
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload == "[DONE]":
                break
            obj = json.loads(payload)
            if ttft is None and obj.get("choices"):
                ttft = time.perf_counter() - t0
            usage = obj.get("usage")
            if usage:
                ptok = usage.get("prompt_tokens")
                ctok = usage.get("completion_tokens")
    return ttft, time.perf_counter() - t0, ptok, ctok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8081")
    ap.add_argument("--model", default="qwen3.8-27b")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--units", type=int, default=1083)
    ap.add_argument("--max-tokens", type=int, default=16)
    a = ap.parse_args()
    base = a.base.rstrip("/")

    prompts = [f"[{uuid.uuid4().hex[:12]}] " + make_prompt(a.units) for _ in range(a.n)]
    t0 = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.n) as ex:
        futs = [ex.submit(stream, base, a.model, p, a.max_tokens) for p in prompts]
        results = [f.result() for f in futs]
    wall = time.perf_counter() - t0

    ptoks = [r[2] or 0 for r in results]
    ttfts = [r[0] or 0 for r in results]
    total_prompt = sum(ptoks)
    print(json.dumps({
        "n": a.n,
        "prompt_tokens_each": ptoks,
        "ttft_s": [round(t, 2) for t in ttfts],
        "wall_s": round(wall, 2),
        "aggregate_prefill_tok_s": round(total_prompt / wall, 1) if wall else 0,
        "per_request_prefill_tok_s": [round(p / t, 1) if t else 0 for p, t in zip(ptoks, ttfts)],
        "completion_tokens": [r[3] for r in results],
    }, indent=1), flush=True)


if __name__ == "__main__":
    main()

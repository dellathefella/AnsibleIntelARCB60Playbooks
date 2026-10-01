#!/usr/bin/env python3
"""Benchmark prefill (prompt processing) vs decode (token generation) tok/s
for coding tasks across OpenAI-compatible endpoints.

Methodology (per prompt, per endpoint):
  A: max_tokens=1  -> cold prefill + 1 decode token
  B: max_tokens=K  -> warm (cached) prefill + K decode tokens
  decode_tps = K / tB                      (prefill ~cached, so tB ~ pure decode)
  per_tok    = tB / K
  prefill_tps = prompt_tokens / (tA - per_tok)   (strip the 1 decode token from A)
"""
import json, time, urllib.request, sys, uuid

K = 256

def post(url, payload, timeout=120):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = json.loads(r.read())
    return time.perf_counter() - t0, body

def model_id(base):
    try:
        with urllib.request.urlopen(base + "/models", timeout=10) as r:
            return json.loads(r.read())["data"][0]["id"]
    except Exception:
        return "default"

ENDPOINTS = {
    "B60-local  Qwen3.8-27B": "http://127.0.0.1:8183/v1",
    "Halogen    Flash-Next   ": "http://10.0.1.74:8731/v1",
}

CODE_REVIEW = """You are a senior Python engineer. Review the function below for bugs, edge cases and performance, then give a corrected version with a short explanation of each change.

```python
def merge(intervals):
    out = []
    for s, e in intervals:
        if not out or s > out[-1][1]:
            out.append([s, e])
        elif e > out[-1][1]:
            out[-1][1] = e
    return out
```

Address: unsorted input, touching intervals like [1,4][4,5], empty input, nested intervals, and integer overflow. Return the fixed code and bullet-point the reasoning."""

CODE_GEN = ("Write a production-quality Python function `debounce(func, wait)` for asyncio that delays "
            "calling `func` until `wait` seconds pass without another call. Include full type hints, a "
            "docstring, cancellation of the pending task, and a short usage example with an async callback.")

# Long-context coding prompt: a realistic module to review, to push prefill into steady state.
_LONG_MODULE = "\n".join(
    f"""class Processor_{i}:
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
    for i in range(12)
)
LONG_REVIEW = ("Review this Python module for correctness, duplication, and performance. "
               "Identify the top 5 issues and propose a refactor that removes the per-index "
               "boilerplate. Be concrete.\n\n```python\n" + _LONG_MODULE + "\n```")

def bench(base, prompt, reps=3):
    mid = model_id(base)
    # Unique nonce => cold prefill every run; A and B share it so B's prefill is warm.
    prompt = f"[{uuid.uuid4().hex[:12]}] " + prompt
    msgs = [{"role": "user", "content": prompt}]
    try:
        post(base + "/chat/completions", {"model": mid, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 1, "temperature": 0})
    except Exception:
        pass
    # Cold prefill: first time this prompt is seen.
    tA, bA = post(base + "/chat/completions", {"model": mid, "messages": msgs, "max_tokens": 1, "temperature": 0})
    p = bA.get("usage", {}).get("prompt_tokens", 0)
    # Warm decode: prefill cached, average over reps.
    decodes, c, tB = [], 0, 0.0
    for _ in range(reps):
        tB, bB = post(base + "/chat/completions", {"model": mid, "messages": msgs, "max_tokens": K + 1, "temperature": 0})
        c = bB.get("usage", {}).get("completion_tokens", 0)
        if tB > 0:
            decodes.append(c / tB)
    per_tok = tB / max(c, 1)
    decode_tps = sum(decodes) / len(decodes) if decodes else 0.0
    prefill_time = tA - per_tok
    prefill_tps = (p / prefill_time) if prefill_time > 0 else float("nan")
    return p, prefill_tps, c, decode_tps

def main():
    print(f"{'endpoint':28s} {'task':11s} {'prompt_tok':>10s} {'prefill t/s':>12s} {'gen_tok':>8s} {'decode t/s':>11s}")
    print("-" * 84)
    for name, base in ENDPOINTS.items():
        for tname, pr in (("code_review", CODE_REVIEW), ("code_gen", CODE_GEN), ("long_review", LONG_REVIEW)):
            try:
                p, pf, c, dc = bench(base, pr)
                print(f"{name:28s} {tname:11s} {p:10d} {pf:12.1f} {c:8d} {dc:11.1f}")
            except Exception as e:
                print(f"{name:28s} {tname:11s} ERROR: {e}")

if __name__ == "__main__":
    main()

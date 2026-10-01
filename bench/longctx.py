#!/usr/bin/env python3
"""Long-context benchmark: cold prefill + warm decode at 16k..128k prompt tokens.

Same methodology as bench.py (nonce -> cold prefill with max_tokens=1, then warm
cached-prefill decode of K tokens averaged over reps), but with prompt sizes that
stress the KV cache and attention path. Sizes are targeted via repeated module
blocks; the ACTUAL token count is reported from the server's usage field.

Run on the host serving the B60 model (halo1): python3 longctx.py
Halogen must be reachable from the same network.
"""
import json, sys, time, urllib.request, uuid

K = 256
REPS = 3
PREFILL_TIMEOUT = 900   # 128k cold prefill can take minutes
DECODE_TIMEOUT = 300

ENDPOINTS = {
    "B60-sycl   Qwen3.8-27B": "http://127.0.0.1:8183/v1",
    "Halogen    Flash-Next   ": "http://10.0.1.74:8731/v1",
}

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

# ~246 tok/unit (calibrated from long_review: 12 units ~ 2950 tok)
SIZES = [("16k", 65), ("32k", 130), ("64k", 260), ("128k", 520)]

def post(url, payload, timeout):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = json.loads(r.read())
    return time.perf_counter() - t0, body

def bench_long(base, prompt):
    prompt = f"[{uuid.uuid4().hex[:12]}] " + prompt
    msgs = [{"role": "user", "content": prompt}]
    tA, bA = post(base + "/chat/completions",
                  {"messages": msgs, "max_tokens": 1, "temperature": 0}, PREFILL_TIMEOUT)
    p = bA.get("usage", {}).get("prompt_tokens", 0)
    decodes, c, tB = [], 0, 0.0
    for _ in range(REPS):
        tB, bB = post(base + "/chat/completions",
                      {"messages": msgs, "max_tokens": K + 1, "temperature": 0}, DECODE_TIMEOUT)
        c = bB.get("usage", {}).get("completion_tokens", 0)
        if tB > 0:
            decodes.append(c / tB)
    per_tok = tB / max(c, 1)
    decode_tps = sum(decodes) / len(decodes) if decodes else 0.0
    prefill_time = tA - per_tok
    prefill_tps = (p / prefill_time) if prefill_time > 0 else float("nan")
    return p, prefill_tps, c, decode_tps

def main():
    print(f"{'endpoint':26s} {'target':6s} {'prompt_tok':>10s} {'prefill t/s':>12s} {'gen_tok':>8s} {'decode t/s':>11s}")
    print("-" * 78)
    for name, base in ENDPOINTS.items():
        for label, units in SIZES:
            try:
                p, pf, c, dc = bench_long(base, make_prompt(units))
                print(f"{name:26s} {label:6s} {p:10d} {pf:12.1f} {c:8d} {dc:11.1f}", flush=True)
            except Exception as e:
                print(f"{name:26s} {label:6s} ERROR: {e}", flush=True)

if __name__ == "__main__":
    main()

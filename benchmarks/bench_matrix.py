#!/usr/bin/env python3
"""Context-length matrix benchmark: prefill + decode t/s at 16k..128k, with
sustained-clock warmup for a fair single-GPU A/B.

Client BMG has no clock lock, so every measurement is preceded by a sustained
load to reach steady-state clocks (the +/-30% cold-ramp noise in RESULTS.md).
Run this against one endpoint at a time; flip containers between runs and keep
the GPU warm across the flip.

Reports, per context size:
  prefill t/s  = prompt_tokens / cold-prefill-time (max_tokens=1)
  decode  t/s  = K tokens on warm (cached) prefill, median over reps

Usage:
  python3 bench_matrix.py --endpoint http://10.0.1.67:8183 --label stock
  python3 bench_matrix.py --endpoint http://10.0.1.67:8186 --label tuned --sizes 32k,64k,128k
"""
import argparse, json, statistics, sys, time, urllib.request, uuid

K = 256
PREFILL_TIMEOUT = 1200
DECODE_TIMEOUT = 360

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
SIZES = {"16k": 65, "32k": 130, "64k": 260, "128k": 520}


def post(url, payload, timeout):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = json.loads(r.read())
    return time.perf_counter() - t0, body


def model_id(base):
    with urllib.request.urlopen(base + "/models", timeout=10) as r:
        return json.loads(r.read())["data"][0]["id"]


def bench_size(base, mid, prompt, reps, prefill_reps=3):
    # Cold prefill: each rep uses a unique nonce so the prompt is never cached.
    # Median over prefill_reps to tame the single-sample noise.
    prefills, p = [], 0
    for _ in range(prefill_reps):
        nonce = uuid.uuid4().hex[:12]
        msgs = [{"role": "user", "content": f"[{nonce}] " + prompt}]
        tA, bA = post(base + "/chat/completions",
                      {"model": mid, "messages": msgs, "max_tokens": 1, "temperature": 0},
                      PREFILL_TIMEOUT)
        p = bA.get("usage", {}).get("prompt_tokens", 0)
        if tA > 0:
            prefills.append(p / tA)
    prefill_med = statistics.median(prefills) if prefills else float("nan")

    # Warm decode: one nonce, then K-token decode averaged over reps.
    msgs = [{"role": "user", "content": f"[{uuid.uuid4().hex[:12]}] " + prompt}]
    post(base + "/chat/completions",
         {"model": mid, "messages": msgs, "max_tokens": 1, "temperature": 0}, PREFILL_TIMEOUT)
    decodes, c, tB = [], 0, 0.0
    for _ in range(reps):
        tB, bB = post(base + "/chat/completions",
                      {"model": mid, "messages": msgs, "max_tokens": K + 1, "temperature": 0},
                      DECODE_TIMEOUT)
        c = bB.get("usage", {}).get("completion_tokens", 0)
        if tB > 0:
            decodes.append(c / tB)
    decode_med = statistics.median(decodes) if decodes else 0.0
    return p, prefill_med, decode_med


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", required=True)
    ap.add_argument("--label", default="")
    ap.add_argument("--sizes", default="16k,32k,64k,128k")
    ap.add_argument("--warm-seconds", type=float, default=120.0)
    ap.add_argument("--reps", type=int, default=5)
    args = ap.parse_args()

    base = args.endpoint.rstrip("/")
    try:
        urllib.request.urlopen(base + "/health", timeout=10).read()
    except Exception as e:
        sys.exit(f"endpoint not healthy: {base} ({e})")
    mid = model_id(base)
    sizes = [s.strip() for s in args.sizes.split(",")]

    print(f"[{args.label}] warming GPU {args.warm_seconds:.0f}s (sustained 16k decode)...")
    warm_prompt = make_prompt(SIZES["16k"])
    t_end = time.time() + args.warm_seconds
    while time.time() < t_end:
        bench_size(base, mid, warm_prompt, 1)

    print(f"\n[{args.label}] {'size':6s} {'prompt_tok':>10s} {'prefill t/s':>12s} {'decode t/s':>11s}")
    print("-" * 44)
    for label in sizes:
        try:
            p, pf, dc = bench_size(base, mid, make_prompt(SIZES[label]), args.reps)
            print(f"[{args.label}] {label:6s} {p:10d} {pf:12.1f} {dc:11.2f}", flush=True)
        except Exception as e:
            print(f"[{args.label}] {label:6s} ERROR: {e}", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Single-GPU A/B decode benchmark with sustained-clock warmup.

MEASUREMENT RIG (learned 2026-10-02, halo1):
  * Lock GPU clocks first: tasks/gpu-clock-lock.yml pins min_freq=max_freq
    (2400 MHz). Without it, prefill loses ~9% and decode sags between bursts.
  * Benchmark with speculative decoding DISABLED. With spec on, each rep's
    random-UUID prompt produces different content -> different MTP/ngram
    acceptance -> effective t/s swings 32-47 (MAD ~5). That variance, NOT
    clocks, was the old "+/-30% intra-session noise". Spec off + clocks
    locked gives MAD ~0.02 t/s (0.1%), enough to resolve 1-2% kernel deltas.
  * Raw spec-off decode (~21.5 t/s) x spec multiplier (1.5-2.2x, content
    dependent) = the effective production number.

Protocol (run per endpoint, back-to-back):
  1. warm: hammer decode continuously for --warm-seconds (default 180)
  2. measure: --reps timed K-token decodes, report median + MAD

Usage:
  python3 single_gpu_ab.py --endpoint http://10.0.1.67:8184 --label variant-a
"""
import argparse, json, statistics, sys, time, urllib.request, uuid

PROMPT = ("You are a senior Python engineer. Review the function below for bugs, "
          "edge cases and performance, then give a corrected version with a short "
          "explanation of each change.\n\n```python\n"
          "def merge(intervals):\n"
          "    out = []\n"
          "    for s, e in intervals:\n"
          "        if not out or s > out[-1][1]:\n"
          "            out.append([s, e])\n"
          "        elif e > out[-1][1]:\n"
          "            out[-1][1] = e\n"
          "    return out\n```\n"
          "Address: unsorted input, touching intervals, empty input, nested intervals.")


def post(url, payload, timeout=180):
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


def decode_once(base, mid, k):
    prompt = f"[{uuid.uuid4().hex[:12]}] " + PROMPT
    msgs = [{"role": "user", "content": prompt}]
    post(base + "/chat/completions", {"model": mid, "messages": msgs, "max_tokens": 1, "temperature": 0})
    t, b = post(base + "/chat/completions", {"model": mid, "messages": msgs, "max_tokens": k + 1, "temperature": 0})
    c = b.get("usage", {}).get("completion_tokens", 0)
    return c / t if t > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", required=True)
    ap.add_argument("--label", default="")
    ap.add_argument("--warm-seconds", type=float, default=180.0)
    ap.add_argument("--reps", type=int, default=9)
    ap.add_argument("--k", type=int, default=256)
    args = ap.parse_args()

    base = args.endpoint.rstrip("/")
    try:
        urllib.request.urlopen(base + "/health", timeout=10).read()
    except Exception as e:
        sys.exit(f"endpoint not healthy: {base} ({e})")
    mid = model_id(base)

    print(f"[{args.label}] warming GPU for {args.warm_seconds:.0f}s (sustained decode)...")
    t_end = time.time() + args.warm_seconds
    n = 0
    while time.time() < t_end:
        decode_once(base, mid, args.k)
        n += 1
    print(f"[{args.label}] warmup done ({n} sustained decodes). measuring {args.reps} reps...")

    vals = [decode_once(base, mid, args.k) for _ in range(args.reps)]
    med = statistics.median(vals)
    mad = statistics.median([abs(v - med) for v in vals])
    print(f"[{args.label}] decode t/s: {[round(v, 1) for v in vals]}")
    print(f"[{args.label}] median {med:.2f}  MAD {mad:.2f}  min {min(vals):.1f}  max {max(vals):.1f}")


if __name__ == "__main__":
    main()

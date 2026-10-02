#!/usr/bin/env python3
"""Interleaved A/B decode benchmark for two OpenAI-compatible endpoints.

Compensates for the +/-32% intra-session clock ramp documented in RESULTS.md:
  - warmup both endpoints first
  - strictly alternate A,B per rep (same thermal position for both)
  - high rep count, report median + MAD (robust to outliers)
  - paired per-rep diff (B - A) with sign count

Usage:
  python3 ab_bench.py --a http://10.0.1.67:8183 --b http://10.0.1.67:8186 \
      --reps 9 --k 256

For locked-clock rigor (preferred when available on the rig):
  sudo xpu-smi set -device 0 ...   # if supported, lock before running
otherwise rely on interleaving + median + sign test.
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


def decode_tps(base, k):
    """Warm-prefill a unique prompt, then time k-token decode."""
    mid = model_id(base)
    prompt = f"[{uuid.uuid4().hex[:12]}] " + PROMPT
    msgs = [{"role": "user", "content": prompt}]
    post(base + "/chat/completions", {"model": mid, "messages": msgs, "max_tokens": 1, "temperature": 0})
    t, b = post(base + "/chat/completions", {"model": mid, "messages": msgs, "max_tokens": k + 1, "temperature": 0})
    c = b.get("usage", {}).get("completion_tokens", 0)
    return c / t if t > 0 else 0.0


def stats(xs):
    med = statistics.median(xs)
    mad = statistics.median([abs(x - med) for x in xs])
    return med, mad, min(xs), max(xs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="baseline endpoint base URL (e.g. http://host:8183)")
    ap.add_argument("--b", required=True, help="tuned endpoint base URL (e.g. http://host:8186)")
    ap.add_argument("--reps", type=int, default=9)
    ap.add_argument("--k", type=int, default=256)
    args = ap.parse_args()

    a_base, b_base = args.a.rstrip("/"), args.b.rstrip("/")
    for base in (a_base, b_base):
        try:
            urllib.request.urlopen(base + "/health", timeout=10).read()
        except Exception as e:
            sys.exit(f"endpoint not healthy: {base} ({e})")

    print(f"warmup (2 rounds each)...")
    for _ in range(2):
        decode_tps(a_base, args.k)
        decode_tps(b_base, args.k)

    a_vals, b_vals = [], []
    print(f"\n{'rep':>4s} {'A t/s':>9s} {'B t/s':>9s} {'B-A':>8s} {'B/A':>7s}")
    for i in range(args.reps):
        a = decode_tps(a_base, args.k)
        b = decode_tps(b_base, args.k)
        a_vals.append(a)
        b_vals.append(b)
        print(f"{i+1:4d} {a:9.2f} {b:9.2f} {b-a:+8.2f} {b/a:7.3f}")

    am, amad, amin, amax = stats(a_vals)
    bm, bmad, bmin, bmax = stats(b_vals)
    diffs = [b - a for a, b in zip(a_vals, b_vals)]
    wins = sum(1 for d in diffs if d > 0)
    print("-" * 44)
    print(f"A  median {am:7.2f} t/s  MAD {amad:5.2f}  range [{amin:.1f}, {amax:.1f}]")
    print(f"B  median {bm:7.2f} t/s  MAD {bmad:5.2f}  range [{bmin:.1f}, {bmax:.1f}]")
    print(f"paired B-A median {statistics.median(diffs):+.2f} t/s  "
          f"({wins}/{args.reps} reps B>A, ratio {bm/am:.3f})")
    if wins >= 8 or wins <= 1:
        print("verdict: consistent effect (sign test p<0.02)")
    else:
        print("verdict: NOT consistent — more reps or locked clocks needed")


if __name__ == "__main__":
    main()

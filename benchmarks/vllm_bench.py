#!/usr/bin/env python3
"""Benchmark vLLM XPU vs llama.cpp SYCL on the Arc B60.

Measures single-stream decode/prefill AND concurrent throughput (the axis
where vLLM's continuous batching + paged KV should win).

Usage: python3 vllm_bench.py [base_url]
  default base_url = http://10.0.1.67:8185/v1
"""
import json, time, urllib.request, sys, concurrent.futures

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://10.0.1.67:8185/v1"
MODEL = "qwen3.8-27b"

LONG_PROMPT = (
    "You are a senior Python engineer. Below is a long module. Review every "
    "function for bugs, edge cases, and performance, then give a corrected "
    "version with a short explanation of each change.\n\n"
    + ("def helper_%d(x):\n    return x * %d + 1\n\n" % (0, 2) * 400)
)

def post(payload, timeout=300):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(BASE + "/chat/completions", data=data,
                                headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = json.loads(r.read())
    return time.perf_counter() - t0, body

def usage(body, kind):
    u = body.get("usage", {})
    return u.get("completion_tokens", 0), u.get("prompt_tokens", 0)

def single_stream():
    # A: cold prefill + 1 decode
    tA, bA = post({"model": MODEL, "messages": [{"role": "user", "content": LONG_PROMPT}],
                   "max_tokens": 1, "temperature": 0})
    _, pA = usage(bA, "p")
    # B: warm (cached) prefill + K decode
    K = 256
    tB, bB = post({"model": MODEL, "messages": [{"role": "user", "content": LONG_PROMPT}],
                   "max_tokens": K, "temperature": 0})
    cB, _ = usage(bB, "c")
    per_tok = tB / max(cB, 1)
    decode_tps = cB / tB
    prefill_tps = pA / max(tA - per_tok, 1e-6)
    print(f"[single] prompt_tokens={pA} completion_tokens={cB}")
    print(f"[single] decode: {decode_tps:.1f} tok/s  ({per_tok*1000:.1f} ms/tok)")
    print(f"[single] prefill: {prefill_tps:.1f} tok/s")
    return decode_tps, prefill_tps

def concurrency(n, max_tokens=128):
    def one(_):
        t, b = post({"model": MODEL,
                    "messages": [{"role": "user", "content": "Write a 100-word poem about the sea."}],
                    "max_tokens": max_tokens, "temperature": 0.7})
        c, _ = usage(b, "c")
        return t, c
    with concurrent.futures.ThreadPoolExecutor(max_workers=n) as ex:
        t0 = time.perf_counter()
        results = list(ex.map(one, range(n)))
        wall = time.perf_counter() - t0
    total_tok = sum(c for _, c in results)
    agg = total_tok / wall
    avg_lat = sum(t for t, _ in results) / n
    print(f"[conc n={n}] wall={wall:.1f}s total_tok={total_tok} "
          f"aggregate={agg:.1f} tok/s avg_latency={avg_lat:.1f}s")
    return agg

if __name__ == "__main__":
    print(f"=== vLLM XPU benchmark @ {BASE} ===")
    single_stream()
    for n in (1, 4, 8):
        concurrency(n)

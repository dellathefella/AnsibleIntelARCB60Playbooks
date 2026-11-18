#!/usr/bin/env python3
"""Concurrency benchmark for a single OpenAI-compatible endpoint.

Usage: python3 concurrency_bench.py --base http://127.0.0.1:8081 --model qwen3.8-27b --levels 1,4,8,16
"""
import argparse, json, time, urllib.request, concurrent.futures


def post(base, model, prompt, max_tokens=256, timeout=1800):
    url = base + "/v1/chat/completions"
    body = {"model": model, "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "temperature": 0.0}
    data = json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        resp = json.loads(r.read())
    dt = time.perf_counter() - t0
    usage = resp.get("usage", {})
    return dt, usage.get("completion_tokens", 0)


def run(base, model, n, max_tokens=256):
    def one(_):
        return post(base, model, "Write a 100-word poem about the sea.", max_tokens)

    t0 = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=n) as ex:
        res = list(ex.map(one, range(n)))
    wall = time.perf_counter() - t0
    total = sum(c for _, c in res)
    lat = [dt for dt, _ in res]
    return {"n": n, "wall_s": round(wall, 2), "total_tokens": total,
            "aggregate_tok_s": round(total / wall, 1),
            "avg_latency_s": round(sum(lat) / n, 2), "max_latency_s": round(max(lat), 2)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8081")
    ap.add_argument("--model", default="qwen3.8-27b")
    ap.add_argument("--levels", default="1,4,8,16")
    ap.add_argument("--max-tokens", type=int, default=256)
    a = ap.parse_args()
    for n in [int(x) for x in a.levels.split(",")]:
        print(json.dumps(run(a.base, a.model, n, a.max_tokens)), flush=True)

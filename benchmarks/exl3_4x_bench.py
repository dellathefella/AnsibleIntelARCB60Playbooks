#!/usr/bin/env python3
"""Benchmark four independent EXL3 instances (one per Arc B60).

Single-stream: per endpoint, TTFT + decode tok/s at 512 / 8K / 32K prompt depths.
Aggregate: one concurrent request per endpoint, total throughput.

Usage: python3 exl3_4x_bench.py --ports 8100-8103 --model qwen3.8-27b-exl3
"""
import argparse, json, time, urllib.request, concurrent.futures


def parse_ports(s):
    if "-" in s:
        a, b = s.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in s.split(",")]


def make_prompt(target_tokens):
    chars = int(target_tokens * 5.5)
    sentence = "The quick brown fox jumps over the lazy dog. "
    n = max(1, chars // len(sentence))
    return "Here is a long document:\n" + (sentence * n) + "\n\nWrite a short summary."


def stream_post(base, model, prompt, max_tokens=512, timeout=1800):
    url = base + "/v1/chat/completions"
    body = {"model": model, "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "temperature": 0.0, "stream": True}
    data = json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    first = None
    tokens = 0
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload == "[DONE]":
                break
            try:
                d = json.loads(payload)
            except Exception:
                continue
            choices = d.get("choices") or []
            if not choices:
                continue
            delta = choices[0].get("delta") or {}
            if delta.get("content") or delta.get("reasoning") or delta.get("reasoning_content"):
                if first is None:
                    first = time.perf_counter()
                tokens += 1
    end = time.perf_counter()
    if first is None:
        return None
    decode = (tokens - 1) / (end - first) if tokens > 1 else 0
    return {"ttft": first - t0, "decode_tok_s": decode, "tokens": tokens, "total_s": end - t0}


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
    return dt, usage.get("completion_tokens", 0), usage.get("prompt_tokens", 0)


def single_stream(ports, model, depths=(512, 8192, 32768), reps=2):
    out = {}
    for port in ports:
        base = f"http://127.0.0.1:{port}"
        per = {}
        for depth in depths:
            prompt = make_prompt(depth)
            vals = []
            for _ in range(reps):
                r = stream_post(base, model, prompt, 512)
                if r:
                    vals.append(r)
            if vals:
                per[depth] = {
                    "decode_tok_s_median": round(sorted(v["decode_tok_s"] for v in vals)[len(vals) // 2], 1),
                    "ttft_s_median": round(sorted(v["ttft"] for v in vals)[len(vals) // 2], 2),
                    "tokens": vals[0]["tokens"],
                }
            else:
                per[depth] = {"error": "no response"}
        out[port] = per
    return out


def aggregate(ports, model, max_tokens=256, reps=3):
    def one(port):
        base = f"http://127.0.0.1:{port}"
        prompt = "Write a 100-word poem about the sea."
        return post(base, model, prompt, max_tokens)

    vals = []
    for _ in range(reps):
        t0 = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(ports)) as ex:
            res = list(ex.map(one, ports))
        wall = time.perf_counter() - t0
        total = sum(c for _, c, _ in res)
        vals.append(total / wall)
    return {"aggregate_tok_s_median": round(sorted(vals)[len(vals) // 2], 1),
            "runs": [round(v, 1) for v in vals]}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ports", default="8100-8103")
    ap.add_argument("--model", default="qwen3.8-27b-exl3")
    a = ap.parse_args()
    ports = parse_ports(a.ports)
    print("=== single-stream ===")
    print(json.dumps(single_stream(ports, a.model), indent=1))
    print("=== aggregate (1 request per endpoint) ===")
    print(json.dumps(aggregate(ports, a.model), indent=1))

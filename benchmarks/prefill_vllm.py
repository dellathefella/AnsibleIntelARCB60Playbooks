#!/usr/bin/env python3
"""Prefill probe for vLLM endpoints (client-side TTFT).

vLLM does not log llama.cpp's "prompt eval time" line, so prefill throughput is
measured as prompt_tokens / TTFT on a cold, nonce'd prompt with max_tokens=1.
Streaming is used so TTFT is the arrival of the first content chunk, not the
end of the request.

Usage:
  python3 prefill_vllm.py --base http://127.0.0.1:8081 --model qwen3.8-27b \
      --sizes 8k,32k,64k,96k
"""
import argparse, json, time, urllib.request, uuid

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


# units -> approximate prompt tokens (one unit is ~120 tokens)
SIZES = {"8k": 68, "16k": 135, "32k": 270, "64k": 540, "96k": 810}


def stream_prefill(base, model, prompt, timeout=1800):
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 1,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    data = json.dumps(body).encode()
    req = urllib.request.Request(base + "/v1/chat/completions", data=data,
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    ttft = None
    prompt_tokens = None
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
            if usage and usage.get("prompt_tokens"):
                prompt_tokens = usage["prompt_tokens"]
    return ttft, prompt_tokens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8081")
    ap.add_argument("--model", default="qwen3.8-27b")
    ap.add_argument("--sizes", default="8k,32k,64k,96k")
    ap.add_argument("--label", default="")
    a = ap.parse_args()
    base = a.base.rstrip("/")
    print(f"[{a.label}] {'size':6s} {'prompt_tok':>10s} {'ttft_s':>8s} {'prefill t/s':>12s}")
    print("-" * 40)
    for label in [s.strip() for s in a.sizes.split(",")]:
        prompt = f"[{uuid.uuid4().hex[:12]}] " + make_prompt(SIZES[label])
        ttft, ptok = stream_prefill(base, a.model, prompt)
        if ttft is None:
            print(f"[{a.label}] {label:6s} (no first chunk)", flush=True)
            continue
        n = ptok or 0
        rate = n / ttft if ttft > 0 else 0
        print(f"[{a.label}] {label:6s} {n:10d} {ttft:8.2f} {rate:12.1f}", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Prefill probe via server-side timing.

Client-side HTTP timing of a cold prefill is too noisy (single sample, network
jitter). llama-server logs the authoritative rate:
    prompt eval time = X ms / N tokens (Y ms per token, Z tokens per second)
This script triggers one cold prefill (unique nonce, max_tokens=1) per size and
then reads the server's reported prompt-eval rate from the container log.

Run on the host (or anywhere the container log is reachable via `podman logs`).

Usage:
  python3 prefill_probe.py --container qwen38-27b-sycl --label stock-ub1024 --sizes 32k,64k,128k
"""
import argparse, json, re, subprocess, sys, time, urllib.request, uuid

PREFILL_TIMEOUT = 1800

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

SIZES = {"16k": 65, "32k": 130, "64k": 260, "128k": 520}
RATE_RE = re.compile(r"prompt eval time\s*=\s*[\d.]+\s*ms\s*/\s*(\d+)\s*tokens\s*\(\s*[\d.]+\s*ms per token,\s*([\d.]+)\s*tokens per second")


def post(url, payload, timeout):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def model_id(base):
    with urllib.request.urlopen(base + "/models", timeout=10) as r:
        return json.loads(r.read())["data"][0]["id"]


def latest_rate(container):
    r = subprocess.run(["sudo", "podman", "logs", "--tail", "40", container],
                      capture_output=True, text=True)
    out = (r.stdout or "") + "\n" + (r.stderr or "")
    rates = RATE_RE.findall(out)
    return rates[-1] if rates else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", default="http://10.0.1.67:8183")
    ap.add_argument("--container", required=True)
    ap.add_argument("--label", default="")
    ap.add_argument("--sizes", default="32k,64k,128k")
    args = ap.parse_args()

    base = args.endpoint.rstrip("/")
    mid = model_id(base)
    sizes = [s.strip() for s in args.sizes.split(",")]

    print(f"[{args.label}] {'size':6s} {'prompt_tok':>10s} {'server prefill t/s':>18s}")
    print("-" * 38)
    for label in sizes:
        msgs = [{"role": "user", "content": f"[{uuid.uuid4().hex[:12]}] " + make_prompt(SIZES[label])}]
        post(base + "/chat/completions",
             {"model": mid, "messages": msgs, "max_tokens": 1, "temperature": 0}, PREFILL_TIMEOUT)
        time.sleep(1)
        rate = latest_rate(args.container)
        if rate:
            print(f"[{args.label}] {label:6s} {int(rate[0]):10d} {float(rate[1]):18.1f}", flush=True)
        else:
            print(f"[{args.label}] {label:6s} (no rate in log)", flush=True)


if __name__ == "__main__":
    main()

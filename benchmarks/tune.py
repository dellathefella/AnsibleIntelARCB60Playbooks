#!/usr/bin/env python3
"""llama.cpp/Vulkan tuning experiments for the Arc B60 (no-ReBAR, TB4).

Each config launches a temporary rootful container with the current Q4_K model,
benches coding prompts (cold prefill + warm decode, nonce-protected), records
draft acceptance, then tears down. Run under sudo. Logs one table row per
(label, task). Designed to run in background; poll the log file.
"""
import json, os, subprocess, sys, time, urllib.request, uuid

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench import CODE_REVIEW, CODE_GEN, LONG_REVIEW  # noqa: E402

IMAGE = "ghcr.io/ggml-org/llama.cpp:server-vulkan"
RENDER = "/dev/dri/renderD129"
MODELS = "/home/jdella/.local/share/b60-llama/models"
MODEL = "/models/Qwen3.8-27B-GSQ-RCO-IQ3_S-Intel-Arc-Tuned-MTP-Q4_K.gguf"
BASE = "http://127.0.0.1:8183/v1"
HEALTH = "http://127.0.0.1:8183/health"
K = 256

def common(ctx="200000", b="1024", ub="1024", spec=None):
    spec = spec if spec is not None else [
        "--spec-type", "ngram-mod,draft-mtp",
        "--spec-ngram-mod-n-match", "24", "--spec-ngram-mod-n-min", "1",
        "--spec-ngram-mod-n-max", "3", "--spec-draft-n-max", "3",
        "--spec-draft-p-min", "0.5",
    ]
    return ["-m", MODEL, "--device", "Vulkan0", "-ngl", "99", "-fa", "on",
            "-c", ctx, "-b", b, "-ub", ub, "-t", "4", "-tb", "6",
            "-ctk", "q4_0", "-ctv", "q4_0", *spec,
            "--host", "0.0.0.0", "--port", "8183"]

MTP3 = ["--spec-type", "draft-mtp", "--spec-draft-n-max", "3", "--spec-draft-p-min", "0.5"]
MTP8 = ["--spec-type", "draft-mtp", "--spec-draft-n-max", "8", "--spec-draft-p-min", "0.0"]
NGMOD = ["--spec-type", "ngram-mod,draft-mtp",
         "--spec-ngram-mod-n-match", "24", "--spec-ngram-mod-n-min", "8",
         "--spec-ngram-mod-n-max", "32", "--spec-draft-n-max", "8",
         "--spec-draft-p-min", "0.5"]

def ngspec(n):
    return ["--spec-type", "ngram-mod,draft-mtp",
            "--spec-ngram-mod-n-match", "24", "--spec-ngram-mod-n-min", "1",
            "--spec-ngram-mod-n-max", str(n), "--spec-draft-n-max", str(n),
            "--spec-draft-p-min", "0.5"]

NOHV = ["GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM=1"]

# (label, env list, server args, prompts) — round 2: combine winners + longer drafts
CONFIGS = [
    ("np1_nohv",      NOHV, common() + ["--parallel", "1"],          ("code_review", "code_gen", "long_review")),
    ("np1_nohv_d4",   NOHV, common(spec=ngspec(4)) + ["--parallel", "1"], ("code_review", "code_gen", "long_review")),
    ("np1_nohv_d5",   NOHV, common(spec=ngspec(5)) + ["--parallel", "1"], ("code_review", "code_gen")),
    ("np1_nohv_d5_131k", NOHV, common(ctx="131072", spec=ngspec(5)) + ["--parallel", "1"], ("code_review", "code_gen")),
]

PROMPTS = {"code_review": CODE_REVIEW, "code_gen": CODE_GEN, "long_review": LONG_REVIEW}

def post(url, payload, timeout=240):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = json.loads(r.read())
    return time.perf_counter() - t0, body

def bench2(prompt, reps=3):
    prompt = f"[{uuid.uuid4().hex[:12]}] " + prompt
    msgs = [{"role": "user", "content": prompt}]
    tA, bA = post(BASE + "/chat/completions", {"messages": msgs, "max_tokens": 1, "temperature": 0})
    p = bA.get("usage", {}).get("prompt_tokens", 0)
    decodes, c, tB, tim = [], 0, 0.0, {}
    for _ in range(reps):
        tB, bB = post(BASE + "/chat/completions", {"messages": msgs, "max_tokens": K + 1, "temperature": 0})
        c = bB.get("usage", {}).get("completion_tokens", 0)
        tim = bB.get("timings", {})
        if tB > 0:
            decodes.append(c / tB)
    per_tok = tB / max(c, 1)
    decode_tps = sum(decodes) / len(decodes) if decodes else 0.0
    prefill_time = tA - per_tok
    prefill_tps = (p / prefill_time) if prefill_time > 0 else float("nan")
    dn = tim.get("draft_n", 0); da = tim.get("draft_n_accepted", 0)
    acc = f"{da}/{dn}" if dn else "-"
    return p, prefill_tps, c, decode_tps, acc

def wait_health(timeout=300):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen(HEALTH, timeout=5) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(2)
    return False

def run(label, envs, args, prompts):
    cmd = ["podman", "run", "-d", "--name", "bench-tune",
           "--device", RENDER, "--security-opt", "label=disable",
           "-v", f"{MODELS}:/models:ro", "-p", "127.0.0.1:8183:8183",
           "--entrypoint", "/app/llama-server"]
    for e in envs:
        cmd += ["-e", e]
    cmd += [IMAGE, *args]
    subprocess.run(cmd, check=True, capture_output=True)
    if not wait_health():
        print(f"{label:14s} LOAD_FAILED", flush=True)
        subprocess.run(["podman", "logs", "--tail", "8", "bench-tune"], capture_output=False)
        subprocess.run(["podman", "rm", "-f", "bench-tune"], capture_output=True)
        return
    for tname in prompts:
        try:
            p, pf, c, dc, acc = bench2(PROMPTS[tname])
            print(f"{label:14s} {tname:11s} prompt={p:5d} prefill={pf:7.1f} gen={c:4d} decode={dc:6.2f} draft_acc={acc}", flush=True)
        except Exception as e:
            print(f"{label:14s} {tname:11s} ERROR: {e}", flush=True)
    subprocess.run(["podman", "rm", "-f", "bench-tune"], capture_output=True)
    time.sleep(5)

def main():
    t0 = time.time()
    for cfg in CONFIGS:
        run(*cfg)
    print(f"TOTAL {time.time() - t0:.0f}s", flush=True)

if __name__ == "__main__":
    main()

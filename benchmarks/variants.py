#!/usr/bin/env python3
"""Sequential on-GPU comparison of Qwen3.8-27B variants on the B60 (Vulkan).

For each variant: stop llama-b60 -> launch a temporary Vulkan llama-server on
127.0.0.1:8183 with identical flags -> bench the 3 coding prompts -> tear down.
Restores llama-b60 at the end. Run under sudo (rootful podman).

Context is capped at 4096 for every variant (all prompts fit; keeps the largest
model's VRAM safe and the comparison uniform).
"""
import os, sys, time, subprocess, urllib.request
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench import bench, CODE_REVIEW, CODE_GEN, LONG_REVIEW  # noqa: E402

LOCAL = "http://127.0.0.1:8183/v1"
HEALTH = "http://127.0.0.1:8183/health"
IMAGE = "ghcr.io/ggml-org/llama.cpp:server-vulkan"
RENDER = "/dev/dri/renderD129"
MODELS = "/home/jdella/.local/share/b60-llama/models"
CTX = "32768"

MTP = ["--spec-type", "ngram-mod,draft-mtp",
       "--spec-ngram-mod-n-match", "24", "--spec-ngram-mod-n-min", "1",
       "--spec-ngram-mod-n-max", "3", "--spec-draft-n-max", "3",
       "--spec-draft-p-min", "0.5"]
NGRAM = ["--spec-type", "ngram-mod",
         "--spec-ngram-mod-n-match", "24", "--spec-ngram-mod-n-min", "1",
         "--spec-ngram-mod-n-max", "3"]

VARIANTS = [
    ("Q4_K MTP (current)", "Qwen3.8-27B-GSQ-RCO-IQ3_S-Intel-Arc-Tuned-MTP-Q4_K.gguf", MTP),
    ("Ridge Q4_K (ngram)", "Qwen3.8-27B-Ridge-Intel-Arc-Tuned-Q4_K.gguf", NGRAM),
]

def sh(cmd):
    subprocess.run(cmd, check=True)

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

def run_variant(label, model, spec):
    args = ["podman", "run", "-d", "--name", "bench-variant",
            "--device", RENDER, "--security-opt", "label=disable",
            "-v", f"{MODELS}:/models:ro",
            "-p", "127.0.0.1:8183:8183",
            "--entrypoint", "/app/llama-server", IMAGE,
            "-m", f"/models/{model}", "--device", "Vulkan0",
            "-ngl", "99", "-fa", "on", "-c", CTX,
            "-b", "1024", "-ub", "1024", "-t", "4", "-tb", "6",
            "-ctk", "q4_0", "-ctv", "q4_0", *spec,
            "--host", "0.0.0.0", "--port", "8183"]
    sh(args)
    if not wait_health():
        print(f"{label:22s} HEALTH TIMEOUT (load failed?)", flush=True)
        subprocess.run(["podman", "logs", "--tail", "15", "bench-variant"])
        subprocess.run(["podman", "rm", "-f", "bench-variant"])
        return
    for tname, pr in (("code_review", CODE_REVIEW), ("code_gen", CODE_GEN), ("long_review", LONG_REVIEW)):
        try:
            p, pf, c, dc = bench(LOCAL, pr)
            print(f"{label:22s} {tname:11s} {p:10d} {pf:12.1f} {c:8d} {dc:11.1f}", flush=True)
        except Exception as e:
            print(f"{label:22s} {tname:11s} ERROR: {e}", flush=True)
    subprocess.run(["podman", "rm", "-f", "bench-variant"])

def main():
    print(f"{'variant':22s} {'task':11s} {'prompt_tok':>10s} {'prefill t/s':>12s} {'gen_tok':>8s} {'decode t/s':>11s}")
    print("-" * 82)
    for label, model, spec in VARIANTS:
        run_variant(label, model, spec)

if __name__ == "__main__":
    main()

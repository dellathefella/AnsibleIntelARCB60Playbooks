#!/usr/bin/env python3
"""OpenVINO feasibility probe on the B60 rig (podman, rootless).

Why: OpenVINO's GPU plugin is Level-Zero-only — the same stack llama.cpp's SYCL
backend needs. Without ReBAR (Thunderbolt eGPU) the compute runtime refuses to
create the GPU device, so OpenVINO sees CPU only. This script proves both halves:

  1. device probe  -> available_devices inside the OV container (expect ['CPU'])
  2. CPU decode    -> GenAI LLMPipeline bench on a pre-converted int4 OV model

Measured 2026-10-01 on this rig (i9-9880H, 2026.4 runtime image):
  devices: ['CPU']  (no GPU — no ReBAR, L0 refuses)
  Qwen2.5-Coder-0.5B int4: ~50 t/s   |  7B int4: ~4.6 t/s (bandwidth-bound)
  -> 27B int4 extrapolates to ~1.3-1.6 t/s vs 27-35 t/s on Vulkan. Not viable
     as a serving path; also no MTP/NextN draft-head support in OV GenAI.

Usage:
  python3 bench/ov_probe.py [model-name]   # default: coder 7B int4
"""
import os, subprocess, sys, time

IMAGE = "docker.io/openvino/ubuntu22_runtime:latest"
RENDER = "/dev/dri/renderD129"
CACHE = "/tmp/ov-test"  # model dirs land here

MODELS = {
    "0.5b": "OpenVINO/Qwen2.5-Coder-0.5B-Instruct-int4-ov",
    "7b":   "OpenVINO/Qwen2.5-Coder-7B-Instruct-int4-ov",
}
FILES = ["openvino_model.xml", "openvino_model.bin",
         "openvino_tokenizer.xml", "openvino_tokenizer.bin",
         "openvino_detokenizer.xml", "openvino_detokenizer.bin",
         "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
         "generation_config.json", "config.json"]

BENCH = r'''
import time, warnings, openvino_genai as g
warnings.filterwarnings("ignore")
pipe = g.LLMPipeline("/model", "CPU", {"PERFORMANCE_HINT": "LATENCY"})
pipe.generate(["hi"], max_new_tokens=8, do_sample=False)
prompt = ("Write a production-quality Python function `debounce(func, wait)` for "
          "asyncio. Include full type hints, a docstring, and a usage example.")
for rep in range(3):
    t0 = time.perf_counter()
    out = pipe.generate([prompt], max_new_tokens=128, do_sample=False)
    wall = time.perf_counter() - t0
    toks = out.perf_metrics.get_num_generated_tokens()
    print(f"rep{rep}: gen_tok={toks} decode={toks/wall:.2f} t/s", flush=True)
'''

def sh(cmd, **kw):
    return subprocess.run(cmd, check=True, **kw)

def probe_devices():
    out = sh(["podman", "run", "--rm", "--device", RENDER,
              "--security-opt", "label=disable", IMAGE,
              "python3", "-c",
              "from openvino import Core; print(Core().available_devices)"],
             capture_output=True, text=True)
    devices = out.stdout.strip()
    print(f"ov available_devices: {devices}", flush=True)
    if "GPU" not in devices:
        print("  -> GPU absent (Level Zero refuses without ReBAR); CPU-only fallback", flush=True)
    return devices

def fetch(repo, dest):
    os.makedirs(dest, exist_ok=True)
    for f in FILES:
        p = os.path.join(dest, f)
        if os.path.exists(p):
            continue
        url = f"https://huggingface.co/{repo}/resolve/main/{f}"
        sh(["curl", "-sfL", "-C", "-", "-o", p, url])
    return dest

def bench_cpu(dest):
    t0 = time.time()
    sh(["podman", "run", "--rm",
        "-v", f"{dest}:/model:ro",
        IMAGE, "python3", "-c", BENCH])
    print(f"TOTAL {time.time() - t0:.0f}s")

def main():
    probe_devices()
    key = sys.argv[1] if len(sys.argv) > 1 else "7b"
    repo = MODELS[key]
    dest = fetch(repo, os.path.join(CACHE, repo.split("/")[1]))
    print(f"CPU decode bench: {repo}", flush=True)
    bench_cpu(dest)

if __name__ == "__main__":
    main()

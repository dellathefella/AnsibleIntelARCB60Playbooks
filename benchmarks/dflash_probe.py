#!/usr/bin/env python3
"""DFlash2 vs MTP on the dense Qwen3.8-27B track (B60/Vulkan).

Fair comparison at identical ctx=131072 (200k lacks VRAM for a draft model).
All configs: nohv env (no-ReBAR fix), --parallel 1, q4_0 KV. Run under sudo
with llama-b60 stopped; restores nothing — caller manages services.
"""
import json, os, subprocess, sys, time, urllib.request, uuid

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench import CODE_REVIEW, CODE_GEN, LONG_REVIEW  # noqa: E402

IMAGE = "ghcr.io/ggml-org/llama.cpp:server-vulkan"
RENDER = "/dev/dri/renderD129"
MODELS = "/home/jdella/.local/share/b60-llama/models"
BASE = "http://127.0.0.1:8183/v1"
HEALTH = "http://127.0.0.1:8183/health"
NOHV = ["GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM=1"]
K = 256
CTX = "131072"
TARGET = "Qwen3.8-27B-GSQ-RCO-IQ3_S-Intel-Arc-Tuned-MTP-Q4_K.gguf"
DRAFT_Q4KM = "Qwen3.8-27B-DFlash2-Q4_K_M.gguf"
DRAFT_Q8 = "Qwen3.8-27B-DFlash2-Q8_0.gguf"

def spec_mtp():
    return ["--spec-type", "ngram-mod,draft-mtp",
            "--spec-ngram-mod-n-match", "24", "--spec-ngram-mod-n-min", "1",
            "--spec-ngram-mod-n-max", "5", "--spec-draft-n-max", "5",
            "--spec-draft-p-min", "0.5"]

def spec_dflash(ngram=False):
    s = ["--spec-type", "draft-dflash,ngram-mod" if ngram else "draft-dflash",
         "--spec-draft-n-max", "15"]
    if ngram:
        s += ["--spec-ngram-mod-n-match", "24", "--spec-ngram-mod-n-min", "1",
              "--spec-ngram-mod-n-max", "5"]
    return s

def args(model, spec, draft=None):
    a = ["-m", f"/models/{model}"]
    if draft:
        a += ["-md", f"/models/{draft}"]
    return a + ["--device", "Vulkan0", "-ngl", "99", "-fa", "on",
                "-c", CTX, "-b", "1024", "-ub", "1024", "-t", "4", "-tb", "6",
                "-ctk", "q4_0", "-ctv", "q4_0", *spec,
                "--parallel", "1", "--host", "0.0.0.0", "--port", "8183"]

# label, draft file (None=target only), spec, prompts
CONFIGS = [
    ("mtp5_131k",       None,         spec_mtp(),            ("code_review", "code_gen", "long_review")),
    ("dflash_q4km",     DRAFT_Q4KM,   spec_dflash(),         ("code_review", "code_gen", "long_review")),
    ("dflash_q4km_ng",  DRAFT_Q4KM,   spec_dflash(ngram=True), ("code_review", "code_gen")),
    ("dflash_q8",       DRAFT_Q8,     spec_dflash(),         ("code_review", "code_gen", "long_review")),
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
    return p, prefill_tps, c, decode_tps, (f"{da}/{dn}" if dn else "-")

def wait_health(timeout=420):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen(HEALTH, timeout=5) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(3)
    return False

def run(label, draft, spec, prompts):
    if draft and not os.path.exists(f"{MODELS}/{draft}"):
        print(f"{label:15s} SKIP (draft missing)", flush=True)
        return
    cmd = ["podman", "run", "-d", "--name", "bench-dflash",
           "--device", RENDER, "--security-opt", "label=disable",
           "-v", f"{MODELS}:/models:ro", "-p", "127.0.0.1:8183:8183",
           "--entrypoint", "/app/llama-server"]
    for e in NOHV:
        cmd += ["-e", e]
    cmd += [IMAGE, *args(TARGET, spec, draft)]
    subprocess.run(cmd, check=True, capture_output=True)
    if not wait_health():
        print(f"{label:15s} LOAD_FAILED", flush=True)
        subprocess.run(["podman", "logs", "--tail", "8", "bench-dflash"])
        subprocess.run(["podman", "rm", "-f", "bench-dflash"], capture_output=True)
        return
    for tname in prompts:
        try:
            p, pf, c, dc, acc = bench2(PROMPTS[tname])
            print(f"{label:15s} {tname:11s} prompt={p:5d} prefill={pf:7.1f} gen={c:4d} decode={dc:6.2f} draft_acc={acc}", flush=True)
        except Exception as e:
            print(f"{label:15s} {tname:11s} ERROR: {e}", flush=True)
    subprocess.run(["podman", "rm", "-f", "bench-dflash"], capture_output=True)
    time.sleep(5)

def main():
    t0 = time.time()
    for cfg in CONFIGS:
        run(*cfg)
    print(f"TOTAL {time.time() - t0:.0f}s", flush=True)

if __name__ == "__main__":
    main()

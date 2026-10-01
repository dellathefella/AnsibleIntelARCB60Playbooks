# Benchmark & Tuning Results — Intel Arc Pro B60 (Thunderbolt, no ReBAR)

All numbers measured on the rig described below with the scripts in `bench/`.
Raw harness output is included verbatim at the bottom.

## Environment

| Item | Value |
|---|---|
| Host | Lenovo ThinkPad P53, i9-9880H (8C/16T), 125 GiB RAM, CachyOS kernel 7.2.8-1 |
| GPU | Intel Arc Pro B60 24 GB (BMG G21), Thunderbolt eGPU, PCI `31:00.0`, driver `xe`, `/dev/dri/renderD129` |
| ReBAR | **Absent** (256 MB prefetchable window behind TB bridge `2f:00.0`) → SYCL/Level-Zero unusable, Vulkan only |
| Engine | `ghcr.io/ggml-org/llama.cpp:server-vulkan` — llama.cpp 0.5.0-dev build 11277 (`eae11d221`) |
| Model | `Qwen3.8-27B-GSQ-RCO-IQ3_S-Intel-Arc-Tuned-MTP-Q4_K.gguf` (14.3 GB, sha256 `2170ff04…`) |
| Compare endpoint | Halogen `qwen3.8-flash-next` @ `http://10.0.1.74:8731/v1` (network) |

## Methodology (`bench/bench.py`, `bench/tune.py`)

- `/v1/chat/completions` (what opencode uses), non-streaming.
- Unique nonce per prompt → first call (`max_tokens=1`) is a **cold prefill**;
  `prefill_tps = prompt_tokens / (t_cold − t_per_token_decode)`.
- **Decode** = warm (cached-prefill) generation of `K=256` tokens, averaged over 3 reps.
- `draft_acc` = MTP/ngram draft tokens accepted/generated (from server `timings`).
- Coding prompts: `code_review` (~207 tok), `code_gen` (~120 tok), `long_review` (~2950 tok).

## Final deployed config (post-tuning)

```
-ngl 99 -fa on -c 200000 -b 1024 -ub 1024 -t 4 -tb 6 -ctk q4_0 -ctv q4_0
--spec-type ngram-mod,draft-mtp --spec-ngram-mod-n-match 24
--spec-ngram-mod-n-min 1 --spec-ngram-mod-n-max 5
--spec-draft-n-max 5 --spec-draft-p-min 0.5 --parallel 1
env: GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM=1
```

### B60 (tuned) vs Halogen — same run

```
endpoint                     task        prompt_tok  prefill t/s  gen_tok  decode t/s
B60-local  Qwen3.8-27B       code_review        206        129.2      257        34.0
B60-local  Qwen3.8-27B       code_gen           120        116.3      257        29.6
B60-local  Qwen3.8-27B       long_review       2951        231.9      257        21.0
Halogen    Flash-Next        code_review        208        146.9      257        56.1
Halogen    Flash-Next        code_gen           118        107.0      257        44.2
Halogen    Flash-Next        long_review       2950        871.1      257        42.0
```

Pre-tuning baseline (same script, same day): B60 decode 30.5 / 31.6 / 24.8,
prefill 190.8 / 120.7 / 267.5. Halogen decode 45.1 / 45.2 / 42.6.

## Quant variants (`bench/variants.py`)

| Variant | MTP head | Decode t/s | Notes |
|---|:---:|---:|---|
| `…IQ3_S-…MTP-Q4_K` (deployed) | yes | 22–35 | chosen |
| `…MTP-3.2BPW` | yes | **0.84** | pathological Vulkan dequant; unusable |
| `…Ridge-…Q4_K` | **no** | ~3–5 (ngram-only) | can't use MTP; larger; not competitive |

3.2BPW diagnostic (54-tok prompt): prefill 6.7 t/s, decode 0.84 t/s,
draft acceptance 3/3 — GPU-bound but ~56× off the memory-bandwidth ideal.

## Tuning experiments (`bench/tune.py`)

### Round 1

```
base           code_review prompt=  209 prefill=   54.9 gen= 257 decode= 27.43 draft_acc=180/195
base           code_gen    prompt=  121 prefill=   93.8 gen= 257 decode= 30.57 draft_acc=188/193
base           long_review prompt= 2949 prefill=  223.4 gen= 257 decode= 19.99 draft_acc=174/192
no_hostvis     code_review prompt=  207 prefill=   54.0 gen= 257 decode= 29.45 draft_acc=189/193
no_hostvis     code_gen    prompt=  121 prefill=   90.3 gen= 257 decode= 23.37 draft_acc=165/201
no_hostvis     long_review prompt= 2952 prefill=  225.3 gen= 257 decode= 18.72 draft_acc=161/186
force_mmvq     code_review prompt=  209 prefill=   56.4 gen= 257 decode= 29.71 draft_acc=190/191
force_mmvq     code_gen    prompt=  121 prefill=  118.7 gen= 257 decode= 26.71 draft_acc=175/203
mtp_only_3     code_review prompt=  208 prefill=   56.5 gen= 257 decode= 21.42 draft_acc=160/191
mtp_only_3     code_gen    prompt=  120 prefill=   92.2 gen= 257 decode= 21.53 draft_acc=148/172
mtp_draft8     LOAD_FAILED  (ErrorOutOfDeviceMemory: +820 MB MTP draft ctx @ -c 200000)
ngram8_mtp8    LOAD_FAILED  (same)
np1            code_review prompt=  209 prefill=   56.5 gen= 257 decode= 30.12 draft_acc=189/194
np1            code_gen    prompt=  121 prefill=   95.0 gen= 257 decode= 33.49 draft_acc=186/190
b2048          code_review prompt=  207 prefill=   55.4 gen= 257 decode= 28.64 draft_acc=183/197
b2048          long_review prompt= 2949 prefill=  219.2 gen= 257 decode= 12.39 draft_acc=168/188
cliff8k_nohv   code_gen    prompt=  121 prefill=   33.3 gen= 257 decode= 24.56 draft_acc=178/204
cliff8k_base   code_gen    prompt=  122 prefill=   17.1 gen= 257 decode=   5.38 draft_acc=181/203
TOTAL 1494s
```

### Round 2

```
np1_nohv       code_review prompt=  210 prefill=   56.0 gen= 257 decode= 25.74 draft_acc=171/195
np1_nohv       code_gen    prompt=  121 prefill=   94.1 gen= 257 decode= 22.96 draft_acc=161/175
np1_nohv       long_review prompt= 2952 prefill=  232.0 gen= 257 decode= 20.56 draft_acc=180/187
np1_nohv_d4    code_review prompt=  208 prefill=   56.4 gen= 257 decode= 25.77 draft_acc=186/212
np1_nohv_d4    code_gen    prompt=  120 prefill=   97.2 gen= 257 decode= 24.26 draft_acc=158/193
np1_nohv_d4    long_review prompt= 2950 prefill=  231.2 gen= 257 decode= 21.72 draft_acc=184/196
np1_nohv_d5    code_review prompt=  209 prefill=   55.0 gen= 257 decode= 31.25 draft_acc=211/216
np1_nohv_d5    code_gen    prompt=  121 prefill=  118.7 gen= 257 decode= 29.24 draft_acc=212/212
np1_nohv_d5_131k code_review prompt=  209 prefill=   55.0 gen= 257 decode= 31.97 draft_acc=210/215
np1_nohv_d5_131k code_gen    prompt=  121 prefill=  119.0 gen= 257 decode= 31.40 draft_acc=211/212
TOTAL 448s
```

### Verdicts

| Change | Effect | Action |
|---|---|---|
| `GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM=1` | `-c 8192`: 5.4 → 24.6 t/s (4.5×); neutral at 200k | **adopted** |
| `--spec-draft-n-max 5` (+ngram n-max 5) | 29–32 t/s, acceptance 98–100% | **adopted** |
| `--parallel 1` | +10% round 1; not replicated round 2 | **adopted** (single-stream) |
| `--spec-draft-n-max 8` | OOM at 200k ctx | rejected |
| drop `ngram-mod` (MTP only) | −25% decode | rejected |
| `-b/-ub 2048` | long-ctx decode 20.0 → 12.4 | rejected |
| `GGML_VK_FORCE_MMVQ=1` | within noise | not adopted |

## Interpretation

- **Small-context cliff root cause:** without ReBAR the host-visible VRAM window
  is the 256 MB BAR; the Vulkan backend allocates small KV caches there and
  decode collapses. Large contexts exceed the host-visible budget and land in
  device-local memory — hence `-c 200000` always looked fine. The env var above
  forces device-local allocation regardless of context size.
- **Decode ceiling:** 14.3 GB weights @ ~512 GB/s VRAM ⇒ ~36 t/s theoretical;
  measured 29–34 t/s with speculation ⇒ near memory-bandwidth-bound. TB4 link
  and missing ReBAR do **not** touch steady-state decode (weights never cross it).
- **Halogen gap:** ~1.6–2× decode, ~3.8× long-prefill — different engine/GPU,
  remote (so its compute edge is understated by network time).
- **Noise:** identical configs vary ±20–30% run-to-run on this TB4 setup;
  only large effects are conclusive.
- **vLLM:** Intel support is XPU/SYCL-only (no Vulkan backend) → blocked by the
  same missing ReBAR. llama.cpp/Vulkan is the only GPU path on this rig.

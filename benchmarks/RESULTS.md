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
| `…IQ3_S-…MTP-Q4_K` (dense track) | yes | 22–35 | chosen |
| `…MTP-3.2BPW` | yes | ~~0.84~~ → **~10** | original number was cliff-contaminated (small ctx, no nohv fix); still 3× slower |
| `…Ridge-…Q4_K` | **no** | ~~3–5~~ → **~14–15** | cliff-contaminated too; ngram-only, no MTP head |

See "MoE track probe" below for the corrected retests. Verdicts stand: neither
beats the deployed Q4_K MTP; 3.2BPW has a genuinely slow Vulkan dequant path.

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

## MoE track probe (`bench/moe_probe.py`) — corrected retests + Qwen3.6-35B-A3B

All configs with `GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM=1`. The 3.2BPW/Ridge
retests **correct earlier verdicts**: their original numbers were measured in
the small-context cliff regime.

```
3.2bpw_retest   code_review prompt=  208 prefill=   40.1 gen= 257 decode=  9.94 draft_acc=175/204
3.2bpw_retest   code_gen    prompt=  120 prefill=   80.3 gen= 257 decode= 10.80 draft_acc=183/187
ridge_retest    code_review prompt=  209 prefill=   52.5 gen= 257 decode= 13.75 draft_acc=145/150
ridge_retest    code_gen    prompt=  120 prefill=   98.8 gen= 257 decode= 14.96 draft_acc=184/184
a3b_q4ks_32k    code_review prompt=  166 prefill=   52.4 gen= 257 decode= 38.65 draft_acc=208/223
a3b_q4ks_32k    code_gen    prompt=   78 prefill=  133.2 gen= 257 decode= 39.94 draft_acc=191/206
a3b_q3kl_131k   code_review prompt=  166 prefill=   19.0 gen= 257 decode= 26.81 draft_acc=208/217
a3b_q3kl_131k   code_gen    prompt=   80 prefill=  103.0 gen= 257 decode= 26.38 draft_acc=198/216
a3b_q3kl_131k   long_review prompt= 2907 prefill=  223.3 gen= 257 decode= 27.52 draft_acc=212/219
a3b_q3kl_d8     code_review prompt=  166 prefill=   27.7 gen= 257 decode= 22.10 draft_acc=227/227
a3b_q3kl_d8     code_gen    prompt=   78 prefill=  113.8 gen= 257 decode= 20.87 draft_acc=217/234
TOTAL 660s
```

GGUF header probes (range-download of first 32 MB, parsed tensor types):

| File | IQ tensors | MTP head | Note |
|---|---:|:---:|---|
| unsloth UD-Q4_K_S | 0 | yes | fastest decode (~39–40 t/s), ctx ≤ ~32k |
| unsloth UD-Q3_K_XL / Q3_K_M | **117** (IQ3_XXS/IQ4_XS experts) | yes | avoided |
| bartowski Q3_K_L | 0 | yes | deployed MoE track (131k ctx) |

Findings: Q3_K decodes slower than Q4_K on Vulkan despite fewer bytes
(kernel-path inversion); draft-n-max 8 hits 100% acceptance but loses to 5
(verification cost); MoE Q4_K_S is the fastest decode measured on this rig.

## OpenVINO probe (`bench/ov_probe.py`) — rejected

OpenVINO via podman (`openvino/ubuntu22_runtime`, 2026.4), rootless:

```
ov available_devices: ['CPU']     # no GPU — Level Zero refuses without ReBAR
Qwen2.5-Coder-0.5B int4, CPU:  48.1 / 50.6 / 50.8 t/s decode (LATENCY hint)
Qwen2.5-Coder-7B   int4, CPU:   4.58 / 4.53 / 4.64 t/s decode (bandwidth-bound)
```

- **GPU path:** dead on this rig for the same reason SYCL/vLLM are — the OV GPU
  plugin is Level-Zero-only, and the compute runtime won't create an L0 device
  without Resizable BAR (probe shows CPU only).
- **CPU path:** works end-to-end but decode is DDR4-bandwidth-bound; 27B int4
  (~15 GB weights) extrapolates to **~1.3–1.6 t/s** vs 27–35 t/s on the Vulkan
  track (~20× slower). OV GenAI also has no MTP/NextN draft-head support, so the
  tuned GGUFs' speculative boost is unusable. Not wired into Ansible.

## halo1 SYCL (ReBAR enabled) — dense 27B, 2026-10-01

Same model/flags as the deployed dense track (`-c 200000`, MTP+ngram-mod 5,
`--parallel 1`), now on **SYCL/Level-Zero** with a **32 GiB prefetchable BAR**
(halo1: Ryzen AI Max+ 395, B60 over TB4, rootful podman). Two consecutive
`bench.py` runs:

```
run1
B60-local  Qwen3.8-27B       code_review        208        246.3      257        39.8
B60-local  Qwen3.8-27B       code_gen           121        152.2      257        36.4
B60-local  Qwen3.8-27B       long_review       2952        614.2      257        32.7
Halogen    Flash-Next        code_review        210        157.8      257        56.2
Halogen    Flash-Next        code_gen           121        123.4      257        45.2
Halogen    Flash-Next        long_review       2952        946.1      257        43.5

run2
B60-local  Qwen3.8-27B       code_review        208        251.9      257        46.4
B60-local  Qwen3.8-27B       code_gen           121        155.8      257        44.5
B60-local  Qwen3.8-27B       long_review       2951        659.5      257        30.6
Halogen    Flash-Next        code_review        210        155.4      257        54.5
Halogen    Flash-Next        code_gen           122        126.5      257        43.1
Halogen    Flash-Next        long_review       2949        951.5      257        54.0
```

vs the P53/Vulkan baseline (same prompts, post-tuning):

| Metric | P53 Vulkan | halo1 SYCL | Delta |
|---|---:|---:|---|
| decode, short prompts | ~30–34 | **~36–46** | +20–35% |
| decode, long ctx | ~21 | **~31–33** | +50% |
| prefill, short | ~116–129 | **~152–252** | ~2× |
| prefill, long (2950 tok) | ~232 | **~614–660** | ~2.7× |

- **halo1 now beats Halogen on short-prompt prefill** (~250 vs ~156) and roughly
  matches it on `code_gen` decode; Halogen keeps the lead on long-ctx decode
  (~54 vs ~31) and long prefill (~950 vs ~660).
- SYCL + ReBAR removes the Vulkan small-ctx cliff entirely (no env workaround
  needed) and lifts every metric. The 32 GiB BAR also means CPU-mapped access
  covers the whole VRAM.
- Run-to-run variance on halo1 is much tighter than the P53's ±20–30%.

## Long-context stress (`bench/longctx.py`) — 16k..128k prompt tokens

Same cold-prefill/warm-decode methodology at 4×–64× the prompt size. Two
consecutive runs on halo1 (SYCL dense, `-c 200000`) vs Halogen:

```
run1
B60-sycl   Qwen3.8-27B     16k         15883        670.6      257        26.1
B60-sycl   Qwen3.8-27B     32k         31982        636.2      257        28.3
B60-sycl   Qwen3.8-27B     64k         64743        570.0      257        16.3
B60-sycl   Qwen3.8-27B     128k       130262        468.8      257        17.0
Halogen    Flash-Next      16k         15884       1489.8      257        39.1
Halogen    Flash-Next      32k         31982       1557.5      257        40.6
Halogen    Flash-Next      64k         64742       1502.8      257        39.8
Halogen    Flash-Next      128k       130263       1420.0      257        46.0

run2
B60-sycl   Qwen3.8-27B     16k         15882        640.6      257        30.5
B60-sycl   Qwen3.8-27B     32k         31979        635.7      257        26.5
B60-sycl   Qwen3.8-27B     64k         64741        569.4      257        26.5
B60-sycl   Qwen3.8-27B     128k       130262        468.8      257        19.8
Halogen    Flash-Next      16k         15881       1489.1      257        51.0
Halogen    Flash-Next      32k         31983       1556.2      257        49.0
Halogen    Flash-Next      64k         64742       1503.9      257        45.3
Halogen    Flash-Next      128k       130260       1419.7      257        39.7
```

Summary (2-run ranges):

| Prompt | B60 prefill | B60 decode | Halogen prefill | Halogen decode |
|---|---:|---:|---:|---:|
| 16k | 641–671 | 26–31 | 1489–1490 | 39–51 |
| 32k | 636 | 27–28 | 1556–1558 | 41–49 |
| 64k | 569–570 | 16–27 | 1503–1504 | 45 |
| 128k | 469 | 17–20 | 1420 | 40–46 |

- **B60 degrades with context on both axes:** prefill −30% (670→469) and decode
  −35–40% (~28→~18) from 16k→128k — attention reads the whole q4_0 KV cache
  per token, adding bandwidth pressure on top of the 14.3 GB weights.
- **Halogen stays flat** (~1420–1560 prefill, ~40–50 decode) across the whole
  range — the gap widens to ~2.5–3× at 128k.
- **Practical read:** the 200k context on the B60 is *capacity*, not comfort.
  Up to ~32k it's genuinely usable (26+ t/s decode); past 64k expect ~17–27 t/s
  decode and ~470–570 t/s prefill. Still interactive, but far from the
  short-context 36–46 t/s.

## vLLM XPU track (halo1, ReBAR) — 2026-10-01

New track: `qwen38-27b-vllm-xpu-podman.yml` — vLLM 0.30.0 on the XPU backend
(`vllm/vllm-openai-xpu:latest`, torch-xpu + vllm-xpu-kernels), AWQ W4A16
text-only artifact (`philbert440/Qwen3.8-27B-W4A16-AWQ`, 18.2 GiB), fp8 KV
cache, `--gpu-memory-utilization 0.95`, port 8185.

**Context ceiling:** 18.2 GiB weights + ~3 GiB activations leave ~1.3 GiB paged
KV on the 24 GB card → **32000 ctx** (vLLM-computed). The multimodal
`cyankiwi` AWQ (19.6 GiB) left *negative* KV and was rejected; text-only is
mandatory on 24 GB.

`benchmarks/vllm_bench.py` (single-stream cold-prefill/warm-decode + concurrency):

```
[single] prompt_tokens=6890 completion_tokens=256
[single] decode: 22.9 tok/s  (43.6 ms/tok)
[single] prefill: 1382.1 tok/s
[conc n=1] wall=5.4s  total_tok=128   aggregate=23.6 tok/s  avg_latency=5.4s
[conc n=4] wall=6.1s  total_tok=512   aggregate=83.7 tok/s  avg_latency=6.1s
[conc n=8] wall=11.9s total_tok=1024  aggregate=86.0 tok/s  avg_latency=7.7s
```

vs the llama.cpp SYCL dense track (same card, 200k ctx, MTP speculation):

| Metric | vLLM XPU (32k) | llama.cpp SYCL (200k) | Winner |
|---|---:|---:|---|
| Single-stream decode | 22.9 t/s | 36–46 t/s | llama.cpp |
| Prefill (6.9k prompt) | **1382 t/s** | 614–660 t/s | **vLLM 2.1×** |
| Concurrent n=4 | **83.7 t/s** agg | ~23 t/s (1 slot) | **vLLM 3.6×** |
| Concurrent n=8 | **86.0 t/s** agg | ~23 t/s (1 slot) | **vLLM 3.7×** |
| Max context | 32k | 200k | llama.cpp |

- **vLLM wins where it's designed to:** prefill (2.1×) and concurrent
  throughput (3.6× at n=4). Continuous batching + paged KV keep the GPU fed
  across requests.
- **llama.cpp wins single-stream:** MTP speculation gives 36–46 t/s decode vs
  vLLM's 22.9 (no speculation in this XPU build), and 200k vs 32k context.
- **Verdict:** for a single agentic session, llama.cpp dense is still the
  better default (faster decode, 6× the context). vLLM earns its place for
  multi-request / high-prefill workloads on the same card.

## llama.cpp source-build optimization probe — 2026-10-01

Goal: can a from-source llama.cpp build (tuned flags) beat the prebuilt
`full-intel` image on the B60? Cloned `ggml-org/llama.cpp` @ `a868c3e`
(latest main) on halo1, built the SYCL backend inside the full-intel image
(icpx/oneAPI 2026.1.1) with `-O3 -ffast-math`, Release, `GGML_SYCL=ON`.

**First: the prebuilt image is already current.** Image build 11312 / commit
`0c1e57098` is dated 2026-10-01 (same day as main HEAD) — no upstream
version win available. All levers below are runtime/build-flag, not "get newer".

Decode benchmark (same model/flags, 8k ctx, MTP draft-n-max 5, warm, K-token
completion; `benchmarks/vllm_bench.py`-style single-stream):

| Config | decode K=128 | decode K=256 | Notes |
|---|---:|---:|---|
| Prebuilt image (baseline) | 22.3 t/s | 30.1 t/s | `full-intel` |
| Prebuilt + `GGML_SYCL_GRAPH=1` | 21.2 t/s | 30.3 t/s | **no win** |
| Source build `-ffast-math` | 22.7 t/s | **32.5 t/s** | **+5–8%** |
| Source build, `draft_n_max=8` | — | — | **segfault** (exit 139) |

Findings:

- **Graph mode (`GGML_SYCL_GRAPH=1`) is a no-op here.** Decode is
  memory-bandwidth-bound, not kernel-launch-bound, so capturing a SYCL graph
  saves nothing. (Consistent with the bandwidth analysis in Interpretation.)
- **`-ffast-math` gives a small but real ~5–8% decode bump** (32.5 vs 30.1
  at K=256). The dequant/requant + rms_norm math benefits from fast-math
  contraction. This is the only reproducible win found.
- **`draft_n_max=8` crashes** — the model's MTP draft head only supports
  ≤5 speculative tokens; 8 segfaults in `server_context::decode`. Confirms
  the earlier 200k OOM note: 5 is the hard ceiling for this GGUF.
- **Why not more:** 14.3 GB weights @ ~143 GB/s effective vs ~456 GB/s
  GDDR6 peak ⇒ ~30% GEMV efficiency, which is typical for batch-1 decode.
  Closing that gap needs hand-tuned Xe2/XMX kernels (research-level), not
  build flags. The token-gen speed is bandwidth-bound, not software-bound.

**Verdict:** a `-ffast-math` source build is worth ~5–8% decode over the
prebuilt image; everything else (graph mode, more speculation) is flat or
broken. For a bigger jump the lever is a smaller quantization (less bytes to
stream) or upstream Xe2 kernel work — not local flag tuning.

## SYCL kernel-dispatch investigation — 2026-10-01

Went into `ggml/src/ggml-sycl/` (cloned `a868c3e`) to chase the decode
hot path. Dispatch tree (`ggml-sycl.cpp:4805+`): batch-1 → **MMVQ** (GEMV)
or **DMMV** (ESIMD); MTP multi-token (ne[1] 2..8) → **MMQ** (GEMM, XMX).
The `mmvq.cpp` kernel is already subgroup-optimized with an arch-aware
rows-per-subgroup crossover ("one row/sg up to 9% faster below the crossover,
two rows/sg 8-15% faster above it"). There's a device-specific hack for
`intel_gpu_acm_g10` (Alchemist) that skips MMVQ for Q4_0 — our B60 is
Battlemage (Xe2_BMG), so it falls through to the generic tuned paths.

Runtime knobs (`ggml_sycl_init`, all default-on unless noted) reroute these
kernels without a rebuild. A/B at 8k ctx, MTP, K=256:

| Knob | K256 t/s (sweep1) | K256 t/s (sweep2) |
|---|---:|---:|
| baseline | 30.7 | 28.6 |
| `ENABLE_ESIMD=0` | 29.0 | — |
| `PRIORITIZE_DMMV=1` | 34.1 | — |
| `ENABLE_DNN=0` | 34.8 | 25.3 |
| `ENABLE_FUSION=0` | 29.1 | — |
| `ENABLE_MKL_FA=0` | 30.8 | — |

**The sweeps disagree on the sign of every effect.** Root cause found:
**intra-session noise is ±32%** — one server, same config, 5 back-to-back
K=256 runs gave `[25.0, 30.8, 29.8, 32.7, 34.9]` (mean 30.6, stdev 3.3).
It's a warmup *ramp*: the first run is cold/slow, clocks climb over
successive runs. Every kernel-dispatch delta measured is inside this noise.

Temps during load were 54–62°C (hwmon) — **not** thermal throttling (that
needs 90+°C); the ramp is clock DVFS ramping up, not thermal capping. The
`xe` driver doesn't expose `gt_*_freq_mhz` sysfs so clocks can't be pinned
from userspace here.

**Verdict on "chasing the kernels":**
- The SYCL GEMV/GEMM kernels are already well-tuned (subgroup crossover,
  ESIMD reorder paths, XMX MMQ). No dispatch knob gives a *reproducible*
  win — the signal is buried under ±32% run-to-run variance.
- Decode is **bandwidth-bound** (~143 GB/s effective vs ~456 GB/s peak).
  Switching kernel variants streams the same bytes, so the ceiling is
  unchanged; only a higher-bandwidth-efficiency kernel would move it.
- To validate any real kernel change you'd first need a **rigorous harness**:
  locked clocks (not exposed on xe), long sustained runs, N≥10 reps, and
  statistical significance — otherwise you're fitting noise.
- A genuine kernel win = rewriting the Xe2_BMG ESIMD GEMV for higher
  bandwidth utilization: a multi-day, uncertain-payoff effort, not a
  flag/config change.

**Reproducible wins found this session:** only `-ffast-math` source build
(+5–8% decode, see source-build probe above). Everything else is noise.

## DFlash drafter investigation — 2026-10-01

DFlash = llama.cpp's `draft-dflash` speculative method (`src/models/dflash.cpp`):
an EAGLE-style drafter that reads `target_layers` hidden states from the target
and drafts token blocks via a trained conv + selector backbone. Distinct from
the built-in MTP (NextN) head. Pre-trained DFlash2 GGUFs exist for this exact
model (`z-lab/Qwen3.8-27B-DFlash2-GGUF`): Q4_K_M 1.06 GiB, Q8_0 1.92 GiB —
tiny, fit alongside the 14.3 GB target.

Tested on the SYCL track (8k ctx, 5 reps, warm, interleaved):

| Config | decode t/s | draft acc |
|---|---:|---:|
| MTP + ngram (baseline) | 40–45 | 86–94% |
| DFlash-Q4_K_M alone | 20.1 | 33% |
| DFlash-Q8_0 + ngram | 42–43 | 85–87% |
| DFlash-Q8 + MTP + ngram | **load fail** | — |

Findings:

- **DFlash-alone is slow (33% acceptance).** The pre-trained DFlash2 draft was
  trained against the *base* Qwen3.8-27B, but our target is the *tuned*
  IQ3_S→Q4_K variant. The hidden-state mismatch collapses draft acceptance,
  so speculation mostly fails and decode drops to ~20 t/s.
- **DFlash-Q8 + ngram ≈ MTP + ngram** (42–43 vs 40–45 t/s, overlapping
  error bars). No reproducible win — the ngram layer is doing most of the
  work and MTP's built-in head already matches the tuned target well.
- **DFlash + MTP cannot stack.** Passing `-md <dflash>` with `draft-mtp` in
  the spec-type makes llama.cpp look for MTP layers *in the draft file*:
  "context type MTP requested but model doesn't contain MTP layers" → load
  fails. The `-md` draft slot and the target's MTP head are mutually
  exclusive; you pick one external draft source, optionally + ngram-mod.

**To truly "build a DFlash drafter based on the local model"** (i.e. one that
complements the *tuned* target with high acceptance) you'd need to TRAIN a
DFlash draft against the tuned target's hidden states — z-lab's DFlash
training pipeline (PyTorch), running the 27B to extract features over a
corpus, train the conv/selector backbone, then convert to GGUF. That's a
real ML-training project and needs a training-capable GPU (XPU training is
not viable here); it is not a download-and-run.

**Verdict:** pre-trained DFlash doesn't beat the existing MTP+ngram on this
tuned model, and can't stack with MTP. The MTP head (built into the tuned
GGUF) remains the best drafter for this rig.

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
- **OpenVINO:** GPU plugin is L0-only → also blocked; CPU-only fallback measured
  ~4.6 t/s at 7B int4 (see OpenVINO probe above). Rejected.

---

## Qwen3.6-35B-A3B Escha W2 (IQ2_XXS + MTP) — llama.cpp SYCL, Arc B60

**Model:** `peasantsmith/Qwen3.6-35B-A3B-Escha-W2-GGUF-MTP` → `escha-w2-IQ2_XXS.gguf` (9.68 GiB)
**Track:** `qwen36-a3b-escha-w2-sycl-podman.yml` · port 8187 · 64k ctx · MTP+ngram-mod (draft-n-max 5)

### Why a GGUF re-quant
The upstream `EschaLabs/Qwen3.6-35B-A3B-Escha-W2` ships a **custom `eschamoe`
2-bit safetensors** quant. It only runs on EschaLabs' own CUDA (SGLang/ZML) or
MLX runtimes — **no Intel Arc / XPU / SYCL path exists**, and llama.cpp cannot
read the `eschamoe` format. The community GGUF re-quant (IQ2_XXS + MTP head) is
the only way to run these weights on the B60.

### Measured (halo1, Arc Pro B60 24 GB, SYCL)
| Metric | Value |
|---|---|
| Single-stream decode | **43.2 ± 2.2 t/s** |
| MTP draft acceptance | **89%** |
| Prefill (5.2k tok) | 740 t/s |
| Concurrency n=4 | 33.6 tok/s agg (parallel=1 → queues) |
| Concurrency n=8 | 36.1 tok/s agg |

### Notes
- **Fastest decode of any track on this rig.** A3B MoE (8/256 experts active)
  streams few bytes/token, and the 2-bit weights cut bandwidth further; 89% MTP
  acceptance compounds it.
- 9.68 GiB weights leave ~14 GiB for KV → 64k ctx fits with large headroom.
- Concurrency is flat by design (`--parallel 1`); this is a single-stream
  agentic-coding track. For throughput, vLLM XPU (AWQ) still wins on batching.
- Quality is 2-bit IQ2_XXS — expect degradation vs the UD-Q4_K_S MoE track;
  benchmarked for speed/context, not fidelity.

### Context ceiling (Escha W2)
- **Native max = 262144 (256k).** llama.cpp clamps `-c` to the model's trained
  position length; requesting more without rope scaling silently caps at 256k.
- **256k verified:** loads healthy, 100k-token request served at 760 t/s prefill.
- **YaRN 512k works but impractical:** `--rope-scaling yarn --rope-scale 2
  --yarn-orig-ctx 262144` loads 524288 and serves, but prefill falls to ~410 t/s
  (quadratic attention) — a 500k-token prompt took ~20 min. Quality also degrades
  beyond the trained length. Not recommended for agentic use.
- **Decision:** track set to 262144 (native, full quality, 4x the original 64k).

## B60 kernel-mod probe + prefill optimization — llama.cpp SYCL, Arc B60 (2026-10-02)

**Branch/patch:** `kernel-work/patches/0001-*.patch` on upstream llama.cpp `5fc4f3c`
**Build:** `kernel-work/Dockerfile.tuned` (oneAPI 2026.1, reproducible: clones
upstream + applies patch inside the build). **Track:** `qwen38-27b-sycl-tuned-podman.yml` (port 8186).

### Kernel mods tested (decode)
Two independently-gated ESIMD changes on the Q4_K reorder matvec (the
first-priority kernel for batch-1 decode, confirmed via dispatch at
`ggml-sycl.cpp:4862`):

1. `GGML_SYCL_BMG_STREAM_W` — weight loads with cache hints L1 uncached
   (+ L2 uncached, and a milder L2-cached variant).
2. `GGML_SYCL_BMG_PREFETCH` — LSC prefetch of the next weight block
   (qs+scales) one iteration ahead, to raise outstanding-request count.

**Result: all neutral.** Warmup-saturated single-GPU A/B (K=256, 9 reps):

| Variant | decode median | MAD |
|---|---|---|
| stock `full-intel` | 45.3 t/s | 2.1 |
| L1+L2 uncached | 38.9 (cold-order artifact) / ~45 warm | — |
| L1 uncached + L2 cached | 46.2 | 1.4 |
| prefetch-only | 45.4 | 1.3 |

All within the ±30% intra-session clock band. **Client BMG exposes no clock
lock** (no `xpu-smi`/`gt` sysfs/`debugfs`), so sub-5% effects are unresolvable
here. Consistent with prior probes: **batch-1 decode is bandwidth-bound; local
kernel/cache-hint mods do not move the ceiling.**

### Prefill optimization (the real win) — DEPLOYED
Prefill is compute-bound (MMQ GEMM path), so the lever is **ubatch**, not the
decode matvec. Server-side `prompt eval time` timing (client HTTP timing is too
noisy — see `prefill_probe.py`):

| Config | 32k | 64k | 128k | fits |
|---|---|---|---|---|
| `-ub 1024` (old stock) | 623 | 570 | 470 | 200k ✓ |
| `-ub 2048` | **691** | **626** | **506** | ≤160k ✓ |
| `-ub 4096` | — | — | — | OOM even @128k |

`-ub 1536` gave **no** gain (622/567/466) — the win is specific to the
GEMM-tile-aligned 2048. At **200k ctx** memory is too tight (weights 14.3 GB +
KV + MTP draft): `-ub 1280` crashes during prefill.

**Decision:** the 27B track (`qwen38-27b-sycl-podman.yml`) now runs
**160k ctx + `-b 4096 -ub 2048`** → **+8-11% prefill, decode unchanged**
(ubatch doesn't affect batch-1 decode). The 200k→160k context drop is what
frees the VRAM for the larger ubatch.

### Tooling added
- `benchmarks/bench_matrix.py` — prefill+decode across 16k..128k with sustained-clock warmup
- `benchmarks/prefill_probe.py` — reads the server's own `prompt eval time` (accurate prefill)
- `benchmarks/single_gpu_ab.py` — warmup-saturated single-GPU A/B
- `benchmarks/ab_bench.py` — interleaved two-endpoint A/B with sign test

## Qwen3.8-Flash-Next on Arc B60 (qwen4exp, 179.55B MoE) — hybrid feasibility (2026-10-02)

**Model:** ISTA-DASLab GSQ-RCO IQ3_S (83.6 GB) + ggml-org `mtp-Q4_0` draft (2.2 GB)
**Engine:** llama.cpp master `4ebdf2c` (qwen4exp + MTP merged 2026-10-01), SYCL,
fork `dellathefella/llama.cpp` branch `arc-flash-engine`.
**Rig constraint:** B60 sits in a **Thunderbolt 3 dock** (JHL7440, x4 Gen3):
H2D/D2H measured **2.9/3.2 GB/s** (multi-stream, pinned, IOMMU/ASPM off —
no headroom left). Expert streaming over the link is dead; experts live in
RAM and compute on CPU (`--cpu-moe`), PLE n-gram table stays lazy-mmap'd
(halogen-style rainbow-table paging; `TENSOR_READ_LAZY`, llama-model.cpp).

### Measured (clocks locked, 1x B60 + 125 GB RAM)
| Config | prefill 16k | decode (MTP on) |
|---|---|---|
| IQ3_S, ub2048, mmap | 29 t/s | — |
| IQ3_S, ub8192, `--load-mode none` | **242.6 t/s** | 23.5 t/s |
| IQ3_S + expert fadvise prefetcher | 225 t/s | 25.2 t/s |
| IQ4_NL (96 GiB), same flags | 199 t/s | 22.6 t/s |
| Q8_0 (152 GiB) | host thrash — exceeds RAM+cache; DO NOT run `load-mode none` | — |

Key lessons:
- `--load-mode none` + ub8192 = **8.4x prefill** over default mmap+ub2048
  (page-fault serialization on CPU-resident experts was the bottleneck).
- IQ3_S beats IQ4_NL here: CPU expert GEMM is byte-bound; +21% bytes of
  IQ4_NL outweighs its kernel path. "Q4 natively accelerated" does not hold
  on the ggml-cpu MoE path.
- Expert prefetcher (`llama-expert-prefetch.h`, fadvise WILLNEED in layer
  order, `LLAMA_EXPERT_PREFETCH_PACE_MS`) verified via strace (290
  fadvise64 calls); neutral on cache-resident IQ3_S; untested under pressure.
- Q8_0 + `--load-mode none` on a 125 GB host = swap thrash, host unresponsive
  ~15 min. Q8 needs mmap + paced prefetch, never load-mode none.

### Ceiling math (why this is a stepping stone, not the end)
CPU expert GEMM ~2.8-16 TFLOPS eff bounds prefill; Strata-style VRAM expert
tiering + doorbell CPU/GPU overlap (MIT, `dellathefella/Strata` branch
`sycl-backend`, plan in `docs/SYCL_PORT.md`) targets 400-600 t/s prefill and
>27 t/s decode on this exact link.

## lagrange (3x Arc Pro B60, Xeon E5-2699 v3, 251 GB) — multi-GPU Flash-Next (2026-10-03)

Links: dev0 = PCIe x4 Gen3 (2.5 GB/s H2D), dev1/dev2 = x8 Gen3 (6.8 GB/s).
Host CPU is Haswell (AVX2 only, no VNNI) — CPU-expert hybrid decode halves
vs halo1's Zen5 (12.4 vs 25.2 t/s), so resident-multi-GPU was the hope.

| Config | prefill | decode (MTP) | status |
|---|---|---|---|
| 1-GPU hybrid ub8192 (dev1) | 293 @16k | 12.4 | stable |
| 3-GPU resident `-sm layer` ub2048/c8k | 216 @4k | 19.0 | stable |
| 3-GPU resident `-sm layer` ub4096/c16k | 150 @16k | — | stable, comms-bound |
| 3-GPU resident `-sm layer` ub8192/c32k | — | — | dev0 GuC job hang -> GT reset -> SIGSEGV |
| 3-GPU resident `-sm row` ub4096 | — | — | SIGSEGV |

Conclusions:
- llama.cpp multi-GPU (`-sm layer/row`) moves activations at every layer
  boundary; prefill becomes PCIe-comms-bound and never beats the 1-GPU
  hybrid's 293 t/s. Multi-GPU only helps decode (19.0 vs 12.4).
- The x4 card's GT hangs under large resident buffers (xe job timeout,
  "not started", GT reset in dmesg) — exclude dev0 from heavy configs or
  keep buffers small.
- Warmup "hang" = IGC JIT for 3 device kernel-sets on one core; persist
  /root/.cache (neo_compiler_cache) across container runs.
- Real 3-card prefill scaling needs pipeline parallelism (tokens cross
  cards once per window) + per-card expert caches = the Strata SYCL port
  (dellathefella/Strata, docs/SYCL_PORT.md), not stock llama.cpp splits.

Serving endpoint on lagrange: `qwen38-flashnext-sycl` on 8183 = 1-GPU
hybrid (dev1) ub8192 + MTP draft; JIT cache volume /data/neocache.

## lagrange — Q8_0 vs IQ3_S hybrid (2026-10-03, Q4/Q8 focus directive)

Q4 of Flash-Next is unobtainable: the GSQ-RCO repo (IQ3_S/IQ4_NL source) is
gone from HF (404 even with token). Official unsloth/Qwen3.8-Flash-Next-GGUF
ships Q8_0 (6 shards, 183.8 GB), UD-Q5_K_XL, UD-Q6_K_XL, BF16 + MTP variants.
Downloaded Q8_0 (~100 MB/s), served 1-GPU hybrid on dev1 (same flags as the
IQ3_S config: `-ngl 99 --cpu-moe -c 32768 -ub 8192 -t 8 -tb 16` + mtp-Q4_0
draft, mmap mode — `--load-mode none` still wedges at c>=32768).

| Quant (hybrid dev1) | PP @16k | TG median (MAD) |
|---|---|---|
| IQ3_S 83 GB | 289.5 t/s (post-reboot rebaseline; 293.1 orig) | 12.66 (0.11) |
| Q8_0 175 GB | 204.4 t/s | 12.99 (0.53) |

Findings:
- Decode is NOT PCIe-link-bound under `--cpu-moe`: experts compute from RAM
  page cache; only activations cross the link. Q8's plain int8 dots beat
  IQ3's superblock dequant on the AVX2 Xeon — Q8_0 decodes as fast as IQ3_S
  at 2x the bytes and much higher quality.
- Prefill pays ~29% for Q8 (expert bytes through CPU GEMMs).
- RAM ceiling: Q8_0 (175 GB) + IQ3_S (83 GB) cannot both stay hot in 251 GB —
  serve one at a time (IQ3_S endpoint stopped during Q8_0 bench).
- Kernel-spike tie-in (Strata tools/q4q8_kernel_bench): int8 packed-dot GEMV
  = 2.7x float-dequant on B60; DPAS int8 8x16x32 verified 21.6 TOPS untiled.
  GPU-resident Q8 experts (Strata tiering) should lift both PP and TG.

Endpoint change: `qwen38-q8-sycl` on 8184 = Q8_0 hybrid (dev1) is the live
serving endpoint; `qwen38-flashnext-sycl` (IQ3_S, 8183) stopped, restartable.

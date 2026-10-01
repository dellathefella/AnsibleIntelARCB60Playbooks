# B60 · Qwen 3.8 27B · llama.cpp via Podman

Ansible wiring for running **Qwen 3.8 27B** on an **Intel Arc Pro B60** (Battlemage G21)
using a **detached Podman container** (no systemd unit). The engine is served through an
OpenAI-compatible API and an `opencode.json` is emitted on the controller.

## The one thing that matters on Arc

The tuned GGUFs (repo `Greatjedi/Qwen3.8-27B-Intel-Arc-Tuned-GGUF`) are built with
**0 I-Quant tensors** so the XMX/DPAS matrix units stay on the GPU. That is a **SYCL**
concern. On this rig the B60 is a **Thunderbolt eGPU with no Resizable BAR**, and the
Intel compute runtime refuses to create a Level-Zero device without it:

```
WARNING: Resizable BAR not detected for device 0000:31:00.0   -> 0 SYCL GPUs
```

**Vulkan does not need ReBAR** (the BAR only limits CPU-mapped access), so the role
auto-selects the Vulkan backend:

```
Vulkan0: Intel(R) Arc(tm) Pro B60 Graphics (BMG G21) (24480 MiB, 21574 MiB free)
```

Backend selection is automatic: **SYCL if a `level_zero:gpu` exists, else Vulkan.**
Force with `-e llama_backend=sycl|vulkan`.

## Run it

Runs **rootful** by default (`podman_as_root: true`), so pass `--ask-become-pass`
(or configure a NOPASSWD sudoers rule).

```bash
# local (this host owns the Thunderbolt B60) — full real model, 24GB card
ansible-playbook site.yml --ask-become-pass

# staged / smoke test with a tiny stand-in model (no 14GB download)
ansible-playbook site.yml \
  -e model_repo=Qwen/Qwen2.5-0.5B-Instruct-GGUF \
  -e model_file=qwen2.5-0.5b-instruct-q4_0.gguf \
  -e model_sha256=7671c0c304e6ce5a7fc577bcb12aba01e2c155cc2efd29b2213c95b18edaf6ed \
  -e model_size=428730208 -e ctx_size=4096 -e enable_speculative=false
```

Everything is tunable in **`group_vars/all.yml`** (quant profile, context, threads,
speculative decoding, ports, toggles). Model downloads are resumable + sha256-verified.

## On the target host

A control script is dropped on every inventory host and is the authoritative runner:

```bash
sudo ~/scripts/llama-b60.sh {start|stop|restart|status|logs}
```

`start` is idempotent — it recreates the container only if the baked-in config hash
changes. API: `http://127.0.0.1:8183/v1`.

## opencode config (written to the executor)

`opencode.json` is rendered on the machine running the playbook (not the target):

```json
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "qwen-arc": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "Qwen 3.8 27B (Arc B60)",
      "options": { "baseURL": "http://127.0.0.1:8183/v1" },
      "models": { "qwen3.8-27b": { "name": "Qwen 3.8 27B (Arc B60)" } }
    }
  },
  "model": "qwen-arc/qwen3.8-27b"
}
```

Quit and restart opencode to load it (config is not hot-reloaded).

## Going remote

Edit `inventory/hosts.yml`: swap the `local` connection for `ssh` + the B60 host's
address/user. Nothing else changes — the render node is auto-detected on whatever host
you target.

## Benchmark findings (B60 / Vulkan)

Full raw outputs + environment: **[RESULTS.md](RESULTS.md)**.

Scripts: `bench/bench.py` (B60 vs Halogen) and `bench/variants.py` (quant-variant
shootout). Methodology per prompt: a unique nonce forces a **cold prefill** call
(`max_tokens=1`), then warm (cached-prefill) decode is averaged over 3 reps at
`K=256` via `/v1/chat/completions` (what opencode actually uses).

### Quant variants from the repo (on the B60)

| Variant | Quant | MTP head | Decode t/s | Verdict |
|---|---|:---:|---:|---|
| `…IQ3_S-…MTP-Q4_K` (**current**) | Q4_K | yes | **~22–35** | Best — keep it |
| `…MTP-3.2BPW` | ~3.2 bpw | yes | **0.84** | Unusable — pathological dequant on Vulkan |
| `…Ridge-…Q4_K` | Q4_K | **no** | ~3–5 (ngram-only) | Not competitive — no MTP head |

- **3.2BPW is dead on arrival here.** ~0.84 t/s decode — ~30× slower than the
  current Q4_K. On Vulkan this quant hits a pathological unpack path (the inverse
  of the author's SYCL "0 IQ tensors" story, which doesn't apply to Vulkan).
- **Ridge Q4_K has no MTP head**, so it can't use the speculative-decoding boost
  that makes the current model fast; with ngram-only it sits at ~3–5 t/s. Same
  Q4_K type but ~2 GB larger and without MTP = a downgrade on this rig.

### Key operational discovery: the small-context cliff (and its fix)

Decode throughput collapses at small `-c` on this rig:

| `-c` | decode t/s (before fix) | decode t/s (with fix) |
|---:|---:|---:|
| 8192 | ~4.6–5.4 | **~24.6** |
| 32768 | ~13–17 | — |
| 200000 | ~22–35 | ~21–34 |

**Root cause (no-ReBAR):** host-visible VRAM is capped at the 256 MB BAR. The
llama.cpp Vulkan backend allocates *small* KV caches in host-visible memory —
which over Thunderbolt without ReBAR is a pathologically slow path. Large
contexts exceed the host-visible budget and land in fast device-local memory,
which is why `-c 200000` was always fine.

**Fix (deployed):** `GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM=1` forces device-local
allocation — 4.5× faster at small contexts, neutral at large. Keep `-c 200000`
regardless; the env var is the safety net.

### Tuning experiments (`bench/tune.py`)

vLLM is **not viable** here: its Intel path is XPU/SYCL-only (no Vulkan backend),
and SYCL needs the ReBAR this Thunderbolt link doesn't expose. All tuning is
llama.cpp/Vulkan. Measured (decode t/s, current Q4_K):

| Change | Result | Action |
|---|---|---|
| `GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM=1` | fixes small-ctx cliff, neutral at 200k | **adopted** |
| `--parallel 1` | +~10% in round 1 (single stream); didn't replicate in round 2 | **adopted** (opencode is single-stream) |
| `--spec-draft-n-max 5` (+ngram n-max 5) | ~31–32 t/s, draft acceptance **98–100%** | **adopted** |
| `--spec-draft-n-max 8` | OOM at 200k ctx (+820 MB MTP draft ctx) | rejected |
| drop `ngram-mod` (MTP only) | −25% decode | rejected |
| `-b/-ub 2048` | prefill flat, long-ctx decode worse (12.4 vs 20.0) | rejected |
| `GGML_VK_FORCE_MMVQ=1` | noise-level | not adopted |

**Noise warning:** B60-over-TB4 run-to-run variance is ±20–30% on identical
configs — only large effects (cliff fix, 3.2BPW, ngram-mod removal) are
conclusive; treat ±10% deltas as unproven.

### B60 vs Halogen (Flash-Next) reference

| Task | B60 prefill | B60 decode | Halogen prefill | Halogen decode |
|---|---:|---:|---:|---:|
| code_review (~200 tok) | ~129 | ~34 | ~147 | ~56 |
| code_gen (~120 tok) | ~116 | ~30 | ~107 | ~44 |
| long_review (~2950 tok) | ~232 | ~21 | ~871 | ~42 |

(Post-tuning numbers, same run for both endpoints.)

Halogen is ~1.6–2× faster on decode and ~3.8× faster on long-context prefill.
Caveats: different model + engine + network (Halogen is remote, so its raw compute
edge is understated); single-stream; MTP on both. **B60-over-Thunderbolt decode is
noisy** — run-to-run variance on identical configs is large. The 3.2BPW signal
(0.84) is far outside that noise and conclusive; treat the Ridge absolute number
as approximate.

**Bottom line:** stay on `…IQ3_S-…MTP-Q4_K` at `-c 200000`. Neither alternative
helps on this hardware.

## Layout

```
site.yml                     entrypoint
ansible.cfg
inventory/hosts.yml          b60 host (local now, ssh later)
group_vars/all.yml           ALL tunables
roles/b60_llama/
  tasks/{main,preflight,model,service,opencode}.yml
  templates/{run-llama-b60.sh.j2, opencode.json.j2}
bench/
  bench.py             B60 vs Halogen (cold prefill + warm decode)
  variants.py          quant-variant shootout (sequential on-GPU, rootful)
  tune.py              flag/env tuning harness (sequential configs, rootful)
```

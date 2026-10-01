# B60 · Qwen 3.8 27B · llama.cpp via Podman

Ansible wiring for running **Qwen 3.8 27B** on an **Intel Arc Pro B60** (Battlemage G21)
using a **detached rootful Podman container** (no systemd unit). The engine is served
through an OpenAI-compatible API and an `opencode.json` is emitted on the controller.

## The one thing that matters: ReBAR is mandatory

**Resizable BAR is a hard requirement — no opt-out.** Without a large prefetchable
BAR the Level-Zero device cannot be created, and the Vulkan fallback performs
abysmally on this workload (~5 t/s small-context cliff on the old P53). The
preflight measures the largest prefetchable BAR on the Arc PCI device and **fails
fast** with the measured size if it is under 1 GiB.

**SYCL/Level-Zero (`full-intel` image) is the only supported backend**, and
podman runs **rootful** (hard-coded `become: true`).

| Rig | Link | ReBAR | Status |
|---|---|:---:|---|
| `halo1` — Strix Halo desktop (Ryzen AI Max+ 395) | TB4 | **32 GiB** | **supported** — SYCL live |
| `b60` — ThinkPad P53 (i9-9880H) | TB3/4 | 256 MiB | legacy — fails the gate by design |

## Run it

Runs **rootful** — pass `-K` (`--ask-become-pass`) or configure NOPASSWD sudoers.

```bash
# deploy the dense 27B track on halo1 (ReBAR-verified, SYCL)
ansible-playbook site.yml --limit halo1 -K

# staged / smoke test with a tiny stand-in model (no 14GB download)
ansible-playbook site.yml --limit halo1 -K \
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

`inventory/hosts.yml` already carries **halo1** (`ssh`, `jdella@10.0.1.67`)
alongside the legacy local `b60`. Target a host with `--limit`. Nothing else
changes — the render node, PCI id and ReBAR window are auto-detected on whatever
host you target.

## Tracks (dense / MoE)

Two model tracks share the one GPU — **mutually exclusive at runtime**; each
control script stops its sibling before claiming the device.

| Track | Model | Ctx | Decode t/s (P53/Vulkan) | Service / port |
|---|---|---:|---:|---|
| `dense` | Qwen3.8-27B IQ3_S→Q4_K (MTP) | 200k | ~27–35 | `llama-b60` / 8183 |
| `moe` | Qwen3.6-35B-A3B Q3_K_L (MTP) | 131k | ~26–28 | `llama-b60-moe` / 8184 |

```bash
ansible-playbook site.yml --limit halo1 -K                      # provision both, start active_track
ansible-playbook site.yml --limit halo1 -K --tags moe -e active_track=moe   # MoE only
sudo ~/scripts/llama-b60.sh start                  # switch to dense (stops moe)
sudo ~/scripts/llama-b60-moe.sh start              # switch to moe (stops dense)
```

`opencode.json` lists **both providers** (`qwen-arc`, `qwen-arc-moe`); the
default model follows `active_track`. Switch tracks in opencode by picking the
model, or restart the sibling service.

**MoE candidate shoot-out** (`bench/moe_probe.py`, all with the nohv fix):

| Candidate | Size | IQ tensors | MTP | Ctx | Decode t/s |
|---|---:|---:|:---:|---:|---:|
| unsloth UD-Q4_K_S | 19.9 GiB | 0 | yes | 32k | **38.7–39.9** |
| bartowski Q3_K_L (**deployed**) | 16.6 GiB | 0 | yes | **131k** | 26.4–27.5 |
| bartowski Q3_K_L, draft-n-max 8 | — | — | yes | 131k | 20.9–22.1 (100% acc, still slower) |

- unsloth UD-Q3_K_M/XL carry **117 IQ expert tensors** (GGUF header-parsed) →
  avoided on this rig; bartowski Q3_K_L is pure K-quants **and keeps the NextN
  MTP head** → the max-context pick.
- Q3_K decodes *slower* than Q4_K on Vulkan despite being 3.3 GB smaller —
  kernel-path inversion. For pure speed at 32k ctx, flip the moe track to
  `model_profile: a3b_q4ks` + `ctx_size: 32768` in `group_vars/all.yml`
  (~+45% decode).
- draft-n-max 8 accepts 100% but verification cost outweighs it; 5 is the sweet spot.

## Benchmark findings (B60 / Vulkan)

Full raw outputs + environment: **[RESULTS.md](RESULTS.md)**.

Scripts: `bench/bench.py` (B60 vs Halogen) and `bench/variants.py` (quant-variant
shootout). Methodology per prompt: a unique nonce forces a **cold prefill** call
(`max_tokens=1`), then warm (cached-prefill) decode is averaged over 3 reps at
`K=256` via `/v1/chat/completions` (what opencode actually uses).

### Quant variants from the repo (on the B60)

Measured with the no-ReBAR fix + `-c 32768` (earlier small-context runs were
cliff-contaminated — see below):

| Variant | Quant | MTP head | Decode t/s | Verdict |
|---|---|:---:|---:|---|
| `…IQ3_S-…MTP-Q4_K` (**dense track**) | Q4_K | yes | **~27–35** | Best dense |
| `…MTP-3.2BPW` | ~3.2 bpw | yes | ~10 | 3× slower — slow dequant path |
| `…Ridge-…Q4_K` | Q4_K | **no** | ~14–15 (ngram-only) | No MTP head; not competitive |

- **3.2BPW:** ~10 t/s even with the cliff fix — this quantization genuinely has
  a slow Vulkan dequant path. (The originally reported 0.84 t/s was a
  small-context cliff artifact.)
- **Ridge Q4_K has no MTP head**, so it can't use the speculative boost;
  ~14–15 t/s with ngram-only. (Originally reported 3–5 t/s — also cliff.)

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
and SYCL needs the ReBAR this Thunderbolt link doesn't expose. **OpenVINO is
likewise rejected**: its GPU plugin is Level-Zero-only (probe shows `['CPU']`
devices on this rig), and the CPU fallback is bandwidth-bound — measured ~4.6 t/s
at 7B int4, extrapolating to ~1.5 t/s at 27B vs 27–35 t/s on Vulkan
(`bench/ov_probe.py`, details in RESULTS.md). All tuning is llama.cpp/Vulkan.
Measured (decode t/s, current Q4_K):

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
  moe_probe.py         MoE candidates + corrected variant retests
  ov_probe.py          OpenVINO feasibility probe (device list + CPU decode)
```

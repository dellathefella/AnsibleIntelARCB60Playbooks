# Lenovo P53 / Arc B60 — SYCL LLM Inference Playbooks

Ansible playbooks that serve LLMs on an **Intel Arc Pro B60** over
**Thunderbolt** using `llama.cpp` (SYCL / Level-Zero) in a rootful **Podman**
container. One GPU, one track at a time.

- **Backend**: SYCL only — **Resizable BAR (ReBAR) is mandatory** and the gate
  fails the play without it (no Vulkan fallback: without ReBAR performance is
  abysmal).
- **Runs as root**: all plays are `become: true` (rootful podman).
- **Convention**: see [SYSTEM.md](SYSTEM.md) — topology-first layout, no
  `roles/`, self-contained per-track playbooks, INI inventory, shared
  `tasks/` + `templates/`.

## Tracks (single-node)

| Track playbook | Model | Container | Port | Ctx |
| --- | --- | --- | --- | --- |
| `qwen38-27b-sycl-podman.yml` | Qwen3.8-27B (IQ3_S→Q4_K tuned GGUF + MTP, ~14.3 GB) | `qwen38-27b-sycl` | 8183 | 200000 |
| `qwen36-a3b-moe-sycl-podman.yml` | Qwen3.6-35B-A3B MoE (UD-Q4_K_S + MTP, ~19.9 GiB) | `qwen36-a3b-moe` | 8184 | 32768 |
| `qwen38-27b-vllm-xpu-podman.yml` | Qwen3.8-27B (W4A16 AWQ, vLLM XPU) | `qwen38-27b-vllm-xpu` | 8185 | 32000 |
| `qwen38-27b-vllm-xpu-tp2-podman.yml` | Qwen3.8-27B (W4A16 AWQ, vLLM XPU, TP=2, 2x B60) | `qwen38-27b-vllm-xpu-tp2` | 8188 | 262144 |
| `qwen36-a3b-escha-w2-sycl-podman.yml` | Qwen3.6-35B-A3B Escha W2 (IQ2_XXS + MTP, ~9.7 GiB) | `qwen36-a3b-escha-w2` | 8187 | 262144 |

Each track exposes an OpenAI-compatible API (`/v1/*`, `/health`) on the LAN and
renders:

- a manual launch script on the GPU host: `~/scripts/<track_stem>-start.sh`
- an opencode config on the controller: `ansible/single-node/rendered/opencode-configs/`

## Requirements

- An Arc B60 with **ReBAR enabled** (≥ 1 GiB prefetchable BAR) — e.g. halo1
  (Strix Halo desktop, 32 GiB BAR over TB4).
- Root/sudo access on the GPU host (feed the password with `-K` or a gitignored
  `group_vars/all/become.yml`).
- The `community.general` collection (Arch/pacman tasks):
  `ansible-galaxy collection install community.general`.

## Quick start

```bash
# from the repo root
ansible-playbook -i ansible/single-node/inventory/hosts \
  ansible/single-node/qwen38-27b-sycl-podman.yml \
  -e ansible_become_pass='<sudo-password>'

# everything (shared setup + both tracks — last track run wins the GPU)
ansible-playbook -i ansible/single-node/inventory/hosts \
  ansible/single-node/bootstrap.yml

# base provisioning only
ansible-playbook -i ansible/single-node/inventory/hosts \
  ansible/single-node/bootstrap.yml --tags install-podman,install-hf-cli
```

The play targets the `sycl` inventory group; edit
`ansible/single-node/inventory/hosts` for your rig (see `hosts.example`).

## Security notes

- The API is published on the **LAN without authentication** — run this on a
  trusted network only.
- `ansible/secrets/hf_token.txt` is gitignored; put a token there only if you
  need gated HF repos. The `PUT_HF_TOKEN_HERE` placeholder is committed.
- The rendered output dir (`ansible/single-node/rendered/`) is gitignored.

## Benchmarks

See [benchmarks/RESULTS.md](benchmarks/RESULTS.md). Highlights on halo1
(32 GiB ReBAR, dense 27B track):

- ~36–46 t/s decode, ~614–660 t/s long-prefill (short context)
- ~17–20 t/s decode, ~469 t/s prefill at 128k context
- OpenVINO on this card is CPU-only without ReBAR and was rejected
  (~1.3–1.6 t/s at 27B)

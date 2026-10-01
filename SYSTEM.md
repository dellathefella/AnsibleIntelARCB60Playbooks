# SYCL Inference Bootstrap — Conventions

## Target platform

- **GPU**: Intel Arc Pro B60 (Xe2), driven by the `xe` kernel driver (render node
  `/dev/dri/renderD*`).
- **Backend**: SYCL / Level-Zero via the prebuilt `ghcr.io/ggml-org/llama.cpp:full-intel`
  image (pulled, never built).
- **ReBAR is mandatory.** The gate (`tasks/rebar-gate.yml`) fails the play unless the
  largest prefetchable PCI BAR is ≥ 1 GiB. Without Resizable BAR the Level-Zero
  device cannot be created and the Vulkan fallback performs abysmally (legacy
  256 MiB BAR rigs cannot run any track). There is no Vulkan fallback and no
  opt-out.
- **Rootful podman.** Every play runs `become: true`; containers are launched
  rootful with `--security-opt label=disable`. Feed the sudo password with `-K`
  or a gitignored `group_vars/all/become.yml` (`ansible_become_pass`).

## Directory layout

The repo is organized **topology-first**. There is **no** `roles/`, no
`host_vars/`, and no per-role template dirs — shared pieces live under
`shared/` (host-level plays) and each topology's own `tasks/` / `templates/`.

```
ansible/
  shared/                      host-level setup plays (imported by the bootstrap)
    install-podman.yml           cross-distro Podman install (Fedora/Debian/Arch)
    install-hf-cli.yml           HuggingFace CLI install
  single-node/                 one B60 box, one model at a time
    bootstrap.yml                ORCHESTRATOR: shared/ + all single-node tracks
    summary.yml                  final per-host completion summary
    qwen38-27b-sycl-podman.yml   dense 27B track (port 8183)
    qwen36-a3b-moe-sycl-podman.yml  MoE 35B-A3B track (port 8184)
    tasks/                       shared task files included by the tracks:
      rebar-gate.yml                 render node + PCI id + ReBAR window (mandatory)
      sycl-probe.yml                 sycl-ls device check inside the image
      podman-models-dir.yml          models dir + ownership
      hf-download-files.yml          HF download loop (skips files already present)
      model-verify-sha.yml           size + sha256 check (sha == HF LFS oid)
      podman-stop-sibling.yml        stop the other track's container (one GPU)
      podman-remove-container.yml    podman rm -f before (re)launch
      podman-check-running.yml       start result + running check
      podman-wait-health.yml         sleep + poll /health
      podman-render-launch-artifacts.yml  launch script + opencode config render
    inventory/
      hosts                      real inventory (sycl / legacy → aiservers)
      hosts.example              sample inventory template
      group_vars/all.yml         placeholder (empty) — tracks define vars inline
    templates/
      scripts/<track_stem>-start.sh.j2                 launch script template
      opencode-configs/opencode-<track_stem>-podman.json.j2  opencode config template
    rendered/                    rendered output (gitignored)
  opencode-configs/            committed example opencode configs (controller-side)
  secrets/                     hf_token.txt (gitignored) + PUT_HF_TOKEN_HERE placeholder
```

## Track naming convention

### File naming

Playbooks use the **descriptive model + backend name**. Plays target a
**capability group** (GPU backend) rather than one model — `aiservers` is the
union of all inference hosts, with `sycl` underneath. A host runs one model at
a time, so the model is chosen by which track playbook you run, not by group
membership.

| Playbook | Play targets (`hosts:`) | Track tag (`tags:`) |
| --- | --- | --- |
| `qwen38-27b-sycl-podman.yml` (Qwen3.8-27B IQ3_S→Q4_K + MTP, 200k ctx, port 8183) | `sycl` | `qwen38-27b-sycl-podman` |
| `qwen36-a3b-moe-sycl-podman.yml` (Qwen3.6-35B-A3B UD-Q4_K_S + MTP, 32k ctx, port 8184) | `sycl` | `qwen36-a3b-moe-sycl-podman` |

### Template naming

Templates live under `single-node/templates/` and are rendered by
`tasks/podman-render-launch-artifacts.yml` using `track_stem`:

- Launch script: `templates/scripts/<track_stem>-start.sh.j2` → renders to the
  **target host's** `~/scripts/<track_stem>-start.sh`
- OpenCode config: `templates/opencode-configs/opencode-<track_stem>-podman.json.j2`
  → renders to the **controller's** `rendered/opencode-configs/opencode-<track_stem>-podman.json`

Note the `-podman` suffix on the opencode config template/filename (the
`track_stem` itself does not carry it).

### Variable conventions

Each track playbook is **self-contained** — it defines its own `vars:` block and
uses **no `group_vars`**. Common per-track vars:

- `track_stem` — model stem; basis for the container name and rendered
  script/config names
- `docker_image` — prebuilt image to pull
- `hf_repo` / `model` / `model_sha256` / `model_size` — model file, HF repo, and
  integrity checks (sha == HF LFS oid)
- `host_port` / `ctx` — host port and context window
- `_user_home` / `_models_dir` — directories; derived from `ansible_user`,
  **not** `ansible_env.HOME` (under `become: true`, HOME is `/root`)

## Playbook structure

### Single-play pattern (Podman tracks)

Each `*-podman.yml` track is a **single play** (`hosts: sycl`,
`gather_facts: true`, `become: true`) that does everything in order by including
shared task files from `tasks/`:

1. `rebar-gate.yml` — resolve render node + PCI id, assert ReBAR window ≥ 1 GiB
2. `sycl-probe.yml` — `sycl-ls` inside the image must show `level_zero:gpu`
3. `podman-models-dir.yml` — ensure the models dir exists + is user-owned
4. `hf-download-files.yml` — download weights (skips files already present)
5. `model-verify-sha.yml` — size + sha256 check
6. `podman-stop-sibling.yml` — stop the other track's container (one GPU, one track)
7. `podman-remove-container.yml` → `podman run` → `podman-check-running.yml`
8. `podman-wait-health.yml` — sleep + poll `/health`
9. `podman-render-launch-artifacts.yml` — render the launch script to the
   target's `~/scripts/` and the opencode config to the controller's
   `rendered/opencode-configs/`

The render tasks use `delegate_to: localhost` + `become: false` **inline** —
there is no separate render play.

### Tags

- Play: `[<track>-podman]` (e.g. `qwen38-27b-sycl-podman`)
- Gate: `[podman, gate]`
- Models dir + download: `[podman, model]`
- Container run + health: `[podman, deploy]`
- Render artifacts: `[podman, launch]`

Run one whole track with its play tag: `--tags <track>-podman`. The shared host
plays carry their own tags (`install-podman`, `install-hf-cli`), so
`--skip-tags install-podman,install-hf-cli` skips base provisioning on an
already-built host.

## ReBAR gate

`tasks/rebar-gate.yml` is mandatory and has no opt-out. It:

1. Finds the Intel render node (driver `xe`, fallback `i915`) by scanning
   `/sys/class/drm/renderD*/device/uevent`.
2. Resolves its PCI id via `readlink -f /sys/class/drm/<node>/device`.
3. Reads `/sys/bus/pci/devices/<pci>/resource` and measures the largest BAR with
   both `IORESOURCE_MEM` (0x200) and `IORESOURCE_PREFETCH` (0x2000) set — note the
   prefetch flag is **0x2000**, not 0x1000.
4. Fails the play unless that BAR is ≥ 1 GiB (2 GiB typical with ReBAR enabled).

## SYCL environment

The container is launched with exactly these env vars (device selection inside
`/dev/dri`, which also carries the host's own iGPU on Strix Halo hosts):

- `UR_LOADER_USE_LEVEL_ZERO_V2=0` — use the v1 Level-Zero plugin
- `ONEAPI_DEVICE_SELECTOR=level_zero:0` — first (and only) Xe2 device
- `ZE_AFFINITY_MASK=0` — no tile affinity masking

## Ansible gotchas

### become on local runs

Plays run rootful. For local (controller-side) targets use a NOPASSWD sudoers
rule or feed `-K`. For SSH targets (halo1) feed `-e ansible_become_pass=...`
or a gitignored `group_vars/all/become.yml`.

### HOME under become

Under `become: true`, `ansible_env.HOME` is `/root`. All user-facing paths are
built from `/home/{{ ansible_user }}` instead.

### Model integrity

`model_sha256` equals the HuggingFace LFS oid of the file, so the sha check
proves a complete download. The download task itself is resumable via `hf`.

## Adding a new track

1. Copy an existing `*-podman.yml` as the new track; set `track_stem`,
   `docker_container_name`, `sibling_container`, model vars, `host_port`, `ctx`.
2. Add `templates/scripts/<track_stem>-start.sh.j2` and
   `templates/opencode-configs/opencode-<track_stem>-podman.json.j2`.
3. Import it in `bootstrap.yml` with its tag.
4. Syntax-check: `ansible-playbook --syntax-check -i ansible/single-node/inventory/hosts ansible/single-node/bootstrap.yml`

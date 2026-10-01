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

## Layout

```
site.yml                     entrypoint
ansible.cfg
inventory/hosts.yml          b60 host (local now, ssh later)
group_vars/all.yml           ALL tunables
roles/b60_llama/
  tasks/{main,preflight,model,service,opencode}.yml
  templates/{run-llama-b60.sh.j2, opencode.json.j2}
```

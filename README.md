# LM Atelier

LM Atelier is a local-first AI workspace for chat, contextual image generation,
and contextual video generation. A single browser conversation can decide
whether a request needs text, an image, or a video, run the selected local
engine, and render the result inline.

The repository contains a working `0.1.0` foundation with mock engines for a
weight-free demo and adapters for `llama-server`, ComfyUI, and Hugging Face.

## What works

- Persistent projects, chats, branched message history, and ordered mixed-media
  message parts in SQLite.
- Auto/Text/Image/Video routing, explicit modality overrides, contextual image
  references, durable streaming, cancellation, and restart reconciliation.
- Inline image/video results, accessible video controls, HTTP byte ranges, browser-compatible proxies, and expensive-video estimates.
- Swappable chat/image/video profiles with Basic, Advanced, and Expert controls.
- Hugging Face search and sorting, compatibility labels, exact file selection,
  disk preflight, resumable cache use, and atomic model activation.
- Versioned ComfyUI workflow import, validation, declared inputs, and immutable
  run provenance.
- Content-addressed media library, versioned project export/import, retention-safe
  cleanup, verified state/media backups, and redacted diagnostics.
- Loopback binding, authenticated same-origin sessions, CSRF protection, safe
  model formats, and no telemetry.

Mock engines exercise the entire product without model weights. Real inference
requires separately installed runtime binaries and compatible models/workflows.

## Reference recipes

The Model library includes curated, one-click recipes pinned to immutable Hub
commits and exact files. Downloads are checked against recorded SHA-256 hashes,
and the managed ComfyUI worker receives generated extra-model-path mappings.

| Role | Reference recipe | Download | Starting hardware | Operations |
| --- | --- | ---: | --- | --- |
| Chat | Qwen3 8B Q4_K_M | 5.0 GB | CPU, 8 GB RAM; 16 GB recommended | Text |
| Image | FLUX.1 Schnell FP8 | 17.2 GB | 12 GB VRAM; 16 GB recommended | Text/image to image |
| Video | Wan 2.1 T2V 1.3B | 9.8 GB | 8 GB VRAM; 12 GB recommended | Text to 480p video |
| Video | Wan 2.1 I2V 14B 480p FP8 | 24.7 GB | 20 GB VRAM; 24 GB recommended | Image to 480p video |

These entries are deliberately labeled **reference candidates**. Their source,
license, defaults, dependencies, and expected hashes are verified, but they are
not labeled certified until the complete recipe passes repeatable generation
tests on declared target hardware. Media recipes use official Comfy-Org
safe-tensor packages and native ComfyUI nodes; LM Atelier does not install custom
node code. Model and workflow licenses remain independent of LM Atelier's license.

The library reports installed, catalog-cache, and incomplete-download storage.
Orphaned staging directories can be cleaned from the interface. Deletion is
blocked while a profile references a model, and multiple installs that share a
revision directory retain files referenced by the remaining install records.

Catalog results support cursor-based “load more” navigation and filters for
compatibility tier, file format, quantization, architecture, license, and gated
access, plus parameter-count and download-size bounds. Before a download is
queued, an install preflight reports exact file selection, revision mutability,
license and gated access, unsafe weight formats, disk fit, runtime compatibility,
and estimated RAM or VRAM fit. Existing local files and directories can also be
registered through the Advanced Import dialog. Search metadata is normalized
from Hub tags, config, and file manifests;
the Compatible-first option orders each fetched page by LM Atelier's explicit
compatibility classification before source download counts.

## Target platform matrix

The initial release targets Windows 11 x64 and Ubuntu 24.04 LTS x64. NVIDIA
CUDA is the primary image/video accelerator, with 12 GB, 16 GB, and 24 GB VRAM
validation tiers. CPU inference is the universal chat fallback on both target
operating systems. Apple Silicon/Metal and Linux AMD/ROCm are experimental until
dedicated machines and repeatable runtime evidence are available.

| Platform | Status | Chat | Image/video | Current evidence |
| --- | --- | --- | --- | --- |
| Windows 11 x64 + NVIDIA CUDA | Target | CPU or CUDA | 12/16/24 GB VRAM tiers | Windows CI; GPU tests pending |
| Ubuntu 24.04 LTS x64 + NVIDIA CUDA | Target | CPU or CUDA | 12/16/24 GB VRAM tiers | Ubuntu CI; GPU tests pending |
| Windows 11/Ubuntu 24.04 x64 CPU | Target fallback | Supported | Not a reference media target | Automated tests; performance tests pending |
| macOS Apple Silicon + Metal | Experimental | Experimental | Experimental | Dedicated runner pending |
| Linux x64 + AMD ROCm | Experimental | Experimental | Experimental | Dedicated runner pending |

The Settings page compares the current machine with this matrix. “Target” means
the platform is in scope; it does not mean a hardware tier is certified. A tier
remains `hardware-pending` until real llama.cpp and ComfyUI generation tests pass
and their runtime, driver, model recipe, timings, and output checks are recorded.

## Quick start

Requirements: Python 3.12+, Node.js 22+, npm, and optionally FFmpeg for mock
video output.

```bash
./scripts/setup.sh
./scripts/start.sh
```

Open `http://127.0.0.1:12340`. For frontend/backend hot reload, use
`./scripts/dev.sh` and open the Vite URL it prints.

Official release archives include the built web interface, so end users do not
need Node.js. On Linux, run `packaging/linux/install.sh`; on Windows, run
`packaging/windows/install.ps1`. Updates install side by side, rollback switches
the active application version without touching data, and uninstall preserves
data unless its explicit purge option is used. Maintainers can build both archive
formats and checksums with `./scripts/package.sh`.

Run every local quality gate with:

```bash
./scripts/verify.sh
```

Application state defaults to `./data` and is ignored by Git. Set
`LOCAL_LM_DATA_DIR` to relocate it.

## Real chat with llama.cpp

1. Install a compatible `llama-server` build separately.
2. Import or download a GGUF model in the Model library.
3. Create a profile for that installed model.
4. Set these values in `.env`:

```dotenv
LOCAL_LM_CHAT_ENGINE=llama.cpp
LOCAL_LM_LLAMA_EXECUTABLE=/absolute/path/to/llama-server
LOCAL_LM_LLAMA_URL=http://127.0.0.1:12341
```

The Settings page can load and unload managed chat profiles. LM Atelier passes
load settings as an argument array without a shell and only swaps after the
worker health check succeeds. If you run `llama-server` yourself, omit the
executable and point `LOCAL_LM_LLAMA_URL` at it.

Managed worker cards distinguish starting, ready, stopped, and exited states;
show active and queued generation counts; and sample current and peak RAM for
the complete worker process tree. Chat cards also show a conservative pre-load
estimate based on GGUF size and context length. CPU/GPU placement can change the
actual result, so estimates and measurements are labeled separately.

The Engines panel can run an executable structured-tool probe against the active
chat adapter. It supplies a function with a strict routing schema, assembles
streamed tool-call fragments, validates the returned JSON arguments, and reports
the observed result separately from the engine's advertised capability.

Profiles and generation presets can be edited at Basic, Advanced, or Expert
visibility, cloned, reset, and exported as versioned JSON bundles. Imported
profiles keep their model binding when the same install exists locally and
otherwise import safely as unbound profiles so they can be attached to a model
on the destination machine.

The default preset for a generation role is applied automatically. Settings
resolve from capability defaults, then the selected model profile, the role's
default preset, and finally per-turn overrides. Load-time controls remain
profile-only because changing them requires replacing the worker.

Before each chat run, LM Atelier asks llama.cpp for the chat-template-aware
input token count. When a conversation exceeds the selected profile's context
window, it preserves project instructions and the newest request while omitting
the oldest turns until the prompt fits. The response shows context usage and
any omission count, and the complete policy is retained in run provenance.

Assistant text streams as Markdown with tables, task lists, links, and fenced
code. The composer becomes a stop control while a run is active. Stopping keeps
the text received so far and marks it with subdued cancellation metadata. If a
chat stream fails after producing text, that partial response remains visible
with the failure shown beneath it. When llama.cpp resets a fixed-seed stream
before its terminal frames, LM Atelier retries once and only joins the missing
suffix after verifying that the regenerated prefix is identical.
Editing a user message creates a new branch from that point, preserves the
source run's modality and settings, and displays only the newly selected branch
while keeping the prior branch in local history.

## Real images and video with ComfyUI

1. Install ComfyUI separately and prepare API-format workflows whose tunable
   values use `${prompt}`, `${negative_prompt}`, `${seed}`, and other declared
   parameter placeholders.
2. Import the workflow in the Workflows page and validate it against the active
   ComfyUI node inventory.
3. Configure either an external server or the managed worker:

```dotenv
LOCAL_LM_MEDIA_ENGINE=comfyui
LOCAL_LM_COMFY_URL=http://127.0.0.1:8188
LOCAL_LM_COMFY_EXECUTABLE=/absolute/path/to/python
LOCAL_LM_COMFY_DIRECTORY=/absolute/path/to/ComfyUI
```

Custom nodes are executable code. LM Atelier stores trust metadata but does not
install or execute a custom node on a user's behalf.

Pinned Wan recipes also download the official example workflow JSON into their
model install directory. Review it in ComfyUI, export it in API format, then
import and trust that graph in LM Atelier. Recipe defaults remain editable through
the same Basic, Advanced, and Expert generation controls as other profiles.

## Configuration

Copy `.env.example` to `.env`. Important settings include:

| Variable | Default | Purpose |
| --- | --- | --- |
| `LOCAL_LM_DATA_DIR` | `./data` | Database, artifacts, models, logs, exports, backups |
| `LOCAL_LM_HOST` | `127.0.0.1` | Bind address |
| `LOCAL_LM_ALLOW_LAN` | `false` | Required before a non-loopback bind is accepted |
| `LOCAL_LM_CHAT_ENGINE` | `mock` | `mock`, `llama.cpp`, or an installed adapter entry point |
| `LOCAL_LM_MEDIA_ENGINE` | `mock` | `mock`, `comfyui`, or an installed adapter entry point |
| `LOCAL_LM_HF_TOKEN` | unset | Optional process-only fallback for gated/private Hub access |
| `LOCAL_LM_ARTIFACT_RETENTION_DAYS` | `30` | Recovery window for unreferenced artifacts |
| `LOCAL_LM_TEMPORARY_RETENTION_HOURS` | `24` | Lifetime of temporary/intermediate media |
| `LOCAL_LM_BACKUP_DAILY_COUNT` | `7` | Daily state snapshots to retain |
| `LOCAL_LM_BACKUP_WEEKLY_COUNT` | `4` | Older weekly state snapshots to retain |

For regular use, save the Hugging Face token from Settings. LM Atelier stores it
in the operating system credential vault, never echoes it back to the browser,
and excludes it from SQLite, logs, exports, backups, and diagnostics. The
`LOCAL_LM_HF_TOKEN` environment variable is a process-only fallback and takes
precedence while set.

## Data and recovery

- Chats and configuration: `data/state/local-lm.sqlite3`
- Generated/imported artifacts: `data/artifacts/sha256`
- Installed models: `data/models`
- Incomplete downloads: `data/downloads`
- Verified backups: `data/backups`

Model downloads can be paused, resumed, or cancelled from the live jobs panel.
Transfers use partial directories, preserve resumable files while paused, and
automatically requeue jobs interrupted by an orderly or unexpected shutdown.

Database restore is staged rather than performed underneath a running API. Use
the backup endpoint/UI to select a verified snapshot, then restart LM Atelier.
Backups keep seven daily and four older weekly versions and can optionally carry
checksum-verified media. Project `.lm-atelier.zip` archives use a versioned JSON
manifest and can include hash-addressed media or metadata only. The import path
validates archive boundaries, sizes, types, and content hashes before creating
new local identities.

The Settings page can download a redacted diagnostic bundle. It contains machine,
capability, aggregate database, model-role, workflow-operation, and job-state
facts, but excludes prompts, chat content, media, credentials, absolute paths,
and log contents.

Runtime releases used for compatibility evaluation are pinned in
`packaging/engines.json`. llama.cpp and ComfyUI remain separately installed and
retain their own licenses; their pinned entries are compatibility targets rather
than bundled binaries or hardware certification claims.

The Workflow studio imports and exports portable LM Atelier workflow bundles,
keeps executable and UI graphs in immutable revisions, restores old versions as
new revisions, and validates declared model, engine-version, node, and device
requirements before execution. JSON-schema input declarations render as tunable
controls without changing the stored ComfyUI graph.

## Development

The web client lives in `apps/web`; the FastAPI package lives in `services/api`.
Normal CI runs the web and API suites on both Ubuntu and Windows using mock
engines and never downloads model weights.

Development flows through descriptively named work branches into `develop`,
then from `develop` into `main` for releases. Feature branch names never use a
`feature/`, `feature-`, or `feat/` prefix. See
[CONTRIBUTING.md](CONTRIBUTING.md).

Third-party runtime packages can integrate through the versioned
[adapter contract](docs/ADAPTERS.md) and run the same conformance probes as the
built-in mock adapters.

LM Atelier is licensed under the [Apache License 2.0](LICENSE). Optional runtimes,
models, workflows, and custom nodes retain their own licenses. ComfyUI is a
separately installed GPL-3.0 runtime and is not part of this Apache-licensed
distribution.

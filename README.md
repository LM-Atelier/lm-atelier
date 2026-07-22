# Local LM

Local LM is a local-first AI workspace for chat, contextual image generation,
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
- Inline image/video results, accessible video controls, and HTTP byte ranges.
- Swappable chat/image/video profiles with Basic, Advanced, and Expert controls.
- Hugging Face search and sorting, compatibility labels, exact file selection,
  disk preflight, resumable cache use, and atomic model activation.
- Versioned ComfyUI workflow import, validation, declared inputs, and immutable
  run provenance.
- Content-addressed artifacts, project ZIP exports, verified SQLite snapshots,
  and restore-on-next-start.
- Loopback binding, authenticated same-origin sessions, CSRF protection, safe
  model formats, and no telemetry.

Mock engines exercise the entire product without model weights. Real inference
requires separately installed runtime binaries and compatible models/workflows.

## Quick start

Requirements: Python 3.12+, Node.js 22+, npm, and optionally FFmpeg for mock
video output.

```bash
./scripts/setup.sh
./scripts/start.sh
```

Open `http://127.0.0.1:12340`. For frontend/backend hot reload, use
`./scripts/dev.sh` and open the Vite URL it prints.

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

The Settings page can load and unload managed chat profiles. Local LM passes
load settings as an argument array without a shell and only swaps after the
worker health check succeeds. If you run `llama-server` yourself, omit the
executable and point `LOCAL_LM_LLAMA_URL` at it.

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

Custom nodes are executable code. Local LM stores trust metadata but does not
install or execute a custom node on a user's behalf.

## Configuration

Copy `.env.example` to `.env`. Important settings include:

| Variable | Default | Purpose |
| --- | --- | --- |
| `LOCAL_LM_DATA_DIR` | `./data` | Database, artifacts, models, logs, exports, backups |
| `LOCAL_LM_HOST` | `127.0.0.1` | Bind address |
| `LOCAL_LM_ALLOW_LAN` | `false` | Required before a non-loopback bind is accepted |
| `LOCAL_LM_CHAT_ENGINE` | `mock` | `mock` or `llama.cpp` |
| `LOCAL_LM_MEDIA_ENGINE` | `mock` | `mock` or `comfyui` |
| `LOCAL_LM_HF_TOKEN` | unset | Optional gated/private Hub access token |

For regular use, keep credentials out of `.env` and launch Local LM with the
token supplied by the operating system's credential facility. Tokens are not
stored in SQLite or returned by diagnostics.

## Data and recovery

- Chats and configuration: `data/state/local-lm.sqlite3`
- Generated/imported artifacts: `data/artifacts/sha256`
- Installed models: `data/models`
- Incomplete downloads: `data/downloads`
- Verified backups: `data/backups`

Database restore is staged rather than performed underneath a running API. Use
the backup endpoint/UI to select a verified snapshot, then restart Local LM.
Project exports are ordinary ZIP files with a versioned JSON manifest and
hash-addressed media.

## Development

The web client lives in `apps/web`; the FastAPI package lives in `services/api`.
Normal CI uses mock engines and never downloads model weights.

Development flows through descriptively named work branches into `develop`,
then from `develop` into `main` for releases. Feature branch names never use a
`feature/`, `feature-`, or `feat/` prefix. See
[CONTRIBUTING.md](CONTRIBUTING.md).

Local LM is licensed under the [Apache License 2.0](LICENSE). Optional runtimes,
models, workflows, and custom nodes retain their own licenses. ComfyUI is a
separately installed GPL-3.0 runtime and is not part of this Apache-licensed
distribution.

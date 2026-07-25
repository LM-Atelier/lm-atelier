# LM Atelier

LM Atelier is a local-first workspace for chat, image generation, and video
generation. It keeps conversations and generated media on your computer and can
choose the appropriate local model automatically.

## Install

### Windows

Download `LM-Atelier-Setup-<version>-windows-x86_64.exe` from
[Releases](https://github.com/ajccarlson/lm-atelier/releases/latest), run the
installer, then open **LM Atelier** from the Start menu, desktop shortcut, or
installation folder. Keep the LM Atelier terminal open while using the app;
closing it stops the local service.

### Linux

Download `LM-Atelier-Setup-<version>-linux-x86_64.run` from
[Releases](https://github.com/ajccarlson/lm-atelier/releases/latest), then run:

```bash
chmod +x LM-Atelier-Setup-*-linux-x86_64.run
./LM-Atelier-Setup-*-linux-x86_64.run
```

The installers are self-contained and do not require Python, Node.js, or
administrator/root access. Model weights and optional inference engines are
installed separately. Verify downloads with the release's `SHA256SUMS` file;
release binaries are not yet code-signed.

## Run from source

Source development requires Python 3.11 or newer, Node.js 22 or newer, and npm.

On Linux or in Git Bash:

```bash
./scripts/setup.sh
./scripts/start.sh
```

On Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e '.\services\api[dev]'
npm.cmd ci
npm.cmd run build
.\.venv\Scripts\lm-atelier.exe
```

Open `http://127.0.0.1:12340`. The default mock engines let developers run the
complete interface without downloading models.

## Development

The React web client is in `apps/web`; the FastAPI service is in `services/api`.
Copy `.env.example` to `.env` only when you need to override defaults. Application
state is stored in `data` by default; set `LOCAL_LM_DATA_DIR` to use another
location.

Run the complete local verification suite before opening a pull request:

```bash
./scripts/verify.sh
```

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\verify.ps1
```

Work flows from a descriptively named branch into `develop`, then from `develop`
into `main`. See [CONTRIBUTING.md](CONTRIBUTING.md) for the short contribution
guide, [SECURITY.md](SECURITY.md) for private vulnerability reporting, and
[docs/ADAPTERS.md](docs/ADAPTERS.md) for third-party runtime integration.

LM Atelier is licensed under the [Apache License 2.0](LICENSE). Models, workflows,
inference engines, and custom nodes retain their own licenses.

# LM Atelier

LM Atelier is a local-first workspace for chat, image generation, and video
generation. It stores conversations and generated media on your computer by
default, and Auto Mode can choose an appropriate configured local model for
each request.

> [!IMPORTANT]
> LM Atelier is a preview. The installers target Windows 11 x64 and Ubuntu 24.04
> LTS x86_64. Installer support does not mean every model, runtime, GPU, or
> workflow is certified; physical hardware certification is still in progress.

![LM Atelier empty local workspace](docs/assets/application-preview.png)

## Install

Download the current files from
[Releases](https://github.com/ajccarlson/lm-atelier/releases/latest).

### Windows

Run `LM-Atelier-Setup-<version>-windows-x86_64.exe`, then open **LM Atelier**
from the Start menu, desktop shortcut, or installation folder. Keep the
LM Atelier terminal open while using the app; closing it stops the local service.

### Linux

Run `LM-Atelier-Setup-<version>-linux-x86_64.run`:

```bash
chmod +x LM-Atelier-Setup-*-linux-x86_64.run
./LM-Atelier-Setup-*-linux-x86_64.run
```

The installers are self-contained and do not require Python, Node.js, or
administrator/root access. Models and optional inference engines are separate
downloads with their own hardware, storage, and license requirements. Managed
llama.cpp chat setup is one-click on both installer targets where the pinned
runtime is compatible. Managed media setup is currently limited to the reviewed
compatible Windows NVIDIA runtime. Linux image/video require an externally
configured compatible media engine and are not certified. Internet access is
needed for catalog, model, and engine downloads.

Verify the installers with the release's `SHA256SUMS` file. Release binaries are
not yet code-signed. Release notes distinguish automated build smoke tests from
physical hardware tests; only completed recorded matrices are called certified.

## Run from source

Development requires Python 3.12, Node.js 22.13 or newer, and npm. The
full cross-platform verification gate also needs Git for Windows (including Git
Bash) on Windows, or PowerShell 7 (`pwsh`) on Linux.

On Linux:

```bash
./scripts/setup.sh
./scripts/start.sh
```

On Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e '.\services\api[dev,package]'
npm.cmd ci
npm.cmd run build
.\.venv\Scripts\lm-atelier.exe
```

Open `http://127.0.0.1:12340`. Mock engines let contributors use the interface
without downloading models: copy `.env.example` to `.env` and leave its two
engine values set to `mock`. Do not put credentials in a shared `.env`.

Run the local verification gate before opening a pull request:

```bash
./scripts/verify.sh
```

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\verify.ps1
```

## Project information

- [Troubleshooting](docs/TROUBLESHOOTING.md)
- [Privacy and local data](docs/PRIVACY.md)
- [Support](SUPPORT.md)
- [Contributing](CONTRIBUTING.md)
- [Security policy](SECURITY.md)
- [Architecture](docs/ARCHITECTURE.md)
- [Runtime adapters](docs/ADAPTERS.md)

LM Atelier is licensed under the [Apache License 2.0](LICENSE). Models,
workflows, inference engines, and custom nodes retain their own licenses.
Copyright 2026 LM Atelier contributors.

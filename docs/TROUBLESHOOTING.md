# Troubleshooting

New install? Start with [Getting started](GETTING-STARTED.md), which explains
the order setup works in and what "Ready" means.

## Setup will not finish

The setup panel shows the first unsatisfied check for a role. What each state
means:

- **No active model is installed for this role.** Nothing is installed yet.
  Choose a model, or use the recommended one if the panel offers it.
- **The required runtime is not installed.** The engine downloads automatically
  the first time it is needed; **Install runtime** starts it now. Confirm free
  disk space first: ComfyUI needs about 2 GB to download and 8 GB free to
  install.
- **The required runtime did not start or install.** The download or extraction
  failed. Retry it. If it keeps failing, check free disk space and confirm that
  security software is not quarantining the extracted files.
- **Automatic setup for the required runtime is unavailable on this machine.**
  This role cannot run here, and the message says why. Image and video need
  Windows with a compatible NVIDIA GPU. No action is offered because none would
  help; see the platform table in [Getting started](GETTING-STARTED.md).
- **The model has not passed an activation probe**, or **must be rechecked for
  the current runtime and hardware.** The files are present, but the model has
  not produced output under the current setup, so it is not usable yet. This
  also appears after an update that changes the runtime or workflow contract.
  Use **Re-check model**: it loads the model again and asks it for one small
  output, without re-downloading anything. You do not need to reinstall.
- **The model did not pass its activation probe.** It loaded but failed to
  generate. Usually the model does not fit in available memory, or its format is
  not supported by the installed runtime. Try a smaller model or a smaller
  quantization.
- **No compatible workflow is installed for this model.** The media runtime has
  no workflow that can drive this model. Confirm the ComfyUI runtime finished
  installing; a partial install exposes no workflows.
- **The managed worker exited unexpectedly.** The engine process stopped.
  Restart the worker from the panel; if it exits again, read the worker error
  shown on its card in Settings.

A role is only **Ready** after a real local generation completes. **Not runtime
verified** in the model library means the same thing: installed, not yet proven.

## Installing a model

- **You are asked to confirm before a download starts.** The confirmation shows
  the download size, your free space, and the memory the model needs to load. If
  it says the model needs more accelerator memory than your machine reports, it
  will either run slowly on the processor or fail to load; a smaller model or a
  smaller quantization is usually the better choice.
- **A recommended model refuses to install.** Recommended models pin exact file
  checksums, and the install refuses if the repository no longer matches them -
  for example if files were replaced upstream. This is deliberate: it will not
  install something other than what was verified. Choose another model, or
  install from the catalog where you can see what is being resolved.
- **The install check cannot reach Hugging Face.** Installing needs the network
  even for a recommended model, because the check confirms the pinned revision
  still exists and its files still match before anything is transferred.

## The application does not open

- Keep the LM Atelier terminal open; it runs the local service.
- Check the terminal for the first error and confirm port `12340` is available.
- Open `http://127.0.0.1:12340` manually after the service reports that it is
  ready.

## A model catalog or download request fails

- Confirm internet access to `huggingface.co`, then use **Retry**.
- Installed models remain available when the online catalog is unreachable.
- Some repositories require accepting their terms and saving a Hugging Face
  token in LM Atelier.
- Confirm enough disk space for the selected files and temporary download data.

## A chat, image, or video request fails

- Copy the exact error before retrying.
- Workers for installed models start automatically, including supported engine
  setup when needed. You do not need to configure backend ports or processes.
  If one remains stopped or exits, use its displayed error before retrying.
- On Linux, automatic engine setup currently covers compatible chat models.
  Image/video require an externally configured compatible media engine and are
  not certified.
- Restart LM Atelier if a worker exited, then retry once.
- Model support depends on its format, runtime workflow, and available system
  memory or VRAM. Installer support alone does not certify a model.

## Find local data

- Windows: `%LOCALAPPDATA%\LMAtelier\data`
- Linux: `$XDG_DATA_HOME/lm-atelier`, or `~/.local/share/lm-atelier` when
  `XDG_DATA_HOME` is unset

The source desktop entry point uses the same platform location. A copied
example `.env` selects `./data`; `LOCAL_LM_DATA_DIR` can select another folder.

Logs are in the `logs` folder inside that location. Installing an update keeps
local data. Uninstalling also preserves it unless you explicitly choose or
request a purge. Purge targets `%LOCALAPPDATA%\LMAtelier\data` on Windows or
`~/.local/share/lm-atelier` on Linux. Custom `XDG_DATA_HOME` and
`LOCAL_LM_DATA_DIR` locations, and operating-system-vault credentials, remain.
Back up the directory before moving between releases if the data is important.

If the problem continues, follow [SUPPORT.md](../SUPPORT.md). Never post a
credential, private chat, private media, or unreviewed log.

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

## When a worker stops, and what the message means

Settings shows one sentence for a stopped worker, with advice underneath and the
engine's own output behind "What the engine reported". Search this page for the
sentence you were shown.

### "needs more graphics memory than this computer has free"

The model did not fit on the graphics card. Nothing is broken and nothing needs
reinstalling - the model is simply larger than the card can hold.

The most reliable fix is a smaller or more compressed model: for chat models a
lower quantisation of the same model, and for image and video models a version
published at a smaller size. Reducing image dimensions and step count lowers the
requirement for image and video work specifically. Close other programs using the
card - a browser with hardware acceleration and a game launcher can hold a
surprising amount. If the model exposes a layer-offload setting, moving fewer
layers onto the card trades speed for fitting.

### "needs more system memory than this computer has free"

The same problem in main memory rather than on the card. Close other programs and
try again, or choose a smaller model. On Windows the paging file is used while a
model loads, so increasing its size can be enough on its own.

### "could not start because its port is already in use"

Almost always a copy of the engine left running from an earlier session, holding
the port the new one needs. Restarting LM Atelier clears it. If it persists,
something else on the machine is using that port and ending that program is the
fix.

### "could not read the selected model"

The file is present but the engine cannot use it: an unsupported format or
architecture, a file built for a newer engine than the one installed, or an
incomplete download. Reinstall the model first, since a truncated download looks
exactly like this. If it fails again, the model is not supported by this engine
version - choose one listed as supported.

### "engine program could not be started"

The runtime is missing or cannot run. Reinstall it from setup. If you set an
engine path yourself, check that it still points at the program.

### "took too long to start"

Not necessarily a failure. A large model read from a slow disk can take several
minutes the first time; a second attempt is usually much faster because the
operating system has cached the file. If it times out repeatedly on the same
model, raise the startup time limit in the Workers section of the Settings page
- the default is 60 seconds, and up to 10 minutes is allowed. If it still times
out at a generous limit, that model is too large for this machine to load
comfortably.

### "stopped unexpectedly"

LM Atelier could not tell what went wrong, so it is showing the engine's own
output unchanged. That output is the best evidence available; the worker log has
more of it.

## Messages in a worker log that are not faults

Worker logs record everything the engine writes, including lines that look
alarming and are not. If you are reading a log to find out why something failed,
you can skip these.

**`OSError: [WinError 10022] An invalid argument was supplied`, under
`Exception in callback _ProactorBasePipeTransport._call_connection_lost`.**
Windows only. This is Python's own networking layer tidying up a connection that
has already closed - it appears when LM Atelier finishes talking to the image
engine, including right after a successful generation test. The connection had
already been handled in full before the message was written, so nothing was lost
and no request failed. LM Atelier keeps these lines in the log file but leaves
them out of the error it shows you when a worker stops, so they cannot crowd out
the real cause.

If a request actually failed, the reason appears in the application, not only in
the log. Look for the error shown next to the message you sent.

## What the setup checklist is telling you

Setup shows one line per problem, and each line is the exact sentence below.
Search this page for the sentence you were shown.

The code in the first column never appears in the app. It is what a diagnostic
bundle and the `/api/setup/readiness` response use, so quote it if you are
asking for help.

### Waiting - nothing is wrong

| Code | What you were shown |
|---|---|
| `runtime_installing` | The required runtime is being installed. |
| `install_in_progress` | The model is being installed. |
| `worker_starting` | The managed worker is starting. |
| `generation_verification_running` | The local generation test is running. |

The first ComfyUI install expands to several gigabytes across tens of thousands
of files, so "being installed" can legitimately last several minutes with no
visible change. Give it time before assuming it has hung.

### The runtime

| Code | What you were shown |
|---|---|
| `runtime_missing` | The required runtime is not installed. |
| `runtime_failed` | The required runtime did not start or install. |
| `runtime_unsupported` | Automatic setup for the required runtime is unavailable on this machine. |

`runtime_missing` and `runtime_failed` both offer a button that installs or
retries; a failed install is safe to retry, and a partial download is discarded
rather than reused.

`runtime_unsupported` deliberately offers **no** action. It means this machine
cannot run that runtime - not that something went wrong - and the message
carries the specific reason. Nothing in the app will change that answer, so use
a different engine or a different machine.

### The model

| Code | What you were shown |
|---|---|
| `model_missing` | No active model is installed for this role. |
| `install_failed` | The last model installation failed. |
| `model_unsupported` | The active model is not compatible with this setup. |
| `profile_missing` | No usable profile is bound to this model. |

`model_unsupported` is about this machine and this engine, not about the model
being bad. A model can be unsupported here and work elsewhere.

### The workflow (image and video only)

| Code | What you were shown |
|---|---|
| `workflow_missing` | No compatible workflow is installed for this model. |
| `workflow_invalid` | The compatible workflow is incomplete. |
| `workflow_untrusted` | The compatible workflow has not been trusted. |

`workflow_untrusted` is usually an imported workflow. Starting verification will
first try to rebuild it from the template it records: if the rebuild is
byte-identical, it is trusted automatically and you will not see this again. If
it cannot be rebuilt - hand-authored, edited after compiling, or needing nodes
outside the ComfyUI core - it stays untrusted and needs review.

### Activation and the generation test

| Code | What you were shown |
|---|---|
| `activation_required` | The model has not passed an activation probe. |
| `activation_failed` | The model did not pass its activation probe. |
| `activation_stale` | The model must be rechecked for the current runtime and hardware. |
| `generation_verification_required` | Run one quick local generation test. |
| `generation_verification_failed` | The local generation test did not complete. |

`activation_stale` does not mean anything broke. Evidence records the runtime
and the machine it was proven on, so a runtime update can require a recheck.
Activation is in-place and does not re-download the model.

The generation test is the only step that proves the whole path works end to
end, which is why setup is not finished without it.

### The worker

| Code | What you were shown |
|---|---|
| `worker_failed` | The managed worker exited unexpectedly. |
| `worker_status_unavailable` | The managed worker status is unavailable. |

Both offer a restart. If a worker fails repeatedly on the same model, the model
is the more likely cause than the worker.

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

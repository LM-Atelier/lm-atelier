# Privacy and local data

LM Atelier stores chats, prompts, settings, model metadata, and generated media
locally. It does not include a telemetry or analytics service.

LM Atelier uses the network when you browse the Hugging Face catalog, download a
model, or use an installed model whose supported engine must be downloaded. A
configured adapter, model, or custom node may also make network requests.
Credentials entered in the app use the operating-system vault rather than
application data. Advanced source deployments may instead provide
`LOCAL_LM_HF_TOKEN`; that value remains the operator's environment-secret
responsibility.

The default data locations are:

- Windows installer: `%LOCALAPPDATA%\LMAtelier\data`
- Linux installer: `$XDG_DATA_HOME/lm-atelier`, or
  `~/.local/share/lm-atelier` when `XDG_DATA_HOME` is unset

The source desktop entry point uses the same platform location. Set
`LOCAL_LM_DATA_DIR` to use another location; the example `.env` uses `./data`.
The public preview accepts loopback connections only and does not provide LAN
access.

Deleting a chat removes its active history. Generated media can be deleted with
the chat when it is not shared elsewhere; otherwise it becomes unreferenced and
uses the 30-day recovery window before cleanup. Rotating recovery backups may
retain earlier database state until their 7-daily/4-weekly retention expires.
An explicit uninstall purge removes only the managed default data directory:
`%LOCALAPPDATA%\LMAtelier\data` on Windows or `~/.local/share/lm-atelier` on
Linux. A custom `XDG_DATA_HOME` or `LOCAL_LM_DATA_DIR`, and credentials in the
operating-system vault, must be removed separately.

Before sharing issue details, inspect them and remove tokens, private prompts,
chats, media, model inputs, and identifying file paths.

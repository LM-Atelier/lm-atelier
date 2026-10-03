# Privacy and local data

LM Atelier stores chats, prompts, settings, model metadata, and generated media
locally. It does not include a telemetry or analytics service.

LM Atelier uses the network when you browse the Hugging Face or CivitAI catalog, check for model updates, download a
model, or use an installed model whose supported engine must be downloaded. A
configured adapter, model, or custom node may also make network requests.
Credentials entered in the app use the operating-system vault rather than
application data. Advanced source deployments may instead provide
`LOCAL_LM_HF_TOKEN`; that value remains the operator's environment-secret
responsibility.

Optional web search sends the displayed query to your configured CRW service.
It needs installation-wide web access and separate permission for the chat.
By default each exact query needs approval; a chat can instead allow automatic
searches after a visible five-second cancellation window. A changed destination
or account requires fresh approval. Stopping a sent request cannot retract its
query from the service.

Search results are quoted evidence for the answer, and result pages are not
opened automatically. Reading a link requires its separate chat permission and
an address included in a user message. Search queries, provider addresses,
results and outcomes are retained in local response history and may be included
in database backups. Credentials are not stored in that history. Search tokens
use the operating-system vault, or the optional `LOCAL_LM_CRW_TOKEN` environment
override.

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

Automatic recovery backups and **Back up state** in Settings save database
records, including chats, settings, and library entries. They do not contain
image or video files, so restoring a state-only backup cannot recover missing
media. **Back up with media** also saves the referenced media files; creation
fails if a required file is missing or changed. Use **Verify** to check the
backup before relying on it, and keep its database snapshot and media archive
together when copying it elsewhere.

The **Generation record** control beside a generated picture or video saves a
small JSON file describing how it was made: the output's file hash, the
operation, the seed and settings, the workflow, model and LoRA files by hash,
and the inputs by hash. The prompt is left out unless you choose to include it,
and text a workflow takes as a setting follows the same choice. The record never
names your chats, messages or files on this computer, and it lists what it left
out. It does not contain the picture or video itself. For a picture, **Download
with picture** saves a ZIP file holding that same record beside a PNG copy of
the picture with only its pixels. Nothing else written inside the picture file,
such as the workflow that made it, is copied; a picture with its own color
profile is converted to sRGB first, so the profile stays behind too.
**Check a record** in the Model library reads such a record, or such a ZIP
file, and says which of the files it names are on this computer and ready. It
keeps no copy of the file and installs, downloads or changes nothing.

Before sharing issue details, inspect them and remove tokens, private prompts,
chats, media, model inputs, and identifying file paths.

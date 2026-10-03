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
profile is converted to sRGB first, so the profile stays behind too. The
pictures it was made from, such as the picture an edit changed or its
selection, go into that ZIP file only when you choose to save them with it, and
then all of them are copied the same way.
**Keep as a recipe**, offered in the record of a picture or video made from
words or a video made from a picture, drafts a recipe from the settings it ran
with for you to review before saving. The prompt, the seed, how many results to
make, LoRAs chosen to match the prompt and any picture it was made from stay out
of the recipe, and the generation and its record are not changed. For an edit,
**Keep as a recipe** saves a recipe of the kind Image Studio offers, with a name
and the words you choose, starting from the ones you typed for the edit; it keeps
whether the edit used a selection, but not the selection itself, the seed or LoRAs
chosen to match the words. A recipe saved in Image Studio leaves the same out.
**Check a record** in the Model library reads such a record, or such a ZIP
file, and says which of the files it names are on this computer and ready. It
keeps no copy of the file and installs, downloads or changes nothing. When
everything it names is here exactly, **Generate again** starts it in a new chat
from the record's prompt, seed and settings. When only its workflow, model,
LoRAs or input pictures are not here, **Make a new version** starts it with the
ones you choose in their place, a picture from your library for a picture; that
generation keeps which record it came from, what you chose and what differs from
the record, and the record file is not changed. When the ZIP file carries a copy
of a missing picture, that copy can be chosen instead; it is then kept on this
computer with the new version, as a picture attached to a message is, and not
added to your Media Library.

**Settings in the file**, under a picture in the Media Library, reads the
generation settings some image tools write into a picture's file: the prompt
and negative prompt, seed, steps, guidance, sampler, scheduler, denoise and size.
From a PNG, only text the file stores uncompressed is read; from a JPEG or WebP,
only its EXIF text, never the picture itself or any XMP text. Nothing read is
kept, and nothing the text names is fetched, installed or run: models, LoRAs,
other files and workflows it names are listed as left out. Other kinds of
picture are not read. When the file carries a ComfyUI workflow, **Review this
picture's workflow** reads it and opens the same review as importing a workflow
file: nothing is added to Workflows until you import it there, and what is
added cannot run until you review it.
**Remix these settings** checks those settings against a workflow and a model
you choose here, never by a name or file the text gives, and shows what a new
picture would be made with before anything is made. When the workflow cannot make
the picture's own size, it can offer the same shape at a size it makes instead.
Making it starts one picture in a new chat with the settings you chose to use; no
saved preset or settings
recipe enters it. The new picture keeps which picture, workflow and model it
came from and which settings were used, but no prompt or value from the file
beyond those its own generation records. The picture and its file are not
changed. A remix can also start from the picture itself: then the picture, only
one in your Media Library, is the starting image that a workflow you choose
changes on this computer, and it is kept for as long as the remix is.

Before sharing issue details, inspect them and remove tokens, private prompts,
chats, media, model inputs, and identifying file paths.

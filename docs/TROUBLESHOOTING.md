# Troubleshooting

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

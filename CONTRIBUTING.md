# Contributing

Thanks for helping improve LM Atelier. For questions and bug reports, start with
[SUPPORT.md](SUPPORT.md). Security reports must follow [SECURITY.md](SECURITY.md).

## Development

Follow the source setup in [README.md](README.md). The React client is in
`apps/web`; the FastAPI service and tests are in `services/api`.

Work follows this branch flow:

```text
descriptive work branch -> develop -> main
```

- Branch from the latest `develop`.
- Use a descriptive branch name without `feature/`, `feature-`, or `feat/`.
- Keep each pull request focused and target `develop`.
- Use `hotfix-<description>` from `main` only for urgent production fixes, then
  apply the same fix to `develop`.
- Releases are promoted from `develop` to `main` and tagged from `main`.

## Pull requests

Explain the user impact, important design choices, and validation performed.
Update tests and concise documentation when behavior changes. Accepted work is
squash-merged, and its branch is deleted.

Contributions are submitted under the project's Apache License 2.0
(inbound equals outbound). Add a `Signed-off-by` trailer to each commit with
`git commit -s` to certify the
[Developer Certificate of Origin 1.1](https://developercertificate.org/).

Model or runtime changes must document the capability contract and add
discovery, preflight, and generation fixtures. Backend-supported models must
work through capability introspection; curated per-model recipes may improve
results but cannot be required for basic support.

During development, run the checks closest to the files being changed:

~~~powershell
Set-Location services/api
uv run ruff format .
uv run ruff check .
uv run mypy local_lm
uv run pytest tests/test_relevant_module.py --basetemp=../../temp/focused -p no:cacheprovider
~~~

~~~bash
npm --workspace @lm-atelier/web test
npm run typecheck
~~~

Run the complete local gate once before requesting review:

```bash
./scripts/verify.sh
```

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\verify.ps1
```

The full gate needs PowerShell 7 on Linux and Git for Windows with Git Bash on
Windows so both platforms' packaging scripts receive syntax checks.

Install the pinned Chromium build once with `npm run e2e:install`, then run the
isolated browser golden path with `npm run e2e`. It uses mock engines and a
fresh operating-system temporary data directory.

Never commit credentials, model weights, generated media, databases, caches,
logs, diagnostics, or anything under `.private`.

By participating, you agree to follow the
[Code of Conduct](CODE_OF_CONDUCT.md).

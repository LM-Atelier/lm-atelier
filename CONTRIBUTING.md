# Contributing

## Branches and pull requests

```text
descriptive work branch -> develop -> main
```

- Branch from the latest `develop` and use a descriptive name without
  `feature/`, `feature-`, or `feat/`.
- Keep each pull request focused and target `develop`.
- Explain the user impact and local validation in the pull request.
- Squash-merge accepted work and delete its branch.
- Promote tested releases from `develop` to `main`; tag releases from `main`.
- Branch urgent production fixes from `main` as `hotfix-<description>`, then
  apply the same fix to `develop`.

Do not commit credentials, model weights, generated media, databases, caches,
logs, diagnostics, or files under `.private`.

## Verification

Run the complete local gate before requesting a merge:

```bash
./scripts/verify.sh
```

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\verify.ps1
```

Hosted compatibility CI is opt-in to conserve the repository's Actions quota.
Maintainers can add the `run-ci` label to a ready pull request or dispatch the
workflow manually when an Ubuntu run is needed.

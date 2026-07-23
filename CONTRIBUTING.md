# Contributing

## Branch flow

```text
descriptive branch -> develop -> main
```

`main` is the release branch. `develop` is the integration branch for the next
release. Normal work must not be committed directly to either branch.

1. Update local `develop` and branch from it.
2. Name the branch for the change: `chat-persistence`, `model-download-queue`,
   or `142-image-routing` are good examples.
3. Do not prefix a feature branch with `feature/`, `feature-`, or `feat/`.
4. Open a pull request from the work branch into `develop`.
5. Squash-merge after required checks and review pass, then delete the branch.
6. Promote a tested release through a pull request from `develop` into `main`.
7. Create release tags from `main` only.

Emergency fixes branch from `main` as `hotfix-<description>`, merge into `main`,
and are then merged or cherry-picked into `develop` immediately.

## Pull requests

- Keep one coherent change per pull request.
- Explain the user impact and the validation performed.
- Update the API, tests, README, or security guidance when behavior or scope
  changes.
- Do not commit model weights, generated media, secrets, local databases, or
  runtime caches.

The project requires formatting, linting, type checks, unit tests, migration
checks, license/dependency scans, and secret scanning before merges. Run
`./scripts/verify.sh` on Linux or
`powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\verify.ps1` on
Windows for the local core checks.

## Hosted CI

Pull requests create a visible skipped check by default so routine verification
does not consume constrained hosted-runner quota. Run the complete local parity
gate above before requesting merge.

Add the `run-ci` label to a ready pull request when an Ubuntu compatibility run
is warranted. The label event starts CI, and later pushes rerun it while the
label remains attached. Remove and re-add the label to request another run
without a new commit. Maintainers can also start the workflow manually with
`workflow_dispatch`.

Before enabling required checks for a public release, replace this temporary
quota policy with protected-branch CI appropriate to the repository's runner
budget.

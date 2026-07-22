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

The project will require formatting, linting, type checks, unit tests,
migration checks, license/dependency scans, and secret scanning before merges
before merges. Run `./scripts/verify.sh` locally for the same core checks.

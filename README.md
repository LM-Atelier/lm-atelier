# Local LM

Local LM is a local-first AI workspace for private chat, contextual image and
video generation, and deep model tuning from one web interface.

The project is currently in the planning and architecture phase. Development
flows from short-lived, descriptively named branches into `develop`, then from
`develop` into `main` for releases.

## Branch policy

- `main` contains releasable code.
- `develop` integrates work for the next release.
- Feature branches start from `develop` and merge back into `develop`.
- Feature branch names describe the work, such as `chat-streaming` or
  `image-routing`; they never use a `feature/` or `feature-` prefix.

See [CONTRIBUTING.md](CONTRIBUTING.md) for the complete workflow.

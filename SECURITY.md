# Security policy

## Supported versions

Security fixes are applied to the latest release and the `develop` branch. This
project is pre-1.0; upgrade to the newest release before reporting a result.

## Reporting a vulnerability

Please use the repository's **Security → Report a vulnerability** workflow so a
report and any proof of concept remain private. Do not open a public issue for a
suspected vulnerability. Include the LM Atelier version, operating system,
whether LAN access was enabled, reproduction steps, and the expected impact.

## Security boundaries

- The server binds to loopback by default. Non-loopback binding requires the
  explicit `LOCAL_LM_ALLOW_LAN=true` opt-in. LM Atelier is not an internet-facing
  multi-user service and must be placed behind authentication and TLS if exposed.
- Browser mutations require a same-origin session and CSRF token. A LAN opt-in
  does not add accounts, authorization, or tenant isolation.
- Hugging Face credentials are stored through the operating system credential
  vault. The environment-variable fallback is intentionally process-scoped.
  Secrets are excluded from diagnostics, exports, backups, and API responses.
- Model imports reject unsafe serialized formats. Reference recipes pin repository
  revisions and file hashes; review third-party licenses before downloading.
- ComfyUI custom nodes execute third-party Python code. LM Atelier requires a
  canonical GitHub source, an exact commit, a reviewed tree hash, and explicit
  trust before activation. Trust is revoked after every update or rollback.
- Generated media, prompts, and chat history remain local unless a configured
  model or custom node performs network access. Review third-party node code and
  firewall the process when strict offline operation is required.

## Operational guidance

Keep LM Atelier and its engines updated, preserve verified backups, and install
releases only when their SHA-256 checksum matches. Release archives are currently
checksum-verified but not code-signed; that limitation is disclosed in release
notes until signing infrastructure is available.

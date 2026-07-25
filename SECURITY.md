# Security policy

## Report a vulnerability

Use GitHub's **Security > Report a vulnerability** workflow. Do not open a public
issue. Include the LM Atelier version, operating system, reproduction steps,
whether LAN access was enabled, and the expected impact.

Security fixes target the latest release and `develop`.

## Important boundaries

- LM Atelier binds to loopback by default. LAN access is an explicit opt-in and
  does not add user accounts or tenant isolation.
- Hugging Face credentials use the operating system credential vault and are
  excluded from application data, diagnostics, exports, and backups.
- Serialized model formats that can execute code are rejected.
- ComfyUI custom nodes are third-party executable code. Install and trust only
  reviewed sources pinned to an exact revision.
- Prompts, chats, and generated media remain local unless a configured model,
  engine, or custom node makes network requests.

Install current releases and verify downloads against the published SHA-256
checksums. Release binaries are not yet code-signed.

# Security policy

## Report a vulnerability privately

Use GitHub's
[private vulnerability reporting](https://github.com/ajccarlson/lm-atelier/security/advisories/new).
Do not open a public issue.

Include the LM Atelier version, operating system, reproduction steps, and the
expected impact. Remove credentials, private prompts, chats, and media that are
not necessary to reproduce the issue.

Security fixes target the latest release and `develop`.

## Security boundaries

- The public preview accepts loopback connections only. LAN access is not
  supported until it has a separate authentication and origin model. The API
  requires a local session, CSRF token for changes, trusted host, and local
  browser origin; WebSocket events require the same session and origin.
- Hugging Face credentials entered in the app use the operating-system vault
  and are excluded from application data, diagnostics, exports, and backups.
  The advanced `LOCAL_LM_HF_TOKEN` environment override remains in the
  launching process environment and must not be saved in a shared `.env`.
- Serialized model formats that can execute code are rejected.
- Imported archives are size-bounded and reject traversal, links, duplicate
  members, and unsupported content before extraction.
- ComfyUI custom nodes and third-party adapters execute with LM Atelier's local
  permissions. Review their source and use pinned revisions.
- Prompts, chats, and generated media remain local unless a configured model,
  runtime, adapter, or custom node makes network requests.

Install current releases and verify downloads against the published SHA-256
checksums. Release binaries are not yet code-signed.

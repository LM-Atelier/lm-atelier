# Security

## Defaults

LM Atelier binds to loopback, issues an authenticated browser session, requires a
CSRF token for state changes, does not send prompts or media as telemetry, and
stores generated content under opaque artifact IDs. A non-loopback bind is
rejected unless LAN mode is explicitly enabled.

Model imports block pickle-compatible formats by default. Worker commands are
constructed as argument arrays and never passed through a shell. ComfyUI custom
nodes remain an explicit external trust decision because they are executable
Python code.

## Sensitive data

Do not commit `.env`, `.private`, the `data` directory, model weights, generated
media, databases, logs, or diagnostic bundles. Hugging Face credentials are
read at process startup and are not written to SQLite.

Project exports and backups can contain private conversations and media. Treat
them as sensitive user data even though their local directories are ignored by
Git.

## Reporting

Until a private security contact is published, do not open a public issue with
credentials, private prompts, model files, or exploit details. Contact the
repository owner privately through GitHub first.

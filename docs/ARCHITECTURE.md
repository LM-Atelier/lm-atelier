# Architecture

LM Atelier is a local web interface and FastAPI control plane. The browser talks
only to the loopback API; model runtimes run as replaceable workers so a failed
or unloaded model does not stop the workspace service.

```text
React UI -> loopback API -> routing and durable jobs -> chat/media workers
                         -> SQLite metadata + content-addressed files
                         -> Hugging Face catalog/downloads when requested
```

- SQLite stores projects, chats, runs, profiles, workflows, and job state.
- Models and generated media stay on disk and are referenced by verified IDs.
- Chat, image, and video profiles are independent. Auto Mode resolves a request
  against conversation context and available role capabilities.
- `llama.cpp` and ComfyUI use adapter contracts; optional adapters must satisfy
  the same typed capability, streaming, cancellation, and lifecycle behavior.
- The service binds to loopback by default. Credentials entered in the app use
  the operating-system vault; an advanced environment override is available.
  Executable model formats and untrusted custom nodes are blocked.

Start with [runtime adapters](ADAPTERS.md) when extending inference support.
Database changes live in `services/api/local_lm/migrations`; API and web tests
define the compatibility contract.

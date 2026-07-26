# Runtime adapters

An installed Python package can provide a chat or media backend through LM
Atelier's version 1 adapter contract. Adapters run in-process with LM Atelier's
local permissions, so users must explicitly select a trusted package.

## Register an adapter

Expose a factory through the appropriate entry-point group:

```toml
[project.entry-points."lm_atelier.chat_adapters"]
my-runtime = "my_package:build_chat_adapter"

[project.entry-points."lm_atelier.media_adapters"]
my-media-runtime = "my_package:build_media_adapter"
```

The factory receives a copy of `local_lm.config.Settings` with credential
fields cleared, returns a `ChatAdapter` or `MediaAdapter`, and declares its
contract version:

```python
def build_chat_adapter(settings):
    return MyChatAdapter(settings)

build_chat_adapter.lm_atelier_contract_version = 1
```

Select it with `LOCAL_LM_CHAT_ENGINE` or `LOCAL_LM_MEDIA_ENGINE`. Startup rejects
unknown or unsafe names, duplicate providers, missing methods, and incompatible
contract versions. Loading an entry point already executes its package code;
the cleared settings object is defense in depth, not a sandbox.

## Required behavior

Chat adapters report capabilities, count context tokens, stream typed events,
cancel by run ID, and close resources. Media adapters also validate a workflow
before generation and return at least one generated asset on success.

Successful streams end with `complete`. Cancellation is idempotent, terminal
events are final, and adapters never mutate persisted chat state. Capability
settings are authoritative: unsupported values must be omitted or marked
unavailable rather than silently ignored.

`input_modalities` always includes `text`. A chat adapter may also advertise
`image` only when its current loaded model can accept OpenAI-compatible
`image_url` message parts; LM Atelier otherwise supplies honest text-only
context and never assumes that the model inspected pixels. Image context is
read only from verified, content-addressed local raster artifacts, bounded,
and embedded as `data:` URLs; adapters never receive artifact paths or remote
image URLs. Video context uses a verified local poster frame when one exists.

Multi-role adapters populate `settings_by_role` and retain the same definitions
in `settings` for older clients. Once a role mapping is present, it must cover
every advertised role. Clients fall back to `settings` only when the mapping is
absent.

LM Atelier bounds individual events, previews, assets, stream volume, and idle
time. Invalid or unterminated streams fail with a generic adapter error so
third-party exception text is not copied into chats or logs.

Role settings are overlaid on LM Atelier's built-ins. Adapters may add bounded
fields, narrow a built-in range or enum, or mark a field unavailable. They may
not remove built-ins, change a built-in type or scope, broaden its constraints,
duplicate a key across scopes, or declare runtime bindings such as `prompt`,
`input_image`, and underscore-prefixed internal keys.

## Conformance

```python
from local_lm.adapters.conformance import probe_chat_adapter, probe_media_adapter

(await probe_chat_adapter(configured_chat_adapter)).assert_passed()
(await probe_media_adapter(
    configured_media_adapter,
    workflow=minimal_workflow,
)).assert_passed()
```

The media probe performs a real 64x64 one-step generation. Additive fields remain
compatible within a contract version; removing methods, requiring fields, or
changing event meaning requires a new contract version.

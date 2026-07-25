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

The factory receives `local_lm.config.Settings`, returns a `ChatAdapter` or
`MediaAdapter`, and declares its contract version:

```python
def build_chat_adapter(settings):
    return MyChatAdapter(settings)

build_chat_adapter.lm_atelier_contract_version = 1
```

Select it with `LOCAL_LM_CHAT_ENGINE` or `LOCAL_LM_MEDIA_ENGINE`. Startup rejects
unknown names, duplicate providers, missing methods, and incompatible contract
versions.

## Required behavior

Chat adapters report capabilities, count context tokens, stream typed events,
cancel by run ID, and close resources. Media adapters also validate a workflow
before generation and return at least one generated asset on success.

Successful streams end with `complete`. Cancellation is idempotent, terminal
events are final, and adapters never mutate persisted chat state. Capability
settings are authoritative: unsupported values must be omitted or marked
unavailable rather than silently ignored.

Multi-role adapters populate `settings_by_role` and retain the same definitions
in `settings` for older clients. Clients fall back to `settings` when a role
mapping is absent.

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

# Runtime adapter contract

LM Atelier's version 1 runtime contract lets an installed Python package supply
a chat or media backend without changing project, chat, message, job, or artifact
code. Adapters are trusted in-process code and are loaded only when their exact
entry-point name is selected in `LOCAL_LM_CHAT_ENGINE` or
`LOCAL_LM_MEDIA_ENGINE`.

## Publishing an adapter

Expose a factory from the appropriate Python entry-point group:

```toml
[project.entry-points."lm_atelier.chat_adapters"]
my-runtime = "my_package:build_chat_adapter"

[project.entry-points."lm_atelier.media_adapters"]
my-media-runtime = "my_package:build_media_adapter"
```

The factory receives `local_lm.config.Settings`, returns an object satisfying
`ChatAdapter` or `MediaAdapter`, and must declare the exact contract version:

```python
def build_chat_adapter(settings):
    return MyChatAdapter(settings)

build_chat_adapter.lm_atelier_contract_version = 1
```

LM Atelier rejects unknown names, duplicate providers, missing methods, and
contract-version mismatches during startup. Selecting an adapter authorizes its
package to execute with the same local permissions as LM Atelier; install and
select only packages you trust.

## Required behavior

Chat adapters report capabilities, count context tokens, stream typed events,
cancel by run ID, and close resources. A successful stream ends with `complete`.
Media adapters additionally validate an executable workflow before generation;
a successful stream ends with `complete` and at least one generated asset.
Cancellation must be idempotent, terminal events must not be followed by more
output, and adapters must not mutate persisted chat state directly.

Capabilities are the source of truth. Unsupported settings must be absent or
marked unavailable with a reason; silently dropping submitted settings violates
the contract. Generated assets return bytes and metadata to the control plane,
which remains responsible for content-addressed storage and provenance.

## Conformance

An adapter package can use the same probes as LM Atelier's built-ins:

```python
from local_lm.adapters.conformance import probe_chat_adapter, probe_media_adapter

async def test_chat_contract(configured_adapter):
    (await probe_chat_adapter(configured_adapter)).assert_passed()

async def test_media_contract(configured_adapter, minimal_workflow):
    report = await probe_media_adapter(configured_adapter, workflow=minimal_workflow)
    report.assert_passed()
```

Media conformance performs a real 64×64, one-step generation, so the fixture must
provide a safe test runtime and any workflow-specific inputs. CI for LM Atelier
runs these probes against its deterministic mock adapters without model weights.

## Compatibility policy

Contract versions are integers independent of the application version. Additive
dataclass fields use defaults and remain compatible within a contract version.
Removing a method, changing event meaning, or making a field required creates a
new contract version. LM Atelier supports the current version and announces
removal of an older version at least one minor release before removal. Pre-1.0
application releases may change internal APIs, but an advertised adapter contract
is changed only through this explicit version mechanism.

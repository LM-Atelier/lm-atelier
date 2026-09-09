# LM Atelier <!-- version --> — preview

Source: `<!-- full tagged commit SHA -->`

LM Atelier is preview software. These installers were build-smoked for:

- Windows 11 x64: <!-- build-smoked / physically tested / certified -->
- Ubuntu 24.04 LTS x86_64: <!-- build-smoked / physically tested / certified -->

Platform and model support remains limited to the combinations tested for this
preview.

Managed llama.cpp chat setup is one-click on both installer targets where the
pinned runtime is compatible. Managed media setup is currently limited to the
reviewed compatible Windows NVIDIA runtime. Linux image/video require an
externally configured compatible media engine and are not certified.

## What changed

- Browse workflow families and variants with search, readiness filters, dependency
  details, editable use cases, and defaults for each operation.
- Reuse Prompt Templates, generate grouped outputs, and manage image references
  from the media library.
- Recognize installed CivitAI checkpoint versions and derive editable profile
  use cases from model metadata.
- Improve queue recovery, artifact retention, shared model storage, and
  workflow validation.

## Install and upgrade

Installers include Python and the built web interface; no separate Python or
Node.js installation is required. Models and inference engines are downloaded
separately. Back up your local data before upgrading from 0.1.8;
the release checks exercise a fresh installation and the 0.1.8 upgrade path.

Upgrades preserve local data. Uninstall preserves it unless purge is explicitly
selected. Application downgrades and data-schema rollback are not guaranteed.

## Known limitations

These preview binaries are not code-signed. Model, workflow, and hardware support
depends on the configured runtime and available resources. Automated installer
checks do not certify every GPU or model combination.

A working local chat model is required for model-filled Prompt Template slots.
Model output that cannot satisfy a template reports an error. Check the template
and chat setup before retrying.

## Verify the download

Download the installer and its release file set. Verify every file with the
single top-level `SHA256SUMS` before running it. Platform-specific evidence uses
`windows` or `linux` in its filename:

- `release-manifest-<platform>.json` — exact source and build metadata
- `sbom-<platform>.cdx.json` — CycloneDX software bill of materials
- `payload-manifest-<platform>.json` — exact frozen-file hashes
- `gitleaks-<platform>-{payload,metadata,installer}.json` — redacted
  secret-scan results for each built scope
- `npm-audit-<platform>.json` and `pip-audit-<platform>.json` — sanitized
  locked-dependency audit results
- `malware-scan-<platform>.txt` — scanner/signature version and clean result
- `THIRD_PARTY_NOTICES-<platform>.md` and the matching license archive
- the Apache-2.0 `LICENSE`

Public release candidates also include a Sigstore bundle named
`provenance-attestation.sigstore.json`. Verify it against this repository with
GitHub CLI before trusting the build.

Models, workflows, inference engines, and custom nodes retain their own licenses
and may have separate usage terms.

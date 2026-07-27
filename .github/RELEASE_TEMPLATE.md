# LM Atelier <!-- version --> — preview

Source: `<!-- full tagged commit SHA -->`

LM Atelier is preview software. These installers were build-smoked for:

- Windows 11 x64: <!-- build-smoked / physically tested / certified -->
- Ubuntu 24.04 LTS x86_64: <!-- build-smoked / physically tested / certified -->

Do not describe a platform, GPU, model, or workflow as certified unless its
recorded test matrix is complete.

Managed llama.cpp chat setup is one-click on both installer targets where the
pinned runtime is compatible. Managed media setup is currently limited to the
reviewed compatible Windows NVIDIA runtime. Linux image/video require an
externally configured compatible media engine and are not certified.

## Install and upgrade

<!-- State prerequisites, supported upgrade path, and data-preservation behavior. -->

## Known limitations

<!-- List material limitations, including unsigned status when applicable. -->

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

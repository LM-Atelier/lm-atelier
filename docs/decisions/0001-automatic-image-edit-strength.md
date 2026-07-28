# Automatic image-edit change strength

Status: Accepted for E0-E3.

## Decision

Image-to-image requests use a local, deterministic, versioned prompt estimator when no numeric strength exists in any user-controlled settings layer. The policy maps requests to five bounded scopes:

| Scope | Canonical value | Intended range |
| --- | ---: | ---: |
| Minimal | 0.38 | 0.30-0.44 |
| Localized | 0.50 | 0.42-0.58 |
| Replacement | 0.66 | 0.58-0.74 |
| Global | 0.82 | 0.72-0.90 |
| Fallback | 0.56 | 0.52-0.60 |

An immutable workflow revision may declare a versioned `x-lm-atelier-edit-calibration` contract. It maps semantic strength to a numeric workflow parameter, supplies per-scope recommendations and bounds, and may identify a resolved step parameter with minimum effective edit budgets. Live setting bounds take precedence. A short schedule may raise an Auto value only within those bounds; it never changes an explicit Manual value or assumes a scheduler or model family.

Missing or malformed optional calibration falls back to the general `denoise` policy. Standard derived image-edit workflows publish the contract as new revisions, preserving historical workflow identity. There is no repository or model-name lookup table.

Explicit profile, preset, project, chat, and per-turn values remain exact and authoritative. Regeneration and inherited branches reuse an earlier Auto value unless the user supplies a new value for the workflow's declared parameter. Text-to-image behavior does not change.

Single and ordered media paths use the same resolver. Provenance records only bounded mode, scope, confidence, reason codes, estimator version, bounds, value, setting source, optional calibration version/hash, and any bounded schedule adjustment; it never copies prompt text. Classification is linear string processing and performs no model or network call.

## Limits

The estimator recognizes a conservative English signal set. Ambiguous or unsupported-language prompts receive the fallback value and low confidence. Synthetic real-runtime calibration and optional vision-assisted estimation are later phases and must not weaken Manual authority.
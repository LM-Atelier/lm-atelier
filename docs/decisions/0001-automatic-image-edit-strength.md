# Automatic image-edit change strength

Status: Accepted for E0/E1.

## Decision

Image-to-image requests use a local, deterministic, versioned prompt estimator when no numeric `denoise` value exists in any user-controlled settings layer. The initial policy maps requests to five bounded scopes:

| Scope | Canonical value | Intended range |
| --- | ---: | ---: |
| Minimal | 0.38 | 0.30–0.44 |
| Localized | 0.50 | 0.42–0.58 |
| Replacement | 0.66 | 0.58–0.74 |
| Global | 0.82 | 0.72–0.90 |
| Fallback | 0.56 | 0.52–0.60 |

The selected value is clamped to the active workflow's advertised bounds. Explicit profile, preset, project, chat, and per-turn values remain exact and authoritative. Regeneration and inherited branches reuse an earlier Auto value unless the user supplies a new numeric value. Text-to-image behavior does not change.

Single and ordered media paths use the same resolver. Provenance records only bounded mode, scope, confidence, reason codes, estimator version, bounds, value, and setting source; it never copies prompt text. Classification is linear string processing and performs no model or network call.

## Limits

The first estimator recognizes a conservative English signal set. Ambiguous or unsupported-language prompts receive the fallback value and low confidence. Workflow-specific parameter mapping and calibration, richer UI controls, and optional vision-assisted estimation are later phases and must not weaken manual override authority.
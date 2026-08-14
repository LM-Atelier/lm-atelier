# Phase A: ArtifactLibraryEntryV1

Status: freeze candidate for independent review
Branch: `grok-media-library-entry-v1`
Base: `226e9154162153cf435f74f55004cca9b3f97549`

## Scope

Durable Media Library membership separate from content-addressed artifact bytes.

In scope:

- `artifact_library_entries` table, model, SQLite triggers
- Idempotent backfill of existing image/video artifacts
- Explicit publication for generation outputs, image/video uploads, and project import media
- Favorite dual-write (entry is canonical; artifact.favorite mirrored)
- Fail-closed retention reference graph including library membership
- User DELETE remains blocked while membership is visible (no Trash yet)

Out of scope (later phases):

- Cursor pagination / facets / list cutover
- Soft-trash recovery UI and purge policy
- Albums, tags, smart collections, near-duplicates, contact sheets

## Identity

| Field | Rule |
| --- | --- |
| `id` | `libentry:sha256:{artifact.sha256}` only |
| `artifact_id` | FK RESTRICT; immutable after insert |
| kinds | image and video only |
| generic ingest | does **not** publish membership |

## Publication writers

| Path | Publishes? |
| --- | --- |
| `ensure_library_entry` | yes, image/video only, idempotent |
| generation artifact create (orchestrator) | yes |
| `POST /api/artifacts?kind=image\|video` | yes |
| project import media | yes |
| generic `ingest_bytes` / non-media kinds | no |
| setup-verification synthetic input | no |

## Retention

`referenced_artifact_ids` is the single authority for strong refs. It includes:

- message / revision parts
- reference subjects and assets
- setup verification inputs
- **visible library entry artifact ids**
- message_references JSON
- run provenance/settings (including masks), work step bindings/settings, studio chat origin
- job payload/result nested artifact ids (bounded, fail-closed)

Corrupt JSON aborts retention with fixed text: `Stored artifact reference data is invalid.`

Visible or recoverable/trashed membership pins bytes even without favorites or message parts.
Destructive paths take a SQLite writer reservation before reference proof and row deletion.
Every ORM JSON-reference writer takes the same reservation and rechecks each referenced
Artifact before flush, so a delete and a reference publication cannot both commit.
Shared create-all/Alembic triggers provide the bulk/raw-SQL backstop: guarded JSON
inserts/updates must reference existing Artifact rows, and JSON-retained Artifacts
cannot be deleted even when the ORM is bypassed.

## Delete authority

| Caller | Behavior |
| --- | --- |
| User `DELETE /api/artifacts/{id}` | 409 `artifact-in-use` if membership exists (Phase A) |
| internal artifact deletion | refuses whenever membership exists; entry DELETE is DB-sealed |
| chat-generated media cleanup | preserves published entries and bytes |
| setup verification cleanup | setup output is never published; clears exact DB refs before deleting unpublished bytes |

Trash/soft-delete and recovery_id transitions are schema-ready (`visible` / `trashed`) but not user-exposed yet.

## Acceptance matrix (read-only review)

1. Migration from parent `c7e1d4a83b56` creates table, indexes, triggers; backfills only image/video once.
2. Non-media and generic ingest never create entries.
3. Re-running publication for the same artifact is idempotent.
4. Favorite PATCH dual-writes entry + artifact; version increments by exactly 1 (trigger enforced).
5. Identity and version triggers refuse illegal updates.
6. Visible membership prevents retention GC and low-level `_delete_artifact`.
7. User DELETE of a published item returns 409 with fixed membership text.
8. Entry deletion never deletes bytes; no generic membership-release authority exists.
9. Corrupt job reference JSON fails closed with fixed error text.
10. Run and ordered-step mask references are retained under the same write fence.
11. No list/pagination API cutover in this phase.

## Holds

No push, PR, merge, engines pin, runtime package mutation, or private-media inspection.

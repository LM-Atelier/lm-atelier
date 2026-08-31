# Artifact library entry contract

## Scope

Durable Media Library membership is separate from content-addressed artifact bytes.
The contract covers the `artifact_library_entries` table and model, SQLite
integrity triggers, idempotent backfill for existing image and video artifacts,
explicit publication by supported media writers, favorite dual-write, and
fail-closed retention. Generic ingest does not publish membership.

The current API does not expose list pagination, facets, soft-trash recovery,
purge policy, albums, tags, smart collections, near-duplicates, or contact
sheets.

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
| generation artifact create | yes |
| `POST /api/artifacts?kind=image\|video` | yes |
| project import media | yes |
| generic `ingest_bytes` / non-media kinds | no |
| setup-verification synthetic input | no |

## Retention

`referenced_artifact_ids` is the single authority for strong references. It
includes message and revision parts, reference subjects and assets, setup
verification inputs, visible library membership, message references, run and
work-step settings, studio chat origin, and bounded nested artifact ids in job
payloads and results.

Corrupt JSON aborts retention with fixed text: `Stored artifact reference data
is invalid.` Visible or recoverable membership pins bytes even without favorites
or message parts. Destructive paths reserve the SQLite writer before reference
proof and deletion. JSON-reference writers use the same reservation and recheck
referenced artifacts before flush. Database triggers provide the bulk and raw-SQL
backstop.

## Delete behavior

| Caller | Behavior |
| --- | --- |
| User `DELETE /api/artifacts/{id}` | 409 `artifact-in-use` while membership exists |
| internal artifact deletion | refuses while membership exists; entry deletion is sealed |
| chat-generated media cleanup | preserves published entries and bytes |
| setup verification cleanup | clears exact references before deleting unpublished bytes |

Trash state is schema-ready but is not exposed to users.

## Invariants

1. Migration creates the table, indexes, and triggers and backfills image/video once.
2. Non-media and generic ingest never create entries.
3. Repeated publication for one artifact is idempotent.
4. Favorite updates dual-write entry and artifact and increment version once.
5. Identity and version triggers refuse illegal updates.
6. Visible membership prevents retention collection and low-level deletion.
7. User deletion of a published item returns a typed 409 refusal.
8. Entry deletion never deletes bytes and no generic release authority exists.
9. Corrupt job-reference JSON fails closed.
10. Run and ordered-step mask references use the same write fence.

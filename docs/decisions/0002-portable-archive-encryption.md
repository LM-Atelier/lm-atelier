# Passphrase encryption for portable archives

Status: Accepted.

## Decision

Portable archives (project exports, output recipe bundles and backups) are encrypted with a passphrase using two pieces of the `cryptography` package. LM Atelier designs no cryptographic construction of its own.

- **Cobblestone-256** encrypts both the archive's body and its key slots. It implements the published C2SP chunked-encryption construction ([c2sp.org/chunked-encryption](https://c2sp.org/chunked-encryption)). Each 16 KiB chunk is authenticated in order, so a chunk that is moved, repeated or removed is refused. The final chunk is shorter than the others, so an archive that was cut off or extended is refused. A key commitment is checked before the first chunk, so a wrong key is refused before any data comes out.
- **Argon2id** turns a passphrase into a key-encryption key.

## Format, version 1

An encrypted archive is a clear header followed by the encrypted body.

The header holds a format magic, the format version, the suite, the archive kind (project, output recipe or backup) and one or two key slots. It reveals nothing about what the archive contains. Each slot records:

- who holds its passphrase;
- a random key id;
- the Argon2id salt, passes, lanes and memory, with memory in KiB;
- the archive's random 32-byte key, encrypted under the slot's key.

The body is encrypted under that archive key, and the whole header is bound to it. Changing any byte of the header therefore makes the body unreadable. Every archive gets a new random key, salt and key id, so the same input never encrypts the same way twice.

New archives use Argon2id with 256 MiB of memory, 3 passes and 4 lanes.

## Limits and refusals

These limits are checked before any key is derived:

- the format, version, suite and kind must be ones this version reads;
- an archive has one or two slots;
- Argon2id memory is between 64 MiB and 1 GiB, passes between 1 and 10, and lanes between 1 and 8.

The memory floor keeps a hostile archive from making passphrase guesses cheap. Opening an archive derives at most two keys.

A passphrase is the UTF-8 text exactly as entered, between 1 and 1024 bytes. It is never trimmed or normalized, so text typed with composed characters on one system and decomposed on another will not match. Plain ASCII is unaffected.

Every refusal is a fixed code with no detail:

| Code | Meaning |
| --- | --- |
| `archive-format-unsupported` | Not this format, or a version, suite or kind this version does not read |
| `archive-kind-mismatch` | An archive of another kind than the one asked for |
| `archive-limits-exceeded` | Settings outside the limits above, refused before any key is derived |
| `archive-passphrase-invalid` | A passphrase outside its bounds when sealing |
| `archive-key-derivation-failed` | The key could not be derived, for example for lack of memory |
| `invalid_passphrase_or_corrupt` | A wrong passphrase, or an archive that was altered, cut off or extended |

A wrong passphrase and a damaged archive deliberately share one refusal. A key that could not be derived is not reported as a wrong passphrase, because it says nothing about the passphrase.

Opening releases each chunk as soon as it is authenticated. Only the final check proves that nothing was cut off, so decrypted data is written somewhere it can be discarded, and kept only when opening succeeds.

## What a backup archive holds, version 1

Opening a backup archive gives one stream, laid out exactly as follows, so that a later version can still read a backup made by this one:

1. the eight bytes `LMABKUP\0`;
2. the header's length, four bytes, big-endian, from 1 to 65536;
3. the header, that many bytes of UTF-8 JSON;
4. the database copy, exactly the header's `database.size_bytes`;
5. the media zip, exactly the header's `media.size_bytes`, when `media` is not null;

and nothing after it. The header has exactly these keys: `format` (`"lm-atelier-backup"`), `version` (1), `created_at`, `app_version`, `schema_revision`, `database` and `media`. `database` is `{"size_bytes", "sha256"}`; `media` is the same, or null for a backup without media. The media zip is the one a recovery backup with media has.

The header lets a damaged stream be refused early; it is not trusted. Each part must match its size and SHA-256, the database must pass the same integrity checks a recovery backup does, its schema revision must be the header's, and the media zip must match the database's records exactly. A backup that fails any of these is refused as `backup-invalid`.

The stream is never held whole. While a backup is made, its plaintext exists only in a staging folder private to the account, and it is deleted once the sealed file has opened again, part by part, to exactly what was written. A backup is checked the same way: opened into that folder, checked, and deleted. Nothing is restored.

## Verification

- **Spec conformance.** Tests decode an archive by the published construction alone: plain Argon2id, HKDF-SHA-512 and AES-256-GCM, without the library's own decoder.
- **Argon2id.** The key derivation reproduces the RFC 9106 test vector.
- **Format stability.** An archive written by the first version is kept as written and must keep opening.
- **Refusals.** Each one is exercised through the real entry point. Header and limit refusals are shown to happen before any key is derived.
- **Packaged builds.** The packaged application's runtime self-test checks the RFC 9106 vector and an encryption round trip. A build that cannot load the encryption fails its release smoke test.

## Limits

Cobblestone is new in `cryptography` 50 and has no published audit. It follows a published specification, the conformance test above pins its behavior, the major version is pinned, and the suite field lets a later suite replace it.

Encryption protects archives copied off the machine. It does not encrypt the live workspace or database, and it cannot protect an archive from an account that is already compromised. A forgotten passphrase cannot be recovered.

"""Opaque consumer identity for the Shared Asset Library (item 58, Phase 0).

A consumer is one application profile's stable, opaque handle in the neutral
registry: claims and leases key on it, and one profile must never be able to
infer another's. The identity composes the installation-bound desktop
instance identity (already an HMAC over a private per-machine seed and the
resolved data root) with manifest-provided product facts, so distinct
products, channels, and builds sharing one machine and even one data root
never collide, and public code never hard-codes the identity of any
specialized edition. Callers pass every input explicitly; nothing here
discovers a desktop installation on its own.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Final, NoReturn

from .filesystem_links import AnchoredDirectory, AnchoredDirectoryError
from .instance_identity import InstanceIdentityError, instance_identity_from_directory

SCHEMA_ID: Final = "lm-atelier-shared-asset-consumer-v1"
INVALID_CONSUMER: Final = "shared asset consumer identity is invalid"
_CONTEXT: Final = b"lm-atelier-shared-asset-consumer-v1\x00"
# Manifest facts are lowercase tokens: product namespace ("lm-atelier"),
# release channel ("stable"), and a build fingerprint (version or hash).
_FACT: Final = re.compile(r"^[a-z0-9][a-z0-9._-]{0,79}$")


class SharedAssetConsumerError(ValueError):
    """Fixed non-echoing refusal for an underivable consumer identity."""


def _invalid() -> NoReturn:
    raise SharedAssetConsumerError(INVALID_CONSUMER)


def _require_fact(value: str) -> bytes:
    if not isinstance(value, str) or not _FACT.fullmatch(value):
        _invalid()
    return value.encode("ascii")


def _length_prefixed(part: bytes) -> bytes:
    # Unambiguous composition: without explicit lengths, ("ab", "c") and
    # ("a", "bc") would hash identically.
    return len(part).to_bytes(2, "big") + part


def _require_absolute_data_dir(data_dir: object) -> Path:
    """Validate the data root's SPELLING, without resolving it.

    Same shape as the shared store contract's root rule. Resolution is
    deliberately absent: it is the step that would follow a redirection and
    hide it from the acquisition that exists to refuse it.
    """

    if not isinstance(data_dir, Path):
        _invalid()
    try:
        chosen = data_dir.expanduser()
    except (OSError, RuntimeError, ValueError):
        _invalid()
    text = str(chosen)
    if not text or "\x00" in text or not chosen.is_absolute():
        _invalid()
    # A ".." component is REFUSED rather than normalised away. Two spellings
    # that walk to the same directory would otherwise derive two identities,
    # because the identity is bound to the name - and POSIX accepts ".." in the
    # walk while Windows refuses it, so the divergence is real.
    #
    # Refusing is the containment-preserving repair: collapsing it lexically
    # would turn "<root>/link/.." into "<root>", erasing the very component
    # acquisition exists to refuse. Refusing keeps one spelling per identity
    # and makes both platforms agree, which normalising cannot do safely.
    if ".." in chosen.parts:
        _invalid()
    return chosen


def derive_consumer_identity(
    *,
    data_dir: Path,
    product_namespace: str,
    channel: str,
    build_fingerprint: str,
) -> str:
    """Return the 64-hex opaque consumer identity for one profile.

    Deterministic for identical inputs; distinct whenever the installation
    seed, the resolved data root, or any manifest fact differs. Every
    failure - hostile facts, an unusable data root, a tampered instance
    seed - surfaces the fixed non-echoing refusal, never a path.
    """

    namespace = _require_fact(product_namespace)
    channel_fact = _require_fact(channel)
    fingerprint = _require_fact(build_fingerprint)
    if not isinstance(data_dir, Path):
        _invalid()
    # The spelling is validated but NOT resolved. Resolving first would follow
    # a linked or redirected root and hand the anchor only the final target,
    # erasing the very link this function's contract calls invalid - an anchor
    # can only refuse a redirection it is allowed to see. Component-wise
    # acquisition of the GIVEN spelling is what refuses it.
    #
    # The root is ACQUIRED, not tested. Deriving a consumer identity is not
    # the step that establishes a store, so an absent root refuses here -
    # otherwise a mistyped path silently mints a second consumer that owns no
    # claims and diverges from the store the caller meant. Acquisition also
    # refuses a link, a regular file, and anything whose ancestry is
    # redirected, so no separate precondition is needed.
    #
    # The same held directory is then carried THROUGH the derivation. A
    # pathname validated here and resolved again inside the delegate would be
    # two resolutions, and the gap between them was enough for the delegate's
    # own mkdir to establish a replacement root.
    root = _require_absolute_data_dir(data_dir)
    try:
        with AnchoredDirectory(root) as anchor:
            installation = instance_identity_from_directory(anchor)
    except (AnchoredDirectoryError, InstanceIdentityError):
        _invalid()
    except OSError:
        _invalid()
    digest = hashlib.sha256()
    digest.update(_CONTEXT)
    for part in (
        installation.encode("ascii"),
        namespace,
        channel_fact,
        fingerprint,
    ):
        digest.update(_length_prefixed(part))
    return digest.hexdigest()

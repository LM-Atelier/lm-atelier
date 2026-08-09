"""What a Reference is called, what it can be, and how a turn asks for one.

A Reference is a subject the user has taught the application about - a person, a
character, an object, a style - which they then name in a chat with `@`. These
are the contracts every later slice inherits, written before any table exists so
that the parts hardest to change are settled first.

The rule that shapes most of this: **the backend never rediscovers a Reference
by reparsing prompt text.** A request carries identifiers and roles as data. A
name that happens to appear in a sentence is prose, and treating prose as a
lookup key is how the wrong real person ends up in someone's image.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from enum import StrEnum

MAX_NAME = 120
MAX_SLUG = 64
MAX_ROLE = 60
MAX_REFERENCES_PER_TURN = 8

# A slug is an addressing token, not a display name: it can be recognised
# without ambiguity about where it ends, and leaves the display name free to
# carry whatever a person actually calls a subject.
#
# One separator, not several. If "ada_lovelace" and "ada-lovelace" were both
# valid, two subjects could occupy what a person reads as a single mention, and
# choosing between them would be a guess - the same ambiguity the Unicode
# normalisation below exists to prevent.
_SLUG = re.compile(r"[a-z0-9](?:[a-z0-9-]{0,62}[a-z0-9])?")
_SLUG_SEPARATORS = re.compile(r"[\s._-]+")
_SLUG_STRIPPED = re.compile(r"[^a-z0-9-]+")


class ReferenceKind(StrEnum):
    """What a subject is. A closed set, because compatibility depends on it.

    A workflow declares which kinds it can condition on, so an open vocabulary
    would mean a subject whose compatibility can never be decided. New kinds are
    a deliberate change here, not something a caller can invent.
    """

    PERSON = "person"
    CHARACTER = "character"
    OBJECT = "object"
    PRODUCT = "product"
    PLACE = "place"
    STYLE = "style"
    WARDROBE = "wardrobe"
    POSE = "pose"
    COMPOSITION = "composition"
    OTHER = "other"


class MentionSource(StrEnum):
    """How a Reference came to be in this turn.

    Recorded because the three differ in how much the user actually asserted.
    A typed mention is explicit; a picker selection is explicit; something
    inherited from context was never stated at all, and a later question about
    why an image contains someone must be able to tell those apart.
    """

    MENTION = "mention"
    PICKER = "picker"
    INHERITED_CONTEXT = "inherited_context"


class ReferenceError(ValueError):
    """A Reference request could not be understood, so it is not guessed at."""


def slugify_mention(name: str) -> str:
    """Derive the addressing token for a display name.

    Case-folded rather than lowercased: `casefold` handles scripts where the two
    differ, and a mention that resolves differently depending on how the name
    was typed would be a way to reach the wrong subject.
    """

    if not isinstance(name, str):
        raise ReferenceError("a name must be text")
    # Compatibility decomposition first, so visually identical characters from
    # different code points do not become two different subjects.
    folded = unicodedata.normalize("NFKD", name).casefold()
    ascii_only = folded.encode("ascii", "ignore").decode("ascii")
    collapsed = _SLUG_SEPARATORS.sub("-", ascii_only.strip())
    cleaned = _SLUG_STRIPPED.sub("", collapsed).strip("-")
    if not cleaned:
        raise ReferenceError(
            f"{name!r} has no characters that can form a mention; give the subject "
            "a mention name of its own"
        )
    return cleaned[:MAX_SLUG].rstrip("-")


def valid_mention_slug(value: str) -> bool:
    return isinstance(value, str) and bool(_SLUG.fullmatch(value))


def parse_kind(value: object) -> ReferenceKind:
    if isinstance(value, ReferenceKind):
        return value
    if not isinstance(value, str):
        raise ReferenceError("a reference kind must be text")
    try:
        return ReferenceKind(value.strip().casefold())
    except ValueError as error:
        permitted = ", ".join(sorted(item.value for item in ReferenceKind))
        raise ReferenceError(
            f"{value!r} is not a reference kind; expected one of: {permitted}"
        ) from error


@dataclass(frozen=True)
class ReferenceRequest:
    """One Reference a turn is asking for, as data rather than as text."""

    reference_subject_id: str
    role: str | None = None
    selected_asset_ids: tuple[str, ...] = ()
    strength: float | None = None
    source: MentionSource = MentionSource.MENTION


def _identifier(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ReferenceError(f"{field} must be a non-empty identifier")
    return value.strip()


def parse_reference_requests(value: object) -> tuple[ReferenceRequest, ...]:
    """Read the structured references a turn carries, or refuse the turn.

    Refusing matters more here than in most parsing. The fallback if this is
    unreliable is to recover References by reading the prompt for `@name`, and
    that path silently binds whoever the text most resembles. A request that
    cannot be read as data is an error, never an invitation to guess.
    """

    if value is None:
        return ()
    if not isinstance(value, list):
        raise ReferenceError("references must be a list")
    if len(value) > MAX_REFERENCES_PER_TURN:
        raise ReferenceError(f"a turn carries at most {MAX_REFERENCES_PER_TURN} references")

    requests: list[ReferenceRequest] = []
    seen: set[tuple[str, str | None]] = set()
    for entry in value:
        if not isinstance(entry, dict):
            raise ReferenceError("each reference must be an object")
        subject_id = _identifier(entry.get("reference_subject_id"), "reference_subject_id")

        raw_role = entry.get("role")
        role = None if raw_role is None else _identifier(raw_role, "role")[:MAX_ROLE]

        # The same subject twice in one role is a request that cannot be
        # honoured twice; in two different roles it is legitimate.
        key = (subject_id, role)
        if key in seen:
            raise ReferenceError(f"reference {subject_id} is requested twice for the same role")
        seen.add(key)

        raw_assets = entry.get("selected_asset_ids", [])
        if not isinstance(raw_assets, list):
            raise ReferenceError("selected_asset_ids must be a list")
        assets = tuple(dict.fromkeys(_identifier(item, "selected asset id") for item in raw_assets))

        strength = entry.get("strength")
        if strength is not None:
            if isinstance(strength, bool) or not isinstance(strength, (int, float)):
                raise ReferenceError("strength must be a number")
            if not 0.0 <= float(strength) <= 2.0:
                raise ReferenceError("strength must be between 0 and 2")
            strength = float(strength)

        raw_source = entry.get("source", MentionSource.MENTION.value)
        try:
            source = MentionSource(str(raw_source).strip().casefold())
        except ValueError as error:
            permitted = ", ".join(sorted(item.value for item in MentionSource))
            raise ReferenceError(
                f"{raw_source!r} is not a reference source; expected one of: {permitted}"
            ) from error

        requests.append(ReferenceRequest(subject_id, role, assets, strength, source))
    return tuple(requests)

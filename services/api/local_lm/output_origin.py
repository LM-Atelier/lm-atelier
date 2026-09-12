"""Which part of a workflow produced a file, as the engine itself said it.

An engine returns its outputs keyed by the graph node that wrote them. That key
is the only first-party statement of which file is which, it exists exactly once
- in the response - and nothing can reconstruct it afterwards. A file is stored
under a digest of its own content, so two nodes that happened to write identical
bytes are one row by then; and a run can return several files that differ only
in which branch of the graph made them.

This records what was said and reaches no conclusion from it. A preview branch
and a save branch are both outputs here: told apart, not judged. Deciding what a
preview means for a person, or which file a requested size refers to, needs this
fact and is not this module's business.

Everything here is total. An engine that attributes nothing, a key that is not a
usable identifier and a type outside the engine's own vocabulary each produce a
named record rather than an exception, because the alternative is an adapter
that fails a finished generation over a label.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Final

RECORD_VERSION: Final = 1

#: A node id is an identifier in the graph's own key space, not free text. The
#: bound matches what the ComfyUI adapter already applies to node ids and output
#: parameters, so a hostile history cannot widen what reaches the record.
MAX_IDENTIFIER: Final = 200

#: The engine's own closed vocabularies. `temp` is what a preview node writes
#: and the engine does not keep; `input` is a file that was handed to it.
OUTPUT_TYPES: Final = frozenset({"input", "output", "temp"})
COLLECTIONS: Final = frozenset({"images", "gifs", "videos"})


def usable_identifier(candidate: object) -> bool:
    """Whether a value can stand as the name of a node, rather than be trusted.

    Whitespace alone is refused as well as the empty string. A record reading
    `"node_id": " "` names nothing a person or a later check could use, and it
    would look like a defect in the recorder rather than like the engine
    answering oddly - which is what it actually is.
    """

    return (
        isinstance(candidate, str)
        and 0 < len(candidate) <= MAX_IDENTIFIER
        and candidate.strip() != ""
        and all(character >= " " for character in candidate)
    )


def _from_vocabulary(value: object, vocabulary: frozenset[str]) -> str | None:
    """A label this vocabulary contains, or nothing at all.

    Membership is asked only of a string. `value in frozenset` raises TypeError
    for a list or a dictionary, and ordinary JSON carries both, so asking
    directly would make these functions total for every input except the two
    shapes an engine is most likely to send by mistake.
    """

    return value if isinstance(value, str) and value in vocabulary else None


def stated_origin(node_id: object, output_type: object, collection: object) -> dict[str, Any]:
    """What an engine said about one file it returned, reduced to safe values.

    Called by an adapter while it still has the response in hand. Anything
    outside the closed vocabularies becomes None here rather than travelling
    further, so what the rest of the system sees is bounded by construction.
    """

    return {
        "node_id": node_id if usable_identifier(node_id) else None,
        "output_type": _from_vocabulary(output_type, OUTPUT_TYPES),
        "collection": _from_vocabulary(collection, COLLECTIONS),
    }


def names_a_preview(record: object) -> bool:
    """Whether this record says the file is a throwaway the engine does not keep.

    Only an ATTRIBUTED record can say so. An engine that named nothing, or a key
    that arrived unusable, leaves us not knowing which part of the workflow wrote
    the file - and treating "we do not know" as "it is a preview" would quietly
    discard somebody's picture. Absence of evidence decides nothing here.
    """

    if not isinstance(record, Mapping):
        return False
    return record.get("state") == "attributed" and record.get("output_type") == "temp"


def record_for(origin: object, engine: str) -> dict[str, Any]:
    """The record stored beside one produced file, or a named reason there is none.

    Two absences, kept apart on purpose. An engine that does not attribute its
    outputs at all is a fact about that engine and will be true of every file it
    ever returns; a key that arrived unusable is a fact about one response. A
    reader that collapsed them would learn nothing from either.
    """

    if not isinstance(origin, Mapping):
        return {
            "v": RECORD_VERSION,
            "state": "unattributed",
            "engine": engine,
            "reason": "engine_does_not_attribute",
        }
    node_id = origin.get("node_id")
    if not usable_identifier(node_id):
        return {
            "v": RECORD_VERSION,
            "state": "unattributed",
            "engine": engine,
            "reason": "node_id_unusable",
        }
    return {
        "v": RECORD_VERSION,
        "state": "attributed",
        "engine": engine,
        "node_id": node_id,
        "output_type": _from_vocabulary(origin.get("output_type"), OUTPUT_TYPES),
        "collection": _from_vocabulary(origin.get("collection"), COLLECTIONS),
    }

"""Is the picture the size that was asked for, and may we even say?

Three facts already exist separately: the size a run ASKED for, frozen into its
accepted inputs; the size the file actually IS, measured from its own bytes; and
whether the graph that RAN still carried the binding proving the declared width
and height decide the output. This puts them together and answers one question.

The answer is deliberately three-valued, and the third value carries a reason.
"They disagree", "we did not look" and "we had no method" are different things to
show a person, and a record that collapsed them would turn every unknown into an
accusation. So every combination that is not a clean comparison produces a NAMED
non-assessment, never a mismatch.

WHAT IT MUST NOT DO, and both are failure modes this area has already produced
once. It must never judge a preview node's throwaway - that file is not the
picture anybody asked for, and measuring it against the requested pair would warn
on a correct run. And it must never warn on a graph that legitimately produces a
different size, such as a hires-fix upscale or a scale after the decode: those
are exactly what the executed-graph confirmation refuses to confirm, so a warning
on one of them means this scope is wrong rather than the run.

Pure and total: no session, no engine, no exceptions. Every input is untrusted.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Final

from .output_origin import names_a_preview

RECORD_VERSION: Final = 1

#: Why a file was not judged. Closed, because a reader shows these to a person
#: and an open set would eventually put a stack trace beside a picture.
NOT_ASSESSED_REASONS: Final = frozenset(
    {
        "throwaway",
        "no_size_requested",
        "origin_unknown",
        "binding_unconfirmed",
        "unmeasured",
    }
)


def _requested_dimension(settings: object, key: str) -> int | None:
    """A width or a height as the run actually asked for it.

    Only a positive integer counts, and deliberately with NO upper bound. The
    ceiling belongs to the acceptance that produced these settings; re-imposing
    one here would let this module answer "nothing was requested" about a size
    the product had already accepted, which is a worse answer than any warning.

    `type(value) is int` rather than `isinstance` keeps `True` out: a settings
    map carrying `"width": True` is a defect somewhere else, and reading it as
    the number one would hide it behind a mismatch.
    """

    if not isinstance(settings, Mapping):
        return None
    value = settings.get(key)
    return value if type(value) is int and value > 0 else None


def _measured_dimension(measurement: object, key: str) -> int | None:
    if not isinstance(measurement, Mapping) or measurement.get("state") != "measured":
        return None
    value = measurement.get(key)
    return value if type(value) is int and value > 0 else None


def _not_assessed(reason: str) -> dict[str, Any]:
    return {"v": RECORD_VERSION, "state": "not_assessed", "reason": reason}


def size_agreement(
    *,
    requested_settings: object,
    measurement: object,
    origin: object,
    binding_confirmed: object,
) -> dict[str, Any]:
    """Compare what was asked for with what arrived, or name why we did not.

    `requested_settings` is the run's ACCEPTED settings - frozen when the turn
    was accepted - and never the artifact's own stored settings. Artifacts are
    content addressed, so two runs producing identical bytes are one row and its
    settings belong to whichever run reached it first; comparing a measurement
    against those would sometimes be comparing a run with a stranger.

    `binding_confirmed` is the caller's answer to whether a geometry proof exists
    for this revision AND the graph that was actually dispatched still carries
    it. It is passed in rather than computed because the proof is about the
    revision and the dispatch is about the run, and neither is this module's to
    fetch. Anything that is not exactly True is treated as unconfirmed - a
    caller that has not looked must not be read as having looked and found
    nothing wrong.

    The order of the refusals is chosen for what it tells a reader. Being a
    throwaway is categorical and comes first: nothing else about that file
    matters, because it was never the picture in question.
    """

    if names_a_preview(origin):
        return _not_assessed("throwaway")
    if not isinstance(origin, Mapping) or origin.get("state") != "attributed":
        return _not_assessed("origin_unknown")

    requested_width = _requested_dimension(requested_settings, "width")
    requested_height = _requested_dimension(requested_settings, "height")
    if requested_width is None or requested_height is None:
        return _not_assessed("no_size_requested")

    if binding_confirmed is not True:
        return _not_assessed("binding_unconfirmed")

    raster_width = _measured_dimension(measurement, "raster_width")
    raster_height = _measured_dimension(measurement, "raster_height")
    if raster_width is None or raster_height is None:
        return _not_assessed("unmeasured")

    agreed = (requested_width, requested_height) == (raster_width, raster_height)
    return {
        "v": RECORD_VERSION,
        "state": "agreed" if agreed else "disagreed",
        "requested_width": requested_width,
        "requested_height": requested_height,
        "raster_width": raster_width,
        "raster_height": raster_height,
    }


def disagrees(record: object) -> bool:
    """Whether this record is the one worth showing a person.

    A function rather than a comparison at each call site, because the states
    that must NOT produce a warning outnumber the one that must, and every
    reader that spelled out `!= "agreed"` would be warning on all of them.
    """

    return isinstance(record, Mapping) and record.get("state") == "disagreed"

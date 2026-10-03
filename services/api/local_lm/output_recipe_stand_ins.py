"""A picture a record's bundle carries, kept here to stand in for one that is not.

A new version can name the copy a bundle holds for one of the record's inputs
instead of a picture already here. The copy is kept the way a picture attached
to a turn is: stored by its own content as an input, outside the Media Library,
and kept for as long as a generation uses it. It is never the recorded picture:
its hash is the copy's own, so the run made with it names its inputs among what
differs from the record.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy.orm import Session

from .artifacts import ArtifactStore
from .domain import ArtifactKind
from .output_recipe_replay import BUNDLED_INPUT, AdaptationChoiceInvalid, AdaptationChoices


def store_bundled_stand_ins(
    session: Session,
    artifacts: ArtifactStore,
    bundle: dict[str, Any] | None,
    choices: AdaptationChoices,
) -> AdaptationChoices:
    """Keep each chosen bundled copy here and name it in place of the choice for it.

    A choice of a bundled copy for a position the bundle does not carry, or for
    a file that is not a bundle at all, names nothing the person can have, and
    is refused before anything is kept. The caller commits.
    """

    wanted = sorted(
        position for position, value in choices.inputs.items() if value == BUNDLED_INPUT
    )
    if not wanted:
        return choices
    carried = {item["position"]: item for item in (bundle or {}).get("inputs", [])}
    if any(position not in carried for position in wanted):
        raise AdaptationChoiceInvalid
    inputs = dict(choices.inputs)
    for position in wanted:
        artifact = artifacts.ingest_bytes(
            session,
            carried[position]["picture"],
            kind=ArtifactKind.INPUT,
            media_type="image/png",
            original_name=f"record-input-{position + 1}.png",
            metadata={"uploaded": True},
        )
        inputs[position] = artifact.id
    return choices._replace(inputs=inputs)

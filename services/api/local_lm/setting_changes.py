"""What an accepted render changed on purpose, said so a client can show it.

A render that is accepted - a turn, a regenerate, a branch, an edit or a
Prompt Library batch - can run with a value other than one it was given or
carried over: a saved value its workflow does not take, a count made another
way, a strength or a length its workflow decides, or a seed drawn again or
stepped. Its answer names each such change in `setting_changes`: the setting,
a code from one closed vocabulary with that code's one fixed sentence, where
the value came from when the request did not send it, and, where they can be
named, the value it was given and the value used. A render that changed
nothing answers without the list.

A value is named only when it is true or false, a finite number, or text that
is one of the choices its setting offers, by the rule a refusal names choices
by; any other value is left out, never echoed. The words of a prompt are never
named. A key that does not read as a plain setting name is given as `*`, with
no value, and so are more settings than one item names, such as more than one
request may name, sent with a reply accepting an offer.

Each run keeps the changes it was accepted with in its provenance. The answer
for a turn is every change its runs keep, in the order of their steps and
each once; a change that only some steps of a plan were accepted with names
each of those steps, counting from 1. A replayed request answers with the
same list.
"""

from __future__ import annotations

import json
from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Final

from pydantic import BaseModel, Field, ValidationError

from .setting_refusals import UNNAMED_SETTING, SettingSource, nameable_value, setting_field_name
from .workflow_use_case_presets_v1 import PROMPT_SETTING_KEYS

#: Where a run's provenance keeps the changes it was accepted with.
SETTING_CHANGES_KEY: Final = "setting_changes"


class SettingChangeCode(StrEnum):
    """Every reason an accepted render changed a value it was given or carried over."""

    NOT_TAKEN = "set-aside-not-taken"
    NOT_OFFERED = "set-aside-not-offered"
    FIXED = "set-aside-fixed"
    NOT_USABLE = "set-aside-not-usable"
    SIZE_FOLLOWS_SOURCE = "size-follows-source"
    FOR_OFFER = "set-aside-for-offer"
    OUTPUTS_FROM_BATCH_SIZE = "outputs-from-batch-size"
    OUTPUT_COUNT_RAISED = "output-count-raised"
    ONE_OUTPUT = "one-output"
    STRENGTH_DEFAULT_FOR_NEW_PICTURE = "strength-default-for-new-picture"
    STRENGTH_FULL_FOR_EXTENSION = "strength-full-for-extension"
    LENGTH_ALIGNED_TO_FRAMES = "length-aligned-to-frames"
    SEED_DRAWN_AGAIN = "seed-drawn-again"
    SEED_PER_OUTPUT = "seed-per-output"
    SEED_PER_DRAFT = "seed-per-draft"


#: The one sentence said about each change. None of them names a value.
SETTING_CHANGE_MESSAGES: Final[Mapping[SettingChangeCode, str]] = {
    SettingChangeCode.NOT_TAKEN: (
        "The chosen workflow takes no setting by this name, so this saved value was set aside."
    ),
    SettingChangeCode.NOT_OFFERED: (
        "The chosen workflow sets this itself, so this saved value was set aside."
    ),
    SettingChangeCode.FIXED: (
        "The chosen workflow fixes this setting itself, so the value given was set aside."
    ),
    SettingChangeCode.NOT_USABLE: (
        "This saved value is not one the chosen workflow takes, so one saved below it, or the "
        "workflow's own, was used."
    ),
    SettingChangeCode.SIZE_FOLLOWS_SOURCE: (
        "Work made from a picture keeps that picture's shape, so this saved size was set aside."
    ),
    SettingChangeCode.FOR_OFFER: (
        "Accepting an offer makes what was offered, so the settings, ordered settings and count "
        "the reply sent were set aside; a role override for the offered role still applies."
    ),
    SettingChangeCode.OUTPUTS_FROM_BATCH_SIZE: (
        "Each output is made on its own, so each one is made as a batch of one."
    ),
    SettingChangeCode.OUTPUT_COUNT_RAISED: (
        "This turn makes more outputs than the count it was given, as its words or its batch size "
        "ask for."
    ),
    SettingChangeCode.ONE_OUTPUT: (
        "A text reply is written once, so the count it was given was not used."
    ),
    SettingChangeCode.STRENGTH_DEFAULT_FOR_NEW_PICTURE: (
        "A picture made from words takes its change strength's default, so the strength given "
        "was set aside."
    ),
    SettingChangeCode.STRENGTH_FULL_FOR_EXTENSION: (
        "An extension paints its new canvas at full strength, so the strength given was set aside."
    ),
    SettingChangeCode.LENGTH_ALIGNED_TO_FRAMES: (
        "The chosen workflow makes whole frames, so the length was brought to the nearest one it "
        "makes."
    ),
    SettingChangeCode.SEED_DRAWN_AGAIN: (
        "A regenerate draws a new seed, so the seed given was set aside."
    ),
    SettingChangeCode.SEED_PER_OUTPUT: (
        "Each output after the first takes the next seed after the one given, so no two are alike."
    ),
    SettingChangeCode.SEED_PER_DRAFT: (
        "Each draft of a batch takes a seed of its own, so this saved seed was set aside."
    ),
}


@dataclass(frozen=True)
class SettingChange:
    """One value a render changed on purpose, before it is said to a client.

    `key` is the setting as the request or the saved layer named it, or a
    request field such as `output_count`; it passes through
    `setting_field_name` before it is said. It is None for more settings than
    one item names, said as `*` and never said to be carried over. `source` is
    where a value the request did not send came from. `before` is the value it
    was given and `after` the value used, each None when there is none to name,
    and `choices` are the choices its setting offers, which decide whether a
    value that is text may be named.
    """

    code: SettingChangeCode
    key: str | None
    source: SettingSource | None = None
    before: object = None
    after: object = None
    choices: tuple[Any, ...] | None = None

    def item(self, sent: Collection[str] | None = None) -> dict[str, Any]:
        """This change as a run keeps it, naming only the values it may name.

        `sent` is every setting key the request itself named, with the request
        fields it gave, such as `output_count`. A change of a value with no
        source of its own whose key is outside it changed a value carried over
        from the turn the request starts from. With no `sent`, every such value
        is the request's own.
        """

        field = setting_field_name(self.key)
        item: dict[str, Any] = {"code": self.code.value, "field": field}
        source = self.source
        if source is None and sent is not None and self.key is not None and self.key not in sent:
            source = SettingSource.INHERITED
        if source is not None:
            item["source"] = source.value
        if field != UNNAMED_SETTING and self.key not in PROMPT_SETTING_KEYS:
            for name, value in (("before", self.before), ("after", self.after)):
                if nameable_value(value, self.choices):
                    item[name] = value
        return item


def _one_of_each(items: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Items by what they say, so the same change is kept once, where it first appears."""

    kept: dict[str, dict[str, Any]] = {}
    for item in items:
        kept.setdefault(json.dumps(item, sort_keys=True), item)
    return kept


def setting_changes_provenance(
    changes: Iterable[SettingChange], *, sent: Collection[str] | None = None
) -> dict[str, Any]:
    """What a run's provenance keeps for these changes: each once, or nothing when there is none."""

    items = list(_one_of_each(change.item(sent) for change in changes).values())
    return {SETTING_CHANGES_KEY: items} if items else {}


def _absent(value: object) -> bool:
    return value is None


class SettingChangeOut(BaseModel):
    """One value an accepted render changed on purpose, and why; never a value it may not name."""

    code: SettingChangeCode
    field: str
    message: str
    source: SettingSource | None = Field(default=None, exclude_if=_absent)
    before: bool | int | float | str | None = Field(default=None, exclude_if=_absent)
    after: bool | int | float | str | None = Field(default=None, exclude_if=_absent)
    step: int | None = Field(default=None, ge=1, exclude_if=_absent)


def _kept(value: object) -> list[dict[str, Any]]:
    """The changes one run keeps, leaving out anything that is not one."""

    if not isinstance(value, list):
        return []
    codes = {code.value for code in SettingChangeCode}
    return [item for item in value if isinstance(item, dict) and item.get("code") in codes]


def turn_setting_changes(held: Sequence[object]) -> list[SettingChangeOut]:
    """Every change a turn's runs keep, each once, in the order of their steps.

    `held` is what each run's provenance keeps, one entry for each step in
    order. A change every step was accepted with is named once; one that only
    some steps were accepted with is named for each of them, with its step.
    """

    steps = [_one_of_each(_kept(value)) for value in held]
    holders: dict[str, int] = {}
    for items in steps:
        for said in items:
            holders[said] = holders.get(said, 0) + 1
    changes: list[SettingChangeOut] = []
    for number, items in enumerate(steps, start=1):
        for said, item in items.items():
            # Every step holds it, so the first does, and it is named there alone.
            everywhere = holders[said] == len(steps)
            if everywhere and number > 1:
                continue
            try:
                changes.append(
                    SettingChangeOut.model_validate(
                        {
                            **item,
                            "message": SETTING_CHANGE_MESSAGES[SettingChangeCode(item["code"])],
                            "step": None if everywhere else number,
                        }
                    )
                )
            except ValidationError:
                continue
    return changes

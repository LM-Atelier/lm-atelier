"""Why a setting was refused, said so a client can act on it without reading prose.

A refused render request keeps the `detail` and `code` it always answered
with, and gains `errors`: one item per refusal, each naming the setting, a code
from one closed vocabulary and that code's one fixed sentence, and, where they
apply, the limit the value passed, the choices it may take, and where a value
the request did not send came from.

An item never carries the value that was refused, or any text the request
sent. A setting's key is named back only when it reads as a plain setting
name; any other key is given as `*`. Choices are named back only when every
one shown is a number, true or false, or a plain word, since a choice can be a
file name or free text.
"""

from __future__ import annotations

import math
import re
from collections.abc import Collection, Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Final

from pydantic import BaseModel, Field

#: A key is named back only when it has this shape; anything else is `*`.
SETTING_NAME: Final = re.compile(r"[a-z][a-z0-9_]{0,63}")
#: Stands for a setting an answer does not name.
UNNAMED_SETTING: Final = "*"
#: A choice that is text is named back only when it has this shape.
_PLAIN_CHOICE: Final = re.compile(r"[A-Za-z0-9][A-Za-z0-9_:+-]{0,63}")
#: The most choices an item names. A longer menu is cut to its first ones.
MAX_ALLOWED_CHOICES: Final = 50


class SettingRefusalCode(StrEnum):
    """Every reason a setting can be refused for."""

    UNKNOWN = "setting-unknown"
    NOT_OFFERED = "setting-not-offered"
    NOT_AN_INTEGER = "setting-not-an-integer"
    NOT_A_NUMBER = "setting-not-a-number"
    NOT_A_STRING = "setting-not-a-string"
    NOT_A_BOOLEAN = "setting-not-a-boolean"
    NOT_AN_ARRAY = "setting-not-an-array"
    NOT_AN_OBJECT = "setting-not-an-object"
    NOT_A_CHOICE = "setting-not-a-choice"
    BELOW_MINIMUM = "setting-below-minimum"
    ABOVE_MAXIMUM = "setting-above-maximum"
    NOT_A_MULTIPLE = "setting-not-a-multiple"
    NOT_FINITE = "setting-not-finite"
    TOO_LONG = "setting-too-long"
    TOO_MANY_ITEMS = "setting-too-many-items"
    TOO_DEEP = "setting-too-deep"
    TOO_COMPLEX = "setting-too-complex"
    TOO_MANY_SETTINGS = "settings-too-many"
    NOT_APPLICABLE = "setting-not-applicable"
    NEEDS_ITS_PAIR = "setting-needs-its-pair"


#: The one sentence said about each reason. None of them names a value.
SETTING_REFUSAL_MESSAGES: Final[Mapping[SettingRefusalCode, str]] = {
    SettingRefusalCode.UNKNOWN: "The chosen model and workflow have no setting by this name.",
    SettingRefusalCode.NOT_OFFERED: "The chosen workflow does not offer this setting.",
    SettingRefusalCode.NOT_AN_INTEGER: "This setting takes a whole number.",
    SettingRefusalCode.NOT_A_NUMBER: "This setting takes a number.",
    SettingRefusalCode.NOT_A_STRING: "This setting takes text.",
    SettingRefusalCode.NOT_A_BOOLEAN: "This setting takes true or false.",
    SettingRefusalCode.NOT_AN_ARRAY: "This setting takes a list.",
    SettingRefusalCode.NOT_AN_OBJECT: "This setting takes a set of named values.",
    SettingRefusalCode.NOT_A_CHOICE: "This setting takes only one of the choices it offers.",
    SettingRefusalCode.BELOW_MINIMUM: "This setting is below its minimum, given as the limit.",
    SettingRefusalCode.ABOVE_MAXIMUM: "This setting is above its maximum, given as the limit.",
    SettingRefusalCode.NOT_A_MULTIPLE: (
        "This setting is not a multiple of its step, given as the limit."
    ),
    SettingRefusalCode.NOT_FINITE: (
        "This setting holds a number that is not finite, or is too large to use."
    ),
    SettingRefusalCode.TOO_LONG: (
        "This setting holds text longer than the limit allows, in characters."
    ),
    SettingRefusalCode.TOO_MANY_ITEMS: (
        "This setting holds a list or a set of named values with more items than the limit allows."
    ),
    SettingRefusalCode.TOO_DEEP: "This setting nests values deeper than the limit allows.",
    SettingRefusalCode.TOO_COMPLEX: "This setting holds more values in all than the limit allows.",
    SettingRefusalCode.TOO_MANY_SETTINGS: "This request names more settings than the limit allows.",
    SettingRefusalCode.NOT_APPLICABLE: (
        "This setting cannot be applied with the chosen workflow and the other choices."
    ),
    SettingRefusalCode.NEEDS_ITS_PAIR: "A size is applied whole: width and height together.",
}


class SettingSource(StrEnum):
    """Where a refused value came from, when the request did not send it."""

    #: Carried over from the turn a regenerate, an edit or a branch starts from.
    INHERITED = "inherited"
    TURN_PRESET = "turn_preset"
    CHAT = "chat"
    CHAT_PRESET = "chat_preset"
    PROJECT = "project"
    PROJECT_PRESET = "project_preset"
    DEFAULT_PRESET = "default_preset"
    PROFILE = "profile"
    #: A saved layer this answer cannot name.
    SAVED = "saved"


#: How a refusal of a saved size words each layer, and the source it names.
_SAVED_SOURCES: Final[Mapping[str, SettingSource]] = {
    "in the preset chosen for this turn": SettingSource.TURN_PRESET,
    "for this conversation": SettingSource.CHAT,
    "in this conversation's preset": SettingSource.CHAT_PRESET,
    "for this project": SettingSource.PROJECT,
    "in this project's preset": SettingSource.PROJECT_PRESET,
    "in the default preset": SettingSource.DEFAULT_PRESET,
    "in the model profile": SettingSource.PROFILE,
}


def saved_setting_source(where: str) -> SettingSource:
    """The source a saved layer is named by, from the words a refusal uses for it."""

    return _SAVED_SOURCES.get(where, SettingSource.SAVED)


def setting_field_name(key: str | None) -> str:
    """A key as an answer may name it: itself when it reads as a setting name, else `*`."""

    return key if key is not None and SETTING_NAME.fullmatch(key) else UNNAMED_SETTING


def _number(value: float) -> int | float:
    """A limit as a whole number when it is one, so 2048.0 reads as 2048."""

    if isinstance(value, float) and math.isfinite(value) and value.is_integer():
        return int(value)
    return value


def _plain_number(value: int | float) -> bool:
    """Whether a number can be named as it is: finite, and small enough to be a float."""

    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _named_choices(choices: Iterable[Any] | None) -> list[str | int | float | bool] | None:
    """The first choices of a menu, or none when one shown could be a file name or free text.

    At most MAX_ALLOWED_CHOICES are named, in the order the setting offers them;
    a longer menu is cut there, so a client reading exactly that many knows it
    may be longer.
    """

    if choices is None:
        return None
    named: list[str | int | float | bool] = []
    for choice in list(choices)[:MAX_ALLOWED_CHOICES]:
        if (
            isinstance(choice, bool)
            or (isinstance(choice, int | float) and _plain_number(choice))
            or (isinstance(choice, str) and _PLAIN_CHOICE.fullmatch(choice))
        ):
            named.append(choice)
        else:
            return None
    return named or None


@dataclass(frozen=True)
class SettingRefusal:
    """One reason one setting was refused, before it is said to a client.

    `key` is the key as the request or the saved layer held it; it is passed
    through `setting_field_name` before it is ever said. It is None only for a
    refusal of the settings as a whole.
    """

    code: SettingRefusalCode
    key: str | None
    limit: float | None = None
    allowed: tuple[Any, ...] | None = None
    source: SettingSource | None = None

    def item(self, sent: Collection[str] | None = None) -> dict[str, Any]:
        """This refusal as an error item, without the value that was refused.

        `sent` is every setting key the request itself named. A refused key
        outside it was carried over from the turn the request starts from. With
        no `sent`, every value without a saved source is the request's own.
        """

        item: dict[str, Any] = {
            "code": self.code.value,
            "field": setting_field_name(self.key),
            "message": SETTING_REFUSAL_MESSAGES[self.code],
        }
        if self.limit is not None:
            item["limit"] = _number(self.limit)
        allowed = _named_choices(self.allowed)
        if allowed is not None:
            item["allowed"] = allowed
        source = self.source
        if source is None and sent is not None and self.key is not None and self.key not in sent:
            source = SettingSource.INHERITED
        if source is not None:
            item["source"] = source.value
        return item


class SettingRefused(ValueError):
    """A request refused for the value of one or more settings, each with why.

    The message is the one the first refusal has always been worded with, so
    a reader of the message alone sees what it always saw. `refusals` holds
    every refusal that was found, not only the first.
    """

    def __init__(self, message: str, refusals: Iterable[SettingRefusal] = ()) -> None:
        super().__init__(message)
        self.refusals = tuple(refusals)


def setting_errors(
    exc: BaseException, *, sent: Collection[str] | None = None
) -> list[dict[str, Any]]:
    """The error items an exception's refusals come to; none for any other exception."""

    if not isinstance(exc, SettingRefused):
        return []
    return [refusal.item(sent) for refusal in exc.refusals]


def setting_error_body(
    exc: BaseException, *, sent: Collection[str] | None = None
) -> dict[str, Any]:
    """What an error body adds for these refusals: `errors`, only when there is one to name."""

    errors = setting_errors(exc, sent=sent)
    return {"errors": errors} if errors else {}


def _absent(value: object) -> bool:
    return value is None


class SettingErrorOut(BaseModel):
    """One setting a request was refused for, and why; never the value that was refused."""

    code: SettingRefusalCode
    field: str
    message: str
    limit: int | float | None = Field(default=None, exclude_if=_absent)
    allowed: list[str | int | float | bool] | None = Field(default=None, exclude_if=_absent)
    source: SettingSource | None = Field(default=None, exclude_if=_absent)


def setting_error_models(refusals: Iterable[SettingRefusal]) -> list[SettingErrorOut]:
    """The same items as `setting_errors`, for a response model that carries them."""

    return [SettingErrorOut.model_validate(refusal.item()) for refusal in refusals]

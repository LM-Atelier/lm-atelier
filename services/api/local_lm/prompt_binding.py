"""Whether what someone typed can reach the model at all.

A ComfyUI graph receives the description through the `${prompt}` placeholder,
and the compiler substitutes a placeholder only when a whole input value is
exactly that text - `ComfyUIAdapter._compile` reads `value[2:-1]` off a string
it has already required to start with `${` and end with `}`. A graph that names
`${prompt}` nowhere therefore receives nothing of what was typed, however long
the description was.

What makes that worth refusing rather than merely noting is the shape it takes
in practice: a graph that HAS a prompt input and holds somebody else's words in
it. Ask such a workflow for an autumn forest at sunrise and it returns the
night-time city its author baked in, every time, while the description sits in
the conversation beside the result as though it had been used. It looks like a
bad model rather than a workflow that was never asked the question.

A graph naming no prompt input at all is a different thing and is left alone. It
may be a fragment, a stub, or a workflow driven entirely by other inputs; what it
is not is a workflow that appears to take a description and quietly discards it.
Refusing it would mean refusing every graph that simply has no text in it, which
is a much larger claim than the one this evidence supports.

This is the same rule as the reference count in media_references.py, applied to
the other input: read the compiled graph for what it actually consumes, and
refuse before the turn runs rather than presenting an unrelated result as an
answer.
"""

from __future__ import annotations

import re
from collections.abc import Collection, Mapping
from typing import Any

from .settings_registry import IMAGE_SETTINGS, VIDEO_SETTINGS, workflow_settings

#: The exact string a ComfyUI input must equal for the description to replace
#: it. Embedding it in a longer value does not substitute, so a longer value is
#: not a binding.
PROMPT_PLACEHOLDER = "${prompt}"


def binds_prompt(workflow: dict[str, Any]) -> bool:
    """Whether this compiled graph takes the description anywhere.

    Walks the graph rather than the declared schema, because the graph is what
    runs and most workflows declare no prompt property at all - the description
    is not one of the settings a schema describes.
    """
    stack: list[Any] = [workflow]
    while stack:
        value = stack.pop()
        if isinstance(value, dict):
            stack.extend(value.values())
        elif isinstance(value, list):
            stack.extend(value)
        elif value == PROMPT_PLACEHOLDER:
            return True
    return False


#: The input names a ComfyUI node gives the words it is to render. `prompt` is
#: what a single-node video or image model takes; `text` is what CLIPTextEncode
#: and its relatives take. A graph using neither has no prompt to freeze.
PROMPT_INPUT_NAMES = frozenset({"prompt", "text"})


#: Names that can be in the parameter map without appearing among the settings
#: this module reads, so a graph may bind one and still be binding something.
#:
#: `prompt`, `negative_prompt`, `input_image` and `input_images` are reserved
#: against a workflow declaring them, so they can only ever come from
#: `ComfyUIAdapter._request_parameters` - which writes the first two for every
#: turn and the picture names when pictures ride. `mask` rides only for a
#: workflow that declares a mask input of its own, so it would be found among
#: the settings anyway; it is named here because the adapter is what puts it in
#: the map. `loras` is the odd one: the orchestrator writes it beside the
#: declared settings, and it can be present on a revision whose schema never
#: mentions it.
#:
#: All of them are listed unconditionally, because whether one rides is a
#: runtime fact this module never sees and naming one too many only ever makes a
#: refusal LESS likely. That asymmetry decides every judgement call here: a
#: wrong refusal makes a working workflow unusable, while a missed one leaves
#: the behaviour that already shipped.
RUN_SUPPLIED_KEYS = frozenset(
    {"prompt", "negative_prompt", "mask", "input_image", "input_images", "loras"}
)
_NUMBERED_INPUT_IMAGE = re.compile(r"input_image_[0-9]+")


def substitutable_keys(operation: str, input_schema: object) -> frozenset[str] | None:
    """The names a `${...}` value can carry for this turn, or None if unreadable.

    `ComfyUIAdapter._compile` substitutes a whole value only when the text
    between the braces is a KEY of the parameter map, so that map is the
    contract, and a value naming anything else stays exactly as written.

    Returning None means "I could not read this schema, so I make no claim about
    names" - the caller then falls back to the shape test, which is what shipped.
    That is deliberate, and it is why the failure is swallowed whatever it turns
    out to be. This is consulted while listing workflows, and it is the first
    thing on that path to read a stored settings schema, so one unreadable row
    would otherwise take the whole catalog down. Naming the failures that were
    measured would not have been enough: an integer too large to become a float,
    stored as a frame rate, raises OverflowError from the length contract - a
    type nobody would have thought to list. The question asked here is only
    about names, its answer when unsure is already "none", and no answer is
    worth an empty catalog.
    """
    if input_schema is not None and not isinstance(input_schema, Mapping):
        return None
    base = VIDEO_SETTINGS if operation == "text_to_video" else IMAGE_SETTINGS
    try:
        offered = workflow_settings(base, input_schema)
    except Exception:
        return None
    # `workflow_settings` answers a different question: which controls to OFFER.
    # It drops an engine setting the workflow marks read-only, and a length
    # contract makes it drop the frame count and the frame rate in favour of a
    # duration. The run supplies all of them anyway, so taking its answer alone
    # would call a real binding the author's own words. The engine's own
    # settings go back in for that reason, whatever the panel does with them.
    # A setting the workflow declares for itself needs no such repair: only the
    # engine's own are filtered this way, so those come through untouched.
    offered_keys = {field.key for field in offered}
    return frozenset(offered_keys | {field.key for field in base} | RUN_SUPPLIED_KEYS)


def _substitutes(value: str, substitutable: Collection[str] | None) -> bool:
    """Whether the compiler would replace this whole value with a parameter."""
    if not (value.startswith("${") and value.endswith("}")):
        return False
    if substitutable is None:
        return True
    name = value[2:-1]
    return name in substitutable or _NUMBERED_INPUT_IMAGE.fullmatch(name) is not None


def _prompt_values(workflow: dict[str, Any]) -> list[str]:
    """Every string a node offers where the description belongs, blanks aside."""
    if type(workflow) is not dict:
        return []
    values: list[str] = []
    for node in workflow.values():
        if not isinstance(node, dict):
            continue
        inputs = node.get("inputs")
        if not isinstance(inputs, dict):
            continue
        for name, value in inputs.items():
            if name in PROMPT_INPUT_NAMES and isinstance(value, str) and value.strip():
                values.append(value)
    return values


def frozen_prompt(workflow: dict[str, Any], substitutable: Collection[str] | None) -> bool:
    """Whether some node holds words of its own where the description should go.

    A literal here is only evidence when nothing binds `${prompt}` anywhere -
    a graph that binds the positive prompt and hard-codes a negative one is
    ordinary, and `ignores_the_description` checks the binding first.

    `substitutable` is the set of names the compiler would replace. It has to be
    asked rather than guessed from the shape of the text: a value like
    `${prompt}${negative_prompt}` opens and closes like a placeholder and is not
    one - the compiler looks up the whole inner text, finds no such key, and
    sends the braces to the model verbatim. Testing the shape instead let four
    such values through. Testing the NAME cannot be done with a pattern
    either. A setting key is not an identifier - it is printable text up to a
    length cap, minus a handful of reserved names - so a workflow can legally
    declare a key spelled exactly like two placeholders joined together, and
    then the compiler really does substitute it.
    """
    return any(not _substitutes(value, substitutable) for value in _prompt_values(workflow))


def ignores_the_description(
    engine: str, operation: str, workflow: dict[str, Any], input_schema: object
) -> bool:
    """Whether this turn's description would be discarded for one of its own.

    Narrow on purpose, and each condition earns its place.

    The engine must be `comfyui`, because the substitution rule above is that
    adapter's. Another engine binds its inputs its own way, and a graph it
    understands is not this module's to judge.

    The operation must be one that starts from a description alone. An edit or
    an animation has a source picture to work from, so a graph that uses only
    the picture is a legitimate workflow - an upscale or a restoration - rather
    than one that ignores its input.

    And the graph must actually hold a prompt of its own. Refusing every graph
    that merely fails to bind `${prompt}` would refuse fragments and stubs that
    were never offering to take a description, which is a wider claim than the
    defect supports; refusing one that holds somebody else's words in the input
    the description belongs in is exactly the defect.

    Whether a value is somebody's own words is decided against the workflow's
    declared settings, because that is what decides it for the compiler too.
    """
    if engine != "comfyui" or operation not in {"text_to_image", "text_to_video"}:
        return False
    if binds_prompt(workflow):
        return False
    values = _prompt_values(workflow)
    if not values:
        return False
    if any(not _substitutes(value, None) for value in values):
        # Words that do not even have the shape of a placeholder. No name could
        # make the compiler replace them, so the schema is never read.
        return True
    # Everything here opens and closes like a placeholder, and only the declared
    # names can say which of them the compiler would really replace. This is the
    # one case that needs the schema, which matters because the question is
    # asked of every workflow in the list.
    return frozen_prompt(workflow, substitutable_keys(operation, input_schema))

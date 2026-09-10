"""Whether what someone typed can reach the model at all.

A ComfyUI graph receives the description through the `${prompt}` placeholder,
and the compiler substitutes a placeholder only when a whole input value is
exactly that text - `ComfyUIAdapter._compile` reads `value[2:-1]` off a string
it has already required to start with `${` and end with `}`. A graph that names
`${prompt}` nowhere therefore receives nothing of what was typed, however long
the description was.

For a text-to-image or text-to-video turn there is nothing else to go on: with
no source picture and no description reaching the sampler, the workflow returns
whatever its own baked-in values produce, the same result for every request.
The description is still shown in the conversation beside it, which is the part
that makes this hard to see - the answer looks like a bad model rather than a
workflow that was never asked the question.

This is the same rule as the reference count in media_references.py, applied to
the other input: read the compiled graph for what it actually consumes, and
refuse before the turn runs rather than presenting an unrelated result as an
answer.
"""

from __future__ import annotations

from typing import Any

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


def ignores_the_description(engine: str, operation: str, workflow: dict[str, Any]) -> bool:
    """Whether this turn's whole input would be discarded before it ran.

    Narrow on purpose, and each condition earns its place.

    The engine must be `comfyui`, because the substitution rule above is that
    adapter's. Another engine binds its inputs its own way, and a graph it
    understands is not this module's to judge.

    The operation must be one that starts from a description alone. An edit or
    an animation has a source picture to work from, so a graph that uses only
    the picture is a legitimate workflow - an upscale or a restoration - rather
    than one that ignores its input.
    """
    return (
        engine == "comfyui"
        and operation in {"text_to_image", "text_to_video"}
        and not binds_prompt(workflow)
    )

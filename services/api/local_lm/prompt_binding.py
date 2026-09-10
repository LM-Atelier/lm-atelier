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


#: The input names a ComfyUI node gives the words it is to render. `prompt` is
#: what a single-node video or image model takes; `text` is what CLIPTextEncode
#: and its relatives take. A graph using neither has no prompt to freeze.
PROMPT_INPUT_NAMES = frozenset({"prompt", "text"})


def frozen_prompt(workflow: dict[str, Any]) -> bool:
    """Whether some node holds words of its own where the description should go.

    A literal here is only evidence when nothing binds `${prompt}` anywhere -
    a graph that binds the positive prompt and hard-codes a negative one is
    ordinary, and `ignores_the_description` checks the binding first.
    """
    if type(workflow) is not dict:
        return False
    for node in workflow.values():
        if not isinstance(node, dict):
            continue
        inputs = node.get("inputs")
        if not isinstance(inputs, dict):
            continue
        for name, value in inputs.items():
            if name not in PROMPT_INPUT_NAMES or not isinstance(value, str):
                continue
            # A placeholder for something else - `${style}`, say - is a binding
            # to that other input, not words of the graph author's own.
            if value.strip() and not (value.startswith("${") and value.endswith("}")):
                return True
    return False


def ignores_the_description(engine: str, operation: str, workflow: dict[str, Any]) -> bool:
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
    """
    return (
        engine == "comfyui"
        and operation in {"text_to_image", "text_to_video"}
        and not binds_prompt(workflow)
        and frozen_prompt(workflow)
    )

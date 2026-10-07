"""Whether a compiled graph offers a placeholder for a named parameter.

One question, asked by several callers: the prompt binding, the video length
contract and the edit calibration all need to know whether the graph takes a
value they declare. It lives in a module of its own because the obvious home -
`prompt_binding`, which already asked it for the description - imports
`settings_registry`, which imports `video_length`, so putting it there is a
circular import. A primitive with no dependencies belongs below all three.
"""

from __future__ import annotations

from typing import Any


def binds_parameter(workflow: object, name: str) -> bool:
    """Whether this graph offers `${name}` anywhere a value can sit.

    Matches the compiler exactly. `ComfyUIAdapter._compile` replaces a value
    only when the WHOLE value is the placeholder, so a name appearing inside a
    longer string is not a binding and must not be read as one.

    WHAT THIS DOES NOT PROVE. It answers whether the graph takes the value
    ANYWHERE, not whether the value decides anything. A placeholder in a node
    nothing reaches, or one feeding an unrelated input, satisfies this and still
    changes no output. Proving an input decides the OUTPUT is what the geometry
    proof does by walking from the save node backwards, and it is a different
    and much stronger claim.
    """

    placeholder = f"${{{name}}}"
    stack: list[Any] = [workflow]
    while stack:
        value = stack.pop()
        if isinstance(value, dict):
            stack.extend(value.values())
        elif isinstance(value, list):
            stack.extend(value)
        elif value == placeholder:
            return True
    return False

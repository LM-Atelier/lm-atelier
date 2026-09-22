"""Recognizing a workflow that can cut a subject out of a picture.

The studio's Isolate tool makes one promise: hand it a picture and get the
subject back with the background gone. That needs a workflow built to do it,
and whether one is installed has to be answerable before the tool is offered
rather than after someone waits for a result.

Every other tool class here is recognized from what a revision DECLARES in its
input schema, because that is the app's contract with a workflow and
`tool_capabilities` is handed schemas alone. Matting is the awkward one: the
others declare a setting the tool needs somewhere to put - a factor, a margin,
a mask - and a cutout has no setting at all. It has no strength, no factor and
no margin; it either separates the subject or it does not.

So what is declared here is not a setting but a statement: this workflow
returns the subject on a transparent background. The alternative was to search
the graph for a background-removal node, which is what the mask and upscale
contracts deliberately stopped doing, because a workflow that carries a node
is not the same as one that promises the result. A workflow that can matte
without saying so is exactly the case this design refuses to guess at.
"""

from __future__ import annotations

from typing import Any

#: The property a matting workflow declares, and the schema kind that marks it.
#: The property carries no value the caller sets: it is present to say what the
#: workflow returns, which is why it is read for its kind alone.
MATTING_SETTING_KEY = "matte"
MATTING_SCHEMA_KIND = "matting"


def workflow_declares_matting(input_schema: dict[str, Any] | None) -> bool:
    """Whether this workflow states that it returns a cut-out subject.

    The declaration is the whole answer. Nothing here reads the graph: a
    revision that separates a subject and does not say so is treated as a
    workflow that does not, so that the promise the tool makes and the promise
    the workflow made are the same promise.
    """

    if not isinstance(input_schema, dict):
        return False
    properties = input_schema.get("properties")
    if not isinstance(properties, dict):
        return False
    declared = properties.get(MATTING_SETTING_KEY)
    if not isinstance(declared, dict):
        return False
    return declared.get("x-lm-atelier-kind") == MATTING_SCHEMA_KIND

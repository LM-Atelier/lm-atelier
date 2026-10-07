"""Which setting values a record keeps when the prompt stays home."""

from __future__ import annotations

from local_lm import output_recipe


def test_a_workflow_text_choice_survives_an_omitted_prompt_and_free_text_does_not() -> None:
    """A value the workflow offers as a fixed choice is not anybody's words."""

    schema = {
        "type": "object",
        "properties": {
            "finish": {"type": "string", "enum": ["matte", "gloss"], "default": "matte"},
            "house_style": {"type": "string", "default": "art nouveau"},
        },
    }
    choices = output_recipe._setting_choices("text_to_image", schema)
    draft = output_recipe._Draft()

    kept = output_recipe._settings(
        {"finish": "gloss", "house_style": "art nouveau"},
        {"finish"},
        choices,
        False,
        draft,
    )

    assert choices["finish"] == frozenset({"matte", "gloss"})
    assert kept == {"bound": {"finish": "gloss"}, "unbound": {}}
    assert draft.removed == {"settings.house_style"}

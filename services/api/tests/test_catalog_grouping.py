"""One card per model, where the provider says a card is a version of one."""

from __future__ import annotations

from typing import Any

from local_lm.api import _grouped_by_parent, _with_installed_counts


def _card(version_id: str, parent: str | None, name: str) -> Any:
    from local_lm.schemas import CatalogModel

    return CatalogModel(
        provider="civitai",
        remote_id=version_id,
        name=name,
        parent_model_id=parent,
        parent_model_name="Lustify" if parent else None,
        compatibility="likely",
    )


def test_versions_of_one_model_become_one_card() -> None:
    """Twelve releases should not be twelve rows differing only in a suffix."""
    grouped = _grouped_by_parent(
        [
            _card("9002", "4201", "Lustify - v4.0"),
            _card("9001", "4201", "Lustify - v3.0"),
            _card("7001", "3300", "Other - v1"),
        ]
    )

    assert [card.remote_id for card in grouped] == ["9002", "7001"]
    assert grouped[0].name == "Lustify"
    assert grouped[0].version_count == 2
    # The card keeps a real version identity rather than becoming the model id.
    assert grouped[0].remote_id == "9002"


def test_a_card_with_no_parent_is_left_exactly_as_it_is() -> None:
    """Hugging Face repositories are already the installable thing."""
    plain = _card("owner/model", None, "owner/model")
    grouped = _grouped_by_parent([plain])

    assert grouped == [plain]
    assert grouped[0].version_count == 1


def test_grouping_keeps_the_order_the_provider_ranked() -> None:
    grouped = _grouped_by_parent(
        [
            _card("7001", "3300", "Other - v1"),
            _card("9002", "4201", "Lustify - v4.0"),
            _card("9001", "4201", "Lustify - v3.0"),
        ]
    )

    assert [card.remote_id for card in grouped] == ["7001", "9002"]


def test_counts_installed_versions_where_that_is_knowable() -> None:
    counted = _with_installed_counts(
        [_card("9002", "4201", "Lustify"), _card("7001", "3300", "Other")],
        {"4201": 2},
    )

    assert counted[0].installed_version_count == 2
    # A model with no recorded identity is unknown, not zero. The two look
    # alike and mean opposite things: "none installed" versus "this kind does
    # not record which version it is, so nothing can be matched".
    assert counted[1].installed_version_count is None


def test_a_card_with_no_parent_is_never_given_a_count() -> None:
    plain = _card("owner/model", None, "owner/model")

    assert _with_installed_counts([plain], {"4201": 2}) == [plain]

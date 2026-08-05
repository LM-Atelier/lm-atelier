"""A refused source dependency says which situation the package is in."""

from __future__ import annotations

import pytest

from local_lm.package_sources import (
    MAX_SOURCE_URL_CHARACTERS,
    classify_source_url,
    source_refusal,
)

PINNED = "git+https://github.com/owner/repo@" + "a" * 40


def test_an_exact_commit_is_read_as_one() -> None:
    source = classify_source_url(PINNED)

    assert source.repository == "owner/repo"
    assert source.commit == "a" * 40
    assert source.pinned is True


def test_a_git_suffix_is_not_part_of_the_repository_name() -> None:
    assert classify_source_url("git+https://github.com/owner/repo.git@main").repository == (
        "owner/repo"
    )


@pytest.mark.parametrize("revision", ["main", "v1.2.3", "a" * 39, "A" * 40 + "b"])
def test_anything_short_of_a_whole_object_name_is_not_a_pin(revision: str) -> None:
    """A prefix is not an object name: it can become ambiguous later."""
    source = classify_source_url(f"git+https://github.com/owner/repo@{revision}")

    assert source.pinned is False
    assert source.reference == revision


def test_no_revision_at_all_is_the_live_case() -> None:
    # Impact Pack declares one of these; WAS 3.0.1 declares three.
    source = classify_source_url("git+https://github.com/owner/repo")

    assert source.repository == "owner/repo"
    assert source.pinned is False
    assert source.reference is None


@pytest.mark.parametrize(
    "url",
    [
        "https://example.com/example.whl",
        "file:///tmp/example.whl",
        "git+https://gitlab.com/owner/repo@" + "a" * 40,
        "git+https://github.com/owner",
        "git+https://github.com/owner/repo/extra",
        "hg+https://github.com/owner/repo",
        "git+https://user@github.com.example.com/owner/repo",
        "",
    ],
)
def test_everything_outside_the_allowed_shape_names_no_repository(url: str) -> None:
    assert classify_source_url(url).repository is None


def test_an_absurdly_long_url_is_not_parsed_at_all() -> None:
    url = "git+https://github.com/owner/" + "r" * MAX_SOURCE_URL_CHARACTERS

    assert classify_source_url(url).repository is None


def test_the_unpinned_refusal_names_the_repository_and_what_would_fix_it() -> None:
    code, message = source_refusal(classify_source_url("git+https://github.com/owner/repo@main"))

    assert code == "unpinned_source_dependency"
    assert "owner/repo" in message
    assert "main" in message
    assert "pin" in message


def test_a_pinned_source_is_refused_as_unresolved_rather_than_as_forbidden() -> None:
    """Fail-closed, but distinguishable: this one a later slice can resolve."""
    code, message = source_refusal(classify_source_url(PINNED))

    assert code == "unresolved_source_dependency"
    assert "owner/repo" in message
    assert "a" * 12 in message


def test_a_url_that_was_never_allowed_keeps_the_original_refusal() -> None:
    code, _ = source_refusal(classify_source_url("file:///tmp/example.whl"))

    assert code == "direct_dependency_url"

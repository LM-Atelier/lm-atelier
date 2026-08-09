"""A refused source dependency says which situation the package is in."""

from __future__ import annotations

import pytest

from local_lm.package_sources import (
    MAX_SOURCE_URL_CHARACTERS,
    classify_source_url,
    partition_unpinned_sources,
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


def test_nothing_is_set_aside_without_an_authorized_workflow() -> None:
    """With no workflow to prove it against, the ordinary refusal stands."""
    declarations = ("example @ git+https://github.com/owner/repo", "requests>=2")

    installable, omitted = partition_unpinned_sources(declarations, authorized=False)

    assert installable == declarations
    assert omitted == ()


def test_only_the_unpinned_allowed_host_declaration_is_set_aside() -> None:
    declarations = (
        "example @ git+https://github.com/owner/repo",
        "pinned @ git+https://github.com/owner/other@" + "a" * 40,
        "elsewhere @ https://example.com/thing.whl",
        "requests>=2",
    )

    installable, omitted = partition_unpinned_sources(declarations, authorized=True)

    # A pinned source names an immutable object; its road is resolution, not
    # omission, and setting it aside would skip something installable exactly.
    assert omitted == ("example @ git+https://github.com/owner/repo",)
    assert installable == (
        "pinned @ git+https://github.com/owner/other@" + "a" * 40,
        "elsewhere @ https://example.com/thing.whl",
        "requests>=2",
    )


def test_a_source_written_bare_is_set_aside_like_any_other() -> None:
    """A line that is only a URL points where it says, so it is read that way.

    This used to be left in the installable set, on the grounds that setting
    aside a line no parser accepts would be a guess about what it meant. It is
    not a guess: pip takes this spelling, real packages ship it - four of them
    across the two this product must install - and the URL is the whole line.

    What the old behaviour actually did was decide by spelling. The same
    dependency written `name @ url` was set aside and the package prepared;
    written bare it refused the package outright. The question worth asking is
    whether the source names an exact commit, and that answer is the same
    either way.

    The guarantee is unchanged and is asserted below: an unpinned source is
    never installed. It is omitted, recorded, and provable.
    """
    installable, omitted = partition_unpinned_sources(
        ("git+https://github.com/owner/repo", "git+https://github.com/owner/repo.git"),
        authorized=True,
    )

    assert installable == ()
    assert omitted == (
        "git+https://github.com/owner/repo",
        "git+https://github.com/owner/repo.git",
    )


def test_a_bare_pinned_source_is_still_never_set_aside() -> None:
    """Omission is for what cannot be installed exactly, not for convenience."""
    pinned = "git+https://github.com/owner/repo@" + "a" * 40

    installable, omitted = partition_unpinned_sources((pinned,), authorized=True)

    assert omitted == ()
    assert installable == (pinned,)


def test_a_line_no_parser_accepts_is_still_not_set_aside() -> None:
    """Reading a URL is not the same as reading anything at all.

    Nothing here points anywhere, so there is nothing to set aside and the
    package is still refused for declaring something unreadable.
    """
    declarations = ("not a requirement at all!!", "two urls https://a.example https://b.example")

    installable, omitted = partition_unpinned_sources(declarations, authorized=True)

    assert omitted == ()
    assert installable == declarations


def test_comments_and_installer_options_are_never_set_aside() -> None:
    declarations = (
        "# git+https://github.com/owner/repo",
        "--index-url https://example.com/simple",
        "",
        "example @ git+https://github.com/owner/repo  # the live case",
    )

    installable, omitted = partition_unpinned_sources(declarations, authorized=True)

    assert omitted == ("example @ git+https://github.com/owner/repo  # the live case",)
    assert len(installable) == 3

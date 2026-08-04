"""Retrieval refuses far more than it accepts, and that is the design."""

from __future__ import annotations

import pytest

from local_lm.web_access import (
    may_fetch_urls,
    may_search,
    must_confirm_each_query,
)
from local_lm.web_retrieval import (
    MAX_TEXT_CHARACTERS,
    WebRetrievalError,
    extract_source,
    validate_target,
)


def _resolves_to(address: str):
    return lambda host, port: [(2, 1, 6, "", (address, 0))]


PUBLIC = _resolves_to("93.184.216.34")


def test_only_a_public_https_page_is_reachable() -> None:
    assert validate_target("https://example.test/article", resolve=PUBLIC)


@pytest.mark.parametrize(
    ("url", "code"),
    [
        ("http://example.test/a", "web-scheme-refused"),
        ("file:///etc/passwd", "web-scheme-refused"),
        ("https://user:secret@example.test/a", "web-credentials-refused"),
        ("https://" + "x" * 3000, "web-url-too-long"),
        ("https:///nohost", "web-url-invalid"),
    ],
)
def test_addresses_that_are_not_ours_to_fetch_are_refused(url: str, code: str) -> None:
    with pytest.raises(WebRetrievalError) as caught:
        validate_target(url, resolve=PUBLIC)
    assert caught.value.code == code


@pytest.mark.parametrize(
    "address",
    ["127.0.0.1", "10.0.0.5", "192.168.1.10", "169.254.169.254", "::1", "fd00::1"],
)
def test_a_host_pointing_inside_this_machine_or_network_is_refused(address: str) -> None:
    # A name is a claim and an address is a fact. This machine runs services
    # on localhost, and the cloud metadata endpoint is a public hostname
    # pointing somewhere very private - neither is ours to hand to a page.
    with pytest.raises(WebRetrievalError) as caught:
        validate_target("https://looks-fine.test/a", resolve=_resolves_to(address))
    assert caught.value.code == "web-private-address-refused"


def test_a_literal_private_address_is_refused_without_resolving_anything() -> None:
    with pytest.raises(WebRetrievalError) as caught:
        validate_target("https://127.0.0.1/admin", resolve=PUBLIC)
    assert caught.value.code == "web-private-address-refused"


def test_a_page_becomes_text_and_nothing_else() -> None:
    html = (
        b"<html><head><title>A title</title>"
        b"<script>fetch('/steal')</script><style>p{color:red}</style></head>"
        b"<body><p>First sentence.</p><p>Second sentence.</p></body></html>"
    )
    source = extract_source("https://a.test/x", "https://a.test/x", "text/html", html)

    assert source.title == "A title"
    assert "First sentence." in source.text and "Second sentence." in source.text
    # Scripts and styles are discarded rather than parsed. Nothing that could
    # act, and no structure a later stage could mistake for instructions.
    assert "fetch" not in source.text and "color:red" not in source.text


def test_something_that_is_not_a_page_is_refused_rather_than_guessed_at() -> None:
    with pytest.raises(WebRetrievalError) as caught:
        extract_source("https://a.test/x", "https://a.test/x", "application/pdf", b"%PDF-1.4")
    assert caught.value.code == "web-content-unreadable"


def test_an_enormous_page_is_cut_and_says_so() -> None:
    body = b"<html><body><p>" + b"word " * 200_000 + b"</p></body></html>"
    source = extract_source("https://a.test/x", "https://a.test/x", "text/html", body)

    assert len(source.text) <= MAX_TEXT_CHARACTERS
    # Truncation is reported rather than hidden, so an answer built on half a
    # page can say that is what it had.
    assert source.truncated is True


def test_a_page_with_nothing_to_read_is_refused() -> None:
    with pytest.raises(WebRetrievalError) as caught:
        extract_source("https://a.test/x", "https://a.test/x", "text/html", b"<html></html>")
    assert caught.value.code == "web-content-empty"


class TestBothGatesMustBeOpen:
    def test_the_installation_can_refuse_for_everyone(self) -> None:
        allowed = {"allow_url_fetch": True, "allow_search": True}
        assert may_fetch_urls(installation_enabled=False, chat_settings=allowed) is False
        assert may_search(installation_enabled=False, chat_settings=allowed) is False

    def test_a_chat_starts_closed_however_the_installation_is_set(self) -> None:
        # Permission is never inherited: a new chat has no settings at all,
        # and no settings means no.
        assert may_fetch_urls(installation_enabled=True, chat_settings=None) is False
        assert may_fetch_urls(installation_enabled=True, chat_settings={}) is False

    def test_fetching_and_searching_are_separate_permissions(self) -> None:
        fetch_only = {"allow_url_fetch": True}
        assert may_fetch_urls(installation_enabled=True, chat_settings=fetch_only) is True
        # Retrieving an address someone pasted and sending their words to a
        # search provider are different acts, so allowing one says nothing
        # about the other.
        assert may_search(installation_enabled=True, chat_settings=fetch_only) is False

    def test_a_malformed_permission_is_no_permission(self) -> None:
        for value in ("true", 1, [], {"nested": True}):
            assert (
                may_fetch_urls(installation_enabled=True, chat_settings={"allow_url_fetch": value})
                is False
            )

    def test_queries_are_confirmed_until_someone_says_otherwise(self) -> None:
        assert must_confirm_each_query(None) is True
        assert must_confirm_each_query({"allow_search": True}) is True
        assert must_confirm_each_query({"allow_search_without_asking": True}) is False

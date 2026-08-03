"""Explicit source links: parsed, validated, and never trusted as given."""

from __future__ import annotations

from typing import Any

from local_lm.workflow_source_candidates import (
    catalog_host_map,
    collect_source_candidates,
    parse_source_url,
)

HOSTS = {"civitai.com": "civitai", "huggingface.co": "huggingface"}


def _note(text: str, node_id: int = 1) -> dict[str, Any]:
    return {"id": node_id, "type": "Note", "widgets_values": [text]}


def test_hosts_come_from_what_this_installation_registered() -> None:
    hosts = catalog_host_map(
        [("huggingface", "https://huggingface.co"), ("civitai", "https://example.test")]
    )
    # A deployment serving a different host resolves its own links; the
    # parser knows nothing about which hosts exist.
    assert hosts == {"huggingface.co": "huggingface", "example.test": "civitai"}
    assert parse_source_url("https://example.test/models/42?modelVersionId=7", allowed_hosts=hosts)


def test_an_unregistered_host_is_refused_outright() -> None:
    assert (
        parse_source_url("https://evil.test/models/1?modelVersionId=2", allowed_hosts=HOSTS) is None
    )


def test_only_https_without_credentials_parses() -> None:
    for url in (
        "http://civitai.com/models/1?modelVersionId=2",
        "https://user:pass@civitai.com/models/1?modelVersionId=2",
        "https://civitai.com/" + "x" * 3000,
    ):
        assert parse_source_url(url, allowed_hosts=HOSTS) is None


def test_a_model_link_without_a_version_is_not_installable() -> None:
    # A model page names a thing; a version names something installable.
    assert parse_source_url("https://civitai.com/models/1662740", allowed_hosts=HOSTS) is None
    candidate = parse_source_url(
        "https://civitai.com/models/1662740/lenovo?modelVersionId=3075606", allowed_hosts=HOSTS
    )
    assert candidate is not None
    assert candidate.provider == "civitai"
    assert candidate.remote_id == "3075606"
    assert candidate.revision == "3075606"


def test_a_repository_file_link_keeps_its_path() -> None:
    candidate = parse_source_url(
        "https://huggingface.co/Comfy-Org/Qwen/resolve/main/split_files/vae/qwen_image_vae.safetensors",
        allowed_hosts=HOSTS,
    )
    assert candidate is not None
    assert candidate.remote_id == "Comfy-Org/Qwen"
    assert candidate.revision == "main"
    assert candidate.filename == "split_files/vae/qwen_image_vae.safetensors"

    bare = parse_source_url("https://huggingface.co/Comfy-Org/Qwen", allowed_hosts=HOSTS)
    assert bare is not None
    assert bare.filename is None


def test_a_traversing_path_is_refused() -> None:
    assert (
        parse_source_url(
            "https://huggingface.co/owner/model/resolve/main/../../etc/passwd",
            allowed_hosts=HOSTS,
        )
        is None
    )


def test_candidates_attach_to_the_file_their_text_names() -> None:
    workflow = {
        "nodes": [
            _note(
                "VAE: qwen_image_vae.safetensors from "
                "https://huggingface.co/Comfy-Org/Qwen/resolve/main/vae/qwen_image_vae.safetensors",
                1,
            ),
            _note(
                "Detail slider https://civitai.com/models/2729908/detail?modelVersionId=3068874",
                2,
            ),
        ]
    }

    found = collect_source_candidates(
        workflow,
        allowed_hosts=HOSTS,
        asset_filenames=["qwen_image_vae.safetensors", "Detailer-KREA2.safetensors"],
    )

    assert found["qwen_image_vae.safetensors"][0].remote_id == "Comfy-Org/Qwen"
    # The second note names no known file, so its link is a general
    # suggestion rather than an assumed answer for some asset.
    assert found[""][0].remote_id == "3068874"


def test_an_ambiguous_note_never_guesses_which_file_it_means() -> None:
    workflow = {
        "nodes": [
            _note(
                "Both first.safetensors and second.safetensors come from "
                "https://civitai.com/models/1/pack?modelVersionId=2"
            )
        ]
    }

    found = collect_source_candidates(
        workflow,
        allowed_hosts=HOSTS,
        asset_filenames=["first.safetensors", "second.safetensors"],
    )

    assert "first.safetensors" not in found
    assert found[""][0].remote_id == "2"


def test_a_stem_mention_still_matches_its_file() -> None:
    workflow = {"nodes": [_note("Detailer-KREA2 https://civitai.com/models/9/d?modelVersionId=11")]}
    found = collect_source_candidates(
        workflow, allowed_hosts=HOSTS, asset_filenames=["Detailer-KREA2.safetensors"]
    )
    assert found["Detailer-KREA2.safetensors"][0].remote_id == "11"


def test_duplicates_collapse_and_the_scan_is_bounded() -> None:
    link = "https://civitai.com/models/1/a?modelVersionId=2"
    workflow = {"nodes": [_note(f"{link} {link} {link}", 1), _note("x" * 200_000, 2)]}
    found = collect_source_candidates(workflow, allowed_hosts=HOSTS)
    assert len(found[""]) == 1


def test_a_graph_without_notes_yields_nothing() -> None:
    assert collect_source_candidates({"nodes": []}, allowed_hosts=HOSTS) == {}
    assert collect_source_candidates({}, allowed_hosts=HOSTS) == {}


def test_display_named_notes_still_yield_usable_suggestions() -> None:
    """The realistic shape: authors name models, not files.

    A note reads "Lenovo UltraReal <link>" while the file is
    lenovo_krea2.safetensors. Nothing there licenses a binding - but the
    link is exactly what the user needs, so it is offered for them to
    assign rather than discarded or guessed at.
    """

    workflow = {
        "nodes": [
            _note(
                "## LoRAs\n\nLenovo UltraReal\nhttps://civitai.com/models/1662740/l?modelVersionId=3075606",
                1,
            ),
            _note(
                "NiceGirls UltraReal\nhttps://civitai.com/models/1862761/n?modelVersionId=3075498",
                2,
            ),
        ]
    }

    found = collect_source_candidates(
        workflow,
        allowed_hosts=HOSTS,
        asset_filenames=["lenovo_krea2.safetensors", "nicegirls_krea2.safetensors"],
    )

    # No filename is named, so nothing is bound...
    assert [key for key in found if key] == []
    # ...but both recorded sources survive for the user to assign.
    assert {candidate.remote_id for candidate in found[""]} == {"3075606", "3075498"}


def test_one_named_file_does_not_swallow_a_neighbors_link() -> None:
    """A note listing several models keeps each link with its own subject."""

    workflow = {
        "nodes": [
            _note(
                "Lenovo UltraReal\nhttps://civitai.com/models/1/l?modelVersionId=2\n\n"
                "VAE qwen_image_vae.safetensors "
                "https://huggingface.co/Comfy-Org/Qwen/resolve/main/qwen_image_vae.safetensors"
            )
        ]
    }

    found = collect_source_candidates(
        workflow,
        allowed_hosts=HOSTS,
        asset_filenames=["lenovo_krea2.safetensors", "qwen_image_vae.safetensors"],
    )

    assert [c.remote_id for c in found["qwen_image_vae.safetensors"]] == ["Comfy-Org/Qwen"]
    assert [c.remote_id for c in found[""]] == ["2"]

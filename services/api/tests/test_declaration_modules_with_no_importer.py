"""Which `_v1` declaration modules nothing in the application imports.

The `_v1` suffix marks a declaration or contract module. It promises that the
shape it describes is versioned and that a matching `test_*_v1.py` covers it. It
does NOT promise that anything runs it. Thirty-one of these modules are imported
by nothing in the package, in families whose consumers never landed, and each of
them has a passing test module beside it, which is exactly the evidence a reader
uses to conclude a feature works.

Writing the list down is the whole point. Without it the number can only be
re-measured, and re-measuring is how it goes wrong: an earlier count of these
said thirty-two, because the scan excluded any file whose name ended with the
module's name in order to skip the module itself, and that also excluded
`conversation_search_http_message_anchor_v1.py`, a real importer of
`message_anchor_v1`. With the list here, a module that gains an importer has to
be struck from it by hand, which reads in a diff as the deliberate act it is,
and a module that joins it cannot arrive quietly.

This is not a prohibition. Adding a declaration module ahead of its consumer is
a reasonable thing to do; landing one and forgetting is not, and the difference
between those two is whether somebody edited this list on purpose.
"""

from __future__ import annotations

import ast
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1] / "local_lm"

#: Declaration modules no module in the package imports, as of 2026-09-22.
#: Strike a name from this list in the same change that gives it an importer.
WITHOUT_AN_IMPORTER = frozenset(
    {
        "chat_item_removal_declaration_v1",
        "chat_item_removal_identity_preserve_v1",
        "chat_item_removal_impact_v1",
        "chat_item_removal_payload_detach_v1",
        "chat_item_removal_replay_refuse_v1",
        "chat_item_removal_reply_preserve_v1",
        "chat_item_removal_revision_block_v1",
        "conversation_search_compose_v1",
        "conversation_search_http_message_anchor_v1",
        "conversation_search_http_message_window_v1",
        "conversation_search_http_snippet_policy_v1",
        "conversation_search_mutation_v1",
        "prior_turn_edit_declaration_v1",
        "prior_turn_edit_queue_order_v1",
        "prior_turn_edit_snapshot_v1",
        "prior_turn_edit_source_preserve_v1",
        "search_chat_detail_hold_v1",
        "search_document_v1",
        "search_filters_v1",
        "search_fts_projection_v1",
        "search_index_status_v1",
        "search_rebuild_v1",
        "shared_asset_collection_v1",
        "shared_asset_consumer_v1",
        "shared_asset_leases_v1",
        "shared_asset_restore_v1",
        "shared_asset_root_v1",
        "shared_asset_store_v1",
        "shared_asset_verify_v1",
        "shared_asset_view_v1",
        "shared_asset_workflow_bundle_v1",
    }
)


def _declaration_modules() -> set[str]:
    """Every `_v1` module in the package, migrations aside."""

    return {
        path.stem
        for path in PACKAGE.rglob("*_v1.py")
        if "migrations" not in path.relative_to(PACKAGE).parts
    }


def _modules_imported_by(source: Path) -> set[str]:
    """The module names this file imports, as modules rather than as symbols.

    `from .thing import name` imports the module `thing`; the `name` beside it
    is a symbol and is deliberately not counted, because a symbol that happens
    to share a module's name would otherwise make that module look imported.
    """

    found: set[str] = set()
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.module:
                found.add(node.module.rsplit(".", 1)[-1])
            elif node.level:
                # `from . import thing`: the names ARE modules here.
                found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.Import):
            found.update(alias.name.rsplit(".", 1)[-1] for alias in node.names)
    return found


def _imported_within_the_package() -> set[str]:
    imported: set[str] = set()
    for source in PACKAGE.rglob("*.py"):
        imported |= _modules_imported_by(source) - {source.stem}
    return imported


def test_no_declaration_module_joins_or_leaves_the_list_unnoticed() -> None:
    declarations = _declaration_modules()
    # A broken enumeration would make every assertion below vacuous, so the
    # scan has to have found the modules before anything is concluded from it.
    assert len(declarations) > len(WITHOUT_AN_IMPORTER)
    assert declarations >= WITHOUT_AN_IMPORTER

    imported = _imported_within_the_package()
    unimported = {name for name in declarations if name not in imported}

    arrived = sorted(unimported - WITHOUT_AN_IMPORTER)
    assert not arrived, (
        "declaration module(s) landed with nothing importing them; add them here "
        f"deliberately or wire them up: {arrived}"
    )
    wired = sorted(WITHOUT_AN_IMPORTER - unimported)
    assert not wired, (
        "declaration module(s) now have an importer; strike them from this list "
        f"in the change that wired them up: {wired}"
    )


def test_the_import_scan_sees_an_importer_that_shares_a_name_suffix() -> None:
    """The earlier hand count got this case wrong, so it is a test now.

    `conversation_search_http_message_anchor_v1` imports `message_anchor_v1`,
    and its own filename ends with that module's name. A scan that skips the
    module under test by matching the end of a filename skips this importer
    too, and reports `message_anchor_v1` as unimported when it is not.
    """

    assert "message_anchor_v1" in _declaration_modules()
    assert "message_anchor_v1" not in WITHOUT_AN_IMPORTER
    assert "message_anchor_v1" in _imported_within_the_package()

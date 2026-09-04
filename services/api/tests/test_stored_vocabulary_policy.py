"""The stored-vocabulary policy is documented, and the documentation is true.

A policy written only as prose goes stale silently: someone constrains a
vocabulary column, the docstring in models.py keeps saying the opposite, and the
next reader believes the wrong thing. So this binds the documentation to the
schema it describes.

WHY NOTHING HERE IDENTIFIES A COLUMN BY ITS NAME. A "vocabulary column" has no
syntactic marker. The obvious classifier is a suffix list - status, kind, state,
mode, role, type - and it does not work, because closed vocabularies in this
schema are also called ``source``, ``operation``, ``authority_rule`` and
``satisfaction``. Each of those holds one of a fixed set of strings defined by a
Python enum, and none would be found by any reasonable list of name endings.
Widening such a list moves the population it reports without ever making it
complete, so the totals it produces describe the classifier rather than the
schema.

WHY NOTHING HERE MATCHES THE SHAPE OF THE SQL EITHER. The next classifier to
reach for reads each constraint's own SQL and looks for string literals - an
``IN`` list, or a comparison against a quoted value. That is closer, but it is
still a pattern over text, and text has spellings: ``x IN ('a')`` and
``x IN('a')`` are the same constraint to the database and different strings to a
regular expression. A pattern that misses one spelling produces a set that
quietly omits a real constraint, and pinning that set does not help, because the
omitted constraint never enters it to be noticed.

So both assertions below quantify over everything and match no pattern at all:

    no mapped column anywhere uses SQLAlchemy's ``Enum`` type, which is
    checkable exactly because the answer is zero;

    the columns named by a ``CheckConstraint`` are exactly the set enumerated
    below, found by looking for each table's real column names inside its own
    constraints - so a constraint added in any spelling is found.

WHAT THIS DELIBERATELY DOES NOT CLAIM. It does not pin the CONTENTS of an
existing constraint. Adding a value to a list that is already there leaves the
same column constrained by the same constraint and does not fail anything here.
That is in keeping with what the policy is about: whether the database is the
authority on stored vocabularies, not which values a constraint that already
exists happens to allow.

The tests fail in both directions. Remove the policy note and one fails. Add or
drop a constraint anywhere in the schema and another fails, naming the column.
"""

from __future__ import annotations

import re

from sqlalchemy import Enum as SQLEnum
from sqlalchemy import String, Table
from sqlalchemy.schema import CheckConstraint

from local_lm.db import Base

#: Every mapped column named by a ``CheckConstraint``, whatever the constraint
#: is for. Most are format and range checks - digest lengths, positive counters,
#: bounded text - and a minority pin a column to a set of string literals. Both
#: kinds are here because separating them needs a judgement about intent, and a
#: judgement call should not sit between a schema change and a failing test.
#:
#: Generated from the schema rather than typed by hand.
CONSTRAINED_COLUMNS: frozenset[tuple[str, str]] = frozenset(
    {
        ("adapter_prompt_grammars", "asset_sha256"),
        ("adapter_prompt_grammars", "source_sha256"),
        ("artifact_library_entries", "deleted_at"),
        ("artifact_library_entries", "display_name"),
        ("artifact_library_entries", "favorite"),
        ("artifact_library_entries", "recovery_id"),
        ("artifact_library_entries", "state"),
        ("artifact_library_entries", "version"),
        ("chat_item_removal_receipts", "message_revision_id"),
        ("chat_item_removal_receipts", "operation_key"),
        ("chat_item_removal_receipts", "request_sha256"),
        ("chat_workflow_selections", "mode"),
        ("chat_workflow_selections", "selector_capability"),
        ("chat_workflow_selections", "workflow_family_id"),
        ("chats", "routing_mode"),
        ("comfy_registry_source_artifact_reviews", "artifact_sha256"),
        ("comfy_registry_source_artifact_reviews", "artifact_size_bytes"),
        ("comfy_registry_source_artifact_reviews", "review_sha256"),
        ("comfy_registry_source_artifact_reviews", "reviewer_kind"),
        ("comfy_registry_source_artifact_reviews", "source_commit"),
        ("comfy_registry_source_artifact_reviews", "source_declaration_sha256"),
        ("edit_templates", "mask_mode"),
        ("media_collection_memberships", "note"),
        ("media_collection_memberships", "position"),
        ("media_collections", "description"),
        ("media_collections", "kind"),
        ("media_collections", "name"),
        ("media_collections", "version"),
        ("media_tags", "color"),
        ("media_tags", "label"),
        ("media_tags", "slug"),
        ("media_tags", "version"),
        ("message_references", "position"),
        ("message_references", "reference_subject_id"),
        ("project_workflow_selections", "mode"),
        ("project_workflow_selections", "selector_capability"),
        ("project_workflow_selections", "workflow_family_id"),
        ("project_workflow_selections", "workflow_revision_id"),
        ("prompt_expansion_batches", "codec_version"),
        ("prompt_expansion_batches", "contract_sha256"),
        ("prompt_expansion_batches", "idempotency_key"),
        ("prompt_expansion_batches", "original_plan_sha256"),
        ("prompt_expansion_batches", "plan_sha256"),
        ("prompt_expansion_batches", "plan_version"),
        ("prompt_expansion_batches", "queue_idempotency_key"),
        ("prompt_expansion_batches", "schema_version"),
        ("prompt_expansion_batches", "state"),
        ("prompt_expansion_items", "media_seed"),
        ("prompt_expansion_items", "ordinal"),
        ("prompt_expansion_items", "original_rendered_prompt"),
        ("prompt_expansion_items", "original_rendered_sha256"),
        ("prompt_expansion_items", "reroll_count"),
        ("prompt_expansion_items", "review_version"),
        ("prompt_expansion_items", "reviewed_prompt"),
        ("prompt_expansion_items", "reviewed_sha256"),
        ("prompt_template_definitions", "current_revision_id"),
        ("prompt_template_definitions", "description"),
        ("prompt_template_definitions", "name"),
        ("prompt_template_import_winners", "authority_rule"),
        ("prompt_template_import_winners", "bundle_sha256"),
        ("prompt_template_import_winners", "contract_sha256"),
        ("prompt_template_import_winners", "idempotency_key"),
        ("prompt_template_import_winners", "prompt_template_id"),
        ("prompt_template_import_winners", "prompt_template_revision_id"),
        ("prompt_template_import_winners", "request_sha256"),
        ("prompt_template_revisions", "contract_sha256"),
        ("prompt_template_revisions", "schema_version"),
        ("prompt_template_revisions", "version"),
        ("reference_asset_review_events", "artifact_sha256"),
        ("reference_asset_review_events", "decision_sha256"),
        ("reference_asset_review_events", "expected_version"),
        ("reference_asset_review_events", "height"),
        ("reference_asset_review_events", "result_version"),
        ("reference_asset_review_events", "width"),
        ("reference_assets", "review_version"),
        ("reference_subjects", "mention_slug"),
        ("reference_subjects", "name"),
        ("workflow_activations", "binding_sha256"),
        ("workflow_activations", "dependency_contract_sha256"),
        ("workflow_activations", "invalidated_at"),
        ("workflow_activations", "is_active"),
        ("workflow_activations", "state"),
        ("workflow_definitions", "family_id"),
        ("workflow_definitions", "variant_key"),
        ("workflow_dependency_bindings", "comfy_registry_install_id"),
        ("workflow_dependency_bindings", "custom_node_install_id"),
        ("workflow_dependency_bindings", "model_asset_install_id"),
        ("workflow_dependency_bindings", "model_install_id"),
        ("workflow_dependency_bindings", "model_profile_id"),
        ("workflow_dependency_bindings", "requirement_key"),
        ("workflow_dependency_bindings", "resource_identity_sha256"),
        ("workflow_dependency_bindings", "runtime_key"),
        ("workflow_dependency_slots", "contract_sha256"),
        ("workflow_dependency_slots", "name"),
        ("workflow_dependency_slots", "ordinal"),
        ("workflow_dependency_slots", "required"),
        ("workflow_dependency_slots", "resource_kind"),
        ("workflow_dependency_slots", "satisfaction"),
        ("workflow_install_offers", "binding_plan_sha256"),
        ("workflow_install_offers", "dependency_contract_sha256"),
        ("workflow_install_offers", "offer_sha256"),
        ("workflow_install_offers", "plan_count"),
        ("workflow_install_offers", "status"),
        ("workflow_install_offers", "total_bytes"),
        ("workflow_install_offers", "workflow_artifact_sha256"),
        ("workflow_preferences", "enabled"),
        ("workflow_preferences", "is_default"),
        ("workflow_preferences", "selector_capability"),
        ("workflow_profile_compatibility", "source_fingerprint_sha256"),
        ("workflow_revisions", "dependency_contract_sha256"),
        ("workflow_trust_attestations", "artifact_sha256"),
        ("workflow_trust_attestations", "launch_scope_sha256"),
        ("workflow_trust_attestations", "node_inventory_sha256"),
        ("workflow_trust_attestations", "runtime_contract_sha256"),
        ("workflow_trust_attestations", "whitelist_sha256"),
    }
)

#: The number of ``CheckConstraint`` objects in the schema. Pinned alongside the
#: columns so that a constraint naming no column at all, or one constraint
#: replaced by two over the same columns, still fails.
CHECK_CONSTRAINT_COUNT = 103


def _mapped_tables() -> dict[str, Table]:
    """Every mapped table, deduplicated across mappers that share one."""

    tables: dict[str, Table] = {}
    for mapper in Base.registry.mappers:
        table = mapper.local_table
        if isinstance(table, Table):
            tables[table.name] = table
    return tables


def _check_constraints() -> list[tuple[str, str]]:
    """Table name and normalised SQL for every CheckConstraint in the schema."""

    found: list[tuple[str, str]] = []
    for name, table in _mapped_tables().items():
        for constraint in table.constraints:
            if isinstance(constraint, CheckConstraint):
                found.append((name, " ".join(str(constraint.sqltext).split())))
    return found


def _constrained_columns() -> set[tuple[str, str]]:
    """Columns named by a CheckConstraint, found by column name inside the SQL.

    This is the direction that matters. Reading the constraint's SQL and asking
    which of the table's real columns appear in it depends on no pattern over
    how the constraint is written, so a constraint is found whether it is
    spelled ``IN ('a')`` or ``IN('a')``, and whatever its column is called.
    """

    constrained: set[tuple[str, str]] = set()
    tables = _mapped_tables()
    for table_name, text in _check_constraints():
        for column in tables[table_name].columns:
            if re.search(rf"\b{re.escape(column.name)}\b", text):
                constrained.add((table_name, column.name))
    return constrained


def _all_columns() -> set[tuple[str, str]]:
    return {
        (name, column.name) for name, table in _mapped_tables().items() for column in table.columns
    }


def test_the_stored_vocabulary_policy_is_documented() -> None:
    """models.py must carry the policy, because nothing else states it.

    The policy is invisible from any single model - it is a fact about how many
    of them lack a constraint - so the module docstring is the only place a
    reader can learn it.
    """
    from local_lm import models

    doc = models.__doc__
    assert doc is not None, "models.py has no module docstring"
    assert "STORED VOCABULARIES ARE ENFORCED BY THE APPLICATION" in doc
    assert "if you are reading one" in doc.lower()
    # It must not claim uniformity the schema does not have.
    assert "MIXED" in doc


def test_the_constrained_columns_are_exactly_these() -> None:
    """Every column a CheckConstraint names, pinned.

    Failing here does not mean someone did something wrong. It means the schema
    no longer matches what models.py says about it, so one of the two has to
    change deliberately.
    """
    actual = _constrained_columns()

    added = actual - CONSTRAINED_COLUMNS
    removed = CONSTRAINED_COLUMNS - actual
    assert not added, (
        f"these columns gained a database constraint: {sorted(added)}. models.py "
        f"documents application-level enforcement as the intended direction for "
        f"stored vocabularies; check which of the two should change."
    )
    assert not removed, (
        f"these columns lost their constraint: {sorted(removed)}. The policy note "
        f"in models.py describes the constrained set and is now wrong."
    )


def test_the_schema_holds_exactly_the_documented_number_of_check_constraints() -> None:
    """Pinned so a constraint that names no column is still visible."""

    constraints = _check_constraints()
    assert len(constraints) == CHECK_CONSTRAINT_COUNT, (
        f"the schema holds {len(constraints)} CheckConstraints against the "
        f"{CHECK_CONSTRAINT_COUNT} recorded here"
    )


def test_no_mapped_column_anywhere_uses_an_enum_column_type() -> None:
    """An Enum() column type is a database-level constraint by another name.

    Universal over the schema, with no name filter. Counting only
    CheckConstraint would let the policy be broken through a different mechanism
    and still pass, and asking the question only of columns whose names look
    like vocabularies would let it be broken on any column whose name does not.
    """
    using_enum = sorted(
        f"{name}.{column.name}"
        for name, table in _mapped_tables().items()
        for column in table.columns
        if isinstance(column.type, SQLEnum)
    )
    assert not using_enum, f"columns using Enum(): {using_enum}"


def test_most_mapped_columns_carry_no_constraint_at_all() -> None:
    """The claim the policy actually rests on, stated over the whole schema."""

    total = len(_all_columns())
    constrained = len(_constrained_columns())
    assert total - constrained > constrained * 3, (
        f"the policy says the database is not the authority on stored "
        f"vocabularies, but {constrained} of {total} mapped columns are now "
        f"named by a constraint"
    )


def test_a_closed_vocabulary_can_carry_no_constraint_at_all() -> None:
    """Two closed vocabularies that no name-based classifier would find.

    ``message_references.source`` is a String written from an enum and read back
    through it, and ``runs.operation`` is a String defaulted from one. Both are
    closed sets of strings, both carry no database constraint, and the policy
    says neither needs one. They are asserted as what they are so that the
    documented behaviour holds in the two places it is least obvious.
    """
    tables = _mapped_tables()
    constrained = _constrained_columns()

    for table_name, column_name in (
        ("message_references", "source"),
        ("runs", "operation"),
    ):
        assert table_name in tables, f"{table_name} is no longer mapped"
        column = tables[table_name].columns[column_name]
        assert isinstance(column.type, String), (
            f"{table_name}.{column_name} is no longer a String column"
        )
        assert (table_name, column_name) not in constrained, (
            f"{table_name}.{column_name} gained a database constraint; the "
            f"policy in models.py says application-level enforcement is the "
            f"intended direction for stored vocabularies"
        )


def test_the_instrument_still_sees_the_schema() -> None:
    """Guard the walk itself.

    Every test above quantifies over _mapped_tables(). If that returns nothing -
    an import order change, a registry refactor - they would all pass over an
    empty set and report health they never measured.
    """
    tables = _mapped_tables()
    columns = _all_columns()
    assert len(tables) >= 40, f"only {len(tables)} mapped tables found"
    assert len(columns) >= 400, f"only {len(columns)} mapped columns found"
    assert ("runs", "operation") in columns

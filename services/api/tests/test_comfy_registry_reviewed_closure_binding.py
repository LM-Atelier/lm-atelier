from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from dataclasses import replace
from typing import TYPE_CHECKING, Any

import pytest
from sqlalchemy import delete
from sqlalchemy.orm import Session
from test_comfy_registry_closure_driver import _Sources
from test_comfy_registry_reviewed_closure import _drive, _review
from test_comfy_registry_reviewed_wheel_metadata import _HEADERS, _review_metadata
from test_comfy_registry_reviewed_wheel_staging import (
    source_review_context as source_review_context,
)
from test_comfy_registry_source_artifacts import DECLARATION
from test_comfy_registry_wheel_artifacts import _environment

from local_lm.artifacts import ArtifactStore
from local_lm.comfy_registry_reviewed_inputs import ComfyRegistryReviewedInputContext
from local_lm.comfy_registry_wheel_inputs_v1 import ComfyRegistryWheelInputError
from local_lm.models import ComfyRegistrySourceArtifactReview

if TYPE_CHECKING:
    from local_lm.comfy_registry_mixed_wheel_closure import ComfyRegistryMixedWheelClosure


def _context(context: tuple[Session, ArtifactStore]) -> ComfyRegistryReviewedInputContext:
    session, store = context
    return ComfyRegistryReviewedInputContext(
        lambda: Session(session.get_bind()), store, _environment(), ("py3-none-any",)
    )


async def _closure(context: tuple[Session, ArtifactStore]) -> ComfyRegistryMixedWheelClosure:
    _review(context)
    return await _drive([DECLARATION], context, _Sources({}, {}), runtime={"runtime": "1.0"})


async def test_persisted_binding_round_trips_exact_closure_and_current_reviews(
    source_review_context: tuple[Session, ArtifactStore],
) -> None:
    from local_lm.comfy_registry_reviewed_closure_binding import (
        reviewed_closure_binding_payload,
        validate_current_reviewed_closure_binding,
    )

    closure = await _closure(source_review_context)
    payload = reviewed_closure_binding_payload(closure)
    serialized = json.dumps(payload)
    assert "Metadata-Version" not in serialized
    assert json.dumps(str(source_review_context[1].root))[1:-1] not in serialized
    manifest = validate_current_reviewed_closure_binding(
        json.loads(serialized),
        expected_closure_sha256=closure.closure_sha256,
        declarations=[DECLARATION],
        reviewed_inputs=_context(source_review_context),
    )
    assert manifest == closure.manifest


async def test_another_valid_review_cannot_replace_the_installed_closure(
    source_review_context: tuple[Session, ArtifactStore],
) -> None:
    from local_lm.comfy_registry_reviewed_closure_binding import (
        reviewed_closure_binding_payload,
        validate_current_reviewed_closure_binding,
    )

    original = await _closure(source_review_context)
    session, store = source_review_context
    session.execute(delete(ComfyRegistrySourceArtifactReview))
    session.commit()
    _review_metadata(session, store, _HEADERS + b"\nChanged package description.\n")
    replacement = await _drive(
        [DECLARATION], source_review_context, _Sources({}, {}), runtime={"runtime": "1.0"}
    )
    assert original.closure_sha256 != replacement.closure_sha256
    payload = reviewed_closure_binding_payload(replacement)
    context = _context(source_review_context)
    assert (
        validate_current_reviewed_closure_binding(
            payload,
            expected_closure_sha256=replacement.closure_sha256,
            declarations=[DECLARATION],
            reviewed_inputs=context,
        )
        == replacement.manifest
    )
    with pytest.raises(ComfyRegistryWheelInputError) as error:
        validate_current_reviewed_closure_binding(
            payload,
            expected_closure_sha256=original.closure_sha256,
            declarations=[DECLARATION],
            reviewed_inputs=context,
        )
    assert error.value.code == "reviewed_closure_hash_mismatch"


async def test_binding_cannot_mix_original_witness_with_replacement_inputs(
    source_review_context: tuple[Session, ArtifactStore],
) -> None:
    from local_lm.comfy_registry_reviewed_closure_binding import (
        parse_reviewed_closure_binding,
        reviewed_closure_binding_payload,
    )

    original = await _closure(source_review_context)
    session, store = source_review_context
    session.execute(delete(ComfyRegistrySourceArtifactReview))
    session.commit()
    _review_metadata(session, store, _HEADERS + b"\nAnother package description.\n")
    replacement = await _drive([DECLARATION], source_review_context, _Sources({}, {}))
    payload = reviewed_closure_binding_payload(original)
    payload["inputs"] = reviewed_closure_binding_payload(replacement)["inputs"]
    with pytest.raises(ComfyRegistryWheelInputError) as error:
        parse_reviewed_closure_binding(
            payload, expected_closure_sha256=original.closure_sha256, declarations=[DECLARATION]
        )
    assert error.value.code == "reviewed_closure_input_mismatch"


@pytest.mark.parametrize("problem", ["revoked", "target"])
async def test_matching_persisted_identity_still_requires_current_review_and_target(
    source_review_context: tuple[Session, ArtifactStore], problem: str
) -> None:
    from local_lm.comfy_registry_reviewed_closure_binding import (
        reviewed_closure_binding_payload,
        validate_current_reviewed_closure_binding,
    )

    closure = await _closure(source_review_context)
    context = _context(source_review_context)
    if problem == "revoked":
        session, _store = source_review_context
        session.execute(delete(ComfyRegistrySourceArtifactReview))
        session.commit()
    else:
        context = replace(context, supported_tags=("py2-none-any",))
    with pytest.raises(ValueError):
        validate_current_reviewed_closure_binding(
            reviewed_closure_binding_payload(closure),
            expected_closure_sha256=closure.closure_sha256,
            declarations=[DECLARATION],
            reviewed_inputs=context,
        )


@pytest.mark.parametrize("declarations", [[], ["example-pkg==1.2.3"], ["invalid @"]])
async def test_binding_refuses_changed_or_downgraded_declarations(
    source_review_context: tuple[Session, ArtifactStore], declarations: list[str]
) -> None:
    from local_lm.comfy_registry_reviewed_closure_binding import (
        parse_reviewed_closure_binding,
        reviewed_closure_binding_payload,
    )

    closure = await _closure(source_review_context)
    with pytest.raises(ComfyRegistryWheelInputError) as error:
        parse_reviewed_closure_binding(
            reviewed_closure_binding_payload(closure),
            expected_closure_sha256=closure.closure_sha256,
            declarations=declarations,
        )
    assert error.value.code == "reviewed_closure_declaration_mismatch"


async def test_all_inactive_source_declarations_keep_a_required_binding(
    source_review_context: tuple[Session, ArtifactStore],
) -> None:
    from local_lm.comfy_registry_reviewed_closure_binding import (
        parse_reviewed_closure_binding,
        reviewed_closure_binding_payload,
        validate_current_reviewed_closure_binding,
    )

    declarations = [DECLARATION + ' ; sys_platform == "missing-platform"']
    closure = await _drive(declarations, source_review_context, _Sources({}, {}))
    assert not closure.manifest.reviewed_local
    payload = reviewed_closure_binding_payload(closure)
    assert (
        validate_current_reviewed_closure_binding(
            payload,
            expected_closure_sha256=closure.closure_sha256,
            declarations=declarations,
            reviewed_inputs=_context(source_review_context),
        )
        == closure.manifest
    )
    for missing in (None, {}, {"version": 1}):
        with pytest.raises(ComfyRegistryWheelInputError):
            parse_reviewed_closure_binding(
                missing, expected_closure_sha256=closure.closure_sha256, declarations=declarations
            )


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("version", True),
        ("version", 2),
        ("kind", "remote-wheel-inputs"),
        ("manifest_sha256", "f" * 64),
        ("metadata_plan_sha256", "invalid"),
        ("complete", 1),
        ("complete", False),
        ("pending_projects", ["helper"]),
        ("manifest_history", []),
        ("manifest_history", ["f" * 64] * 66),
        ("manifest_history", [{"invalid": True}]),
        ("round_number", True),
        ("round_number", 1),
        ("runtime_distributions", {}),
        ("runtime_distributions", [{"name": "runtime", "version": "1.0"}] * 4097),
        ("runtime_distributions", [{"name": "runtime", "version": "1.0", "extra": 1}]),
        ("runtime_distributions", [{"name": "runtime", "version": 1}]),
        ("runtime_distributions", [{"name": "runtime", "version": "invalid"}]),
        ("runtime_distributions", [{"name": "runtime", "version": "1" * 201}]),
        ("runtime_distributions", [{"name": "Runtime_Name", "version": "1.0"}]),
        ("unknown", "field"),
    ],
)
async def test_malformed_witness_refuses_even_with_a_matching_external_digest(
    source_review_context: tuple[Session, ArtifactStore], key: str, value: object
) -> None:
    from local_lm.comfy_registry_reviewed_closure_binding import (
        parse_reviewed_closure_binding,
        reviewed_closure_binding_payload,
    )

    closure = await _closure(source_review_context)
    payload: dict[str, Any] = deepcopy(reviewed_closure_binding_payload(closure))
    payload["closure"][key] = value
    digest = hashlib.sha256(
        json.dumps(payload["closure"], sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    with pytest.raises(ComfyRegistryWheelInputError) as error:
        parse_reviewed_closure_binding(
            payload, expected_closure_sha256=digest, declarations=[DECLARATION]
        )
    assert error.value.code == "invalid_reviewed_closure_binding"


@pytest.mark.parametrize("problem", ["unknown", "boolean_version", "missing", "bad_hash"])
async def test_binding_envelope_requires_exact_fields_and_external_identity(
    source_review_context: tuple[Session, ArtifactStore], problem: str
) -> None:
    from local_lm.comfy_registry_reviewed_closure_binding import (
        parse_reviewed_closure_binding,
        reviewed_closure_binding_payload,
    )

    closure = await _closure(source_review_context)
    payload = reviewed_closure_binding_payload(closure)
    if problem == "unknown":
        payload["unknown"] = True
    elif problem == "boolean_version":
        payload["version"] = True
    elif problem == "missing":
        payload.pop("closure")
    with pytest.raises(ComfyRegistryWheelInputError):
        parse_reviewed_closure_binding(
            payload,
            expected_closure_sha256="invalid" if problem == "bad_hash" else closure.closure_sha256,
            declarations=[DECLARATION],
        )


async def test_incomplete_closure_cannot_be_persisted_as_an_installed_binding(
    source_review_context: tuple[Session, ArtifactStore],
) -> None:
    from test_comfy_registry_mixed_wheel_closure import _start
    from test_comfy_registry_mixed_wheel_metadata import _mixed

    from local_lm.comfy_registry_reviewed_closure_binding import reviewed_closure_binding_payload

    manifest, documents = _mixed(source_review_context, ["helper>=2"], [])
    closure = _start(manifest, documents)
    assert not closure.complete
    with pytest.raises(ComfyRegistryWheelInputError):
        reviewed_closure_binding_payload(closure)

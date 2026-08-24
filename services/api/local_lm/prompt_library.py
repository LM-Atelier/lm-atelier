"""Bounded persistence operations for the immutable Prompt Library."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import NoReturn, cast

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from .domain import Operation
from .models import (
    ModelAssetInstall,
    PromptTemplateDefinition,
    PromptTemplateRevision,
    WorkflowActivation,
    WorkflowDefinition,
    WorkflowRevision,
)
from .prompt_templates import (
    PromptTemplateContract,
    PromptTemplateError,
    PromptTemplateLoraPolicyMode,
    PromptTemplateResourceMode,
    parse_prompt_template_contract,
    prompt_template_contract_payload,
    prompt_template_contract_sha256,
)
from .workflow_bindings import WorkflowBindingError, materialize_model_asset

PROMPT_LIBRARY_INVALID = "Prompt template request is invalid."
PROMPT_LIBRARY_NOT_FOUND = "Prompt template does not exist."
PROMPT_LIBRARY_CONFLICT = "Prompt template conflicts with an existing template."
PROMPT_LIBRARY_STALE = "Prompt template changed. Refresh and try again."
PROMPT_LIBRARY_RESOURCES_UNAVAILABLE = "Prompt template resources are unavailable."

_IDEMPOTENCY_KEY = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,199}", re.ASCII)
_SHA256 = re.compile(r"[0-9a-f]{64}", re.ASCII)


class PromptLibraryError(ValueError):
    """One stable, non-echoing Prompt Library boundary failure."""

    def __init__(self, code: str, message: str, *, status_code: int) -> None:
        super().__init__(message)
        self.code = code
        self.status_code = status_code


@dataclass(frozen=True, slots=True)
class PromptTemplateWriteResult:
    definition: PromptTemplateDefinition
    revision: PromptTemplateRevision
    idempotent: bool


def _invalid() -> NoReturn:
    raise PromptLibraryError(
        "prompt-template-request-invalid",
        PROMPT_LIBRARY_INVALID,
        status_code=422,
    )


def _conflict() -> NoReturn:
    raise PromptLibraryError(
        "prompt-template-conflict",
        PROMPT_LIBRARY_CONFLICT,
        status_code=409,
    )


def _stale() -> NoReturn:
    raise PromptLibraryError(
        "prompt-template-stale",
        PROMPT_LIBRARY_STALE,
        status_code=409,
    )


def _resources_unavailable() -> NoReturn:
    raise PromptLibraryError(
        "prompt-template-resources-unavailable",
        PROMPT_LIBRARY_RESOURCES_UNAVAILABLE,
        status_code=409,
    )


def _text(value: object, *, maximum: int, allow_empty: bool) -> str:
    if type(value) is not str or len(value) > maximum or "\x00" in value:
        _invalid()
    normalized = value.strip()
    if not allow_empty and not normalized:
        _invalid()
    return normalized


def _idempotency_key(value: object) -> str:
    if type(value) is not str or _IDEMPOTENCY_KEY.fullmatch(value) is None:
        _invalid()
    return value


def _stable_id(prefix: str, *parts: str) -> str:
    encoded = "\x00".join(("prompt-library-v1", prefix, *parts)).encode("utf-8")
    return f"{prefix}_{hashlib.sha256(encoded).hexdigest()[:32]}"


def _current_revision(
    session: Session,
    definition: PromptTemplateDefinition,
) -> PromptTemplateRevision:
    revision = (
        session.get(PromptTemplateRevision, definition.current_revision_id)
        if definition.current_revision_id is not None
        else None
    )
    if revision is None or revision.prompt_template_id != definition.id:
        _conflict()
    return revision


def _workflow_revision_is_ready(
    session: Session,
    revision: WorkflowRevision,
    *,
    expected_engine: str,
) -> bool:
    definition = session.get(WorkflowDefinition, revision.workflow_id)
    if (
        definition is None
        or definition.operation != Operation.TEXT_TO_IMAGE.value
        or revision.engine != expected_engine
        or (expected_engine != "mock" and not revision.api_graph_json)
        or not revision.trusted
    ):
        return False
    if revision.dependency_contract_sha256 is None:
        return True
    activation = session.scalar(
        select(WorkflowActivation).where(
            WorkflowActivation.workflow_revision_id == revision.id,
            WorkflowActivation.is_active.is_(True),
            WorkflowActivation.state == "ready",
            WorkflowActivation.invalidated_at.is_(None),
        )
    )
    launch_sha256 = (
        activation.details_json.get("launch_sha256")
        if activation is not None and type(activation.details_json) is dict
        else None
    )
    return bool(
        activation is not None
        and activation.dependency_contract_sha256 == revision.dependency_contract_sha256
        and type(launch_sha256) is str
        and _SHA256.fullmatch(launch_sha256) is not None
    )


def validate_prompt_template_resources(
    session: Session,
    contract: PromptTemplateContract,
    *,
    expected_engine: str,
) -> None:
    """Prove every fixed local resource without returning mutable names or paths."""

    policy = contract.resource_policy
    if policy.mode is PromptTemplateResourceMode.INHERITED:
        return
    resources = (
        ((policy.workflow_revision_id, policy.lora_policy),)
        if policy.mode is PromptTemplateResourceMode.FIXED
        else tuple((option.workflow_revision_id, option.lora_policy) for option in policy.options)
    )
    for revision_id, _ in resources:
        revision = session.get(WorkflowRevision, revision_id) if revision_id is not None else None
        if revision is None or not _workflow_revision_is_ready(
            session,
            revision,
            expected_engine=expected_engine,
        ):
            _resources_unavailable()
    expected: set[str] = set()
    for _, lora_policy in resources:
        if lora_policy is None:
            _resources_unavailable()
        if lora_policy.mode is PromptTemplateLoraPolicyMode.FIXED:
            expected.update(item.sha256 for item in lora_policy.stack)
        elif lora_policy.mode is PromptTemplateLoraPolicyMode.POOL:
            expected.update(item.sha256 for stack in lora_policy.stacks for item in stack)
    if not expected:
        return
    candidates = session.scalars(
        select(ModelAssetInstall).where(
            ModelAssetInstall.kind == "lora",
            ModelAssetInstall.active.is_(True),
            ModelAssetInstall.verified_at.is_not(None),
        )
    ).all()
    matched: set[str] = set()
    for candidate in candidates:
        if type(candidate.manifest_json) is not dict:
            continue
        digest = candidate.manifest_json.get("sha256")
        if type(digest) is not str or digest not in expected:
            continue
        try:
            materialized = materialize_model_asset(candidate)
        except WorkflowBindingError:
            continue
        identity_digest = materialized.identity.get("sha256")
        if identity_digest == digest:
            matched.add(digest)
    if matched != expected:
        _resources_unavailable()


def _canonical_contract(value: object) -> tuple[PromptTemplateContract, dict[str, object], str]:
    try:
        contract = parse_prompt_template_contract(value)
    except PromptTemplateError as exc:
        raise PromptLibraryError(
            exc.code,
            str(exc),
            status_code=exc.status_code,
        ) from exc
    return (
        contract,
        prompt_template_contract_payload(contract),
        prompt_template_contract_sha256(contract),
    )


def _exact_create_retry(
    session: Session,
    *,
    definition_id: str,
    revision_id: str,
    name: str,
    description: str,
    digest: str,
) -> PromptTemplateWriteResult | None:
    definition = session.get(PromptTemplateDefinition, definition_id)
    revision = session.get(PromptTemplateRevision, revision_id)
    if definition is None and revision is None:
        return None
    if (
        definition is None
        or revision is None
        or revision.prompt_template_id != definition.id
        or definition.name != name
        or definition.description != description
        or definition.archived
        or definition.current_revision_id != revision.id
        or revision.version != 1
        or revision.contract_sha256 != digest
    ):
        _conflict()
    return PromptTemplateWriteResult(definition, revision, True)


def create_prompt_template(
    session: Session,
    *,
    idempotency_key: object,
    name: object,
    description: object,
    contract_value: object,
    expected_engine: str,
) -> PromptTemplateWriteResult:
    key = _idempotency_key(idempotency_key)
    normalized_name = _text(name, maximum=200, allow_empty=False)
    normalized_description = _text(description, maximum=4_000, allow_empty=True)
    contract, payload, digest = _canonical_contract(contract_value)
    validate_prompt_template_resources(session, contract, expected_engine=expected_engine)
    definition_id = _stable_id("ptdef", key)
    revision_id = _stable_id("ptrev", definition_id, key)
    retried = _exact_create_retry(
        session,
        definition_id=definition_id,
        revision_id=revision_id,
        name=normalized_name,
        description=normalized_description,
        digest=digest,
    )
    if retried is not None:
        return retried
    definition = PromptTemplateDefinition(
        id=definition_id,
        name=normalized_name,
        description=normalized_description,
        archived=False,
    )
    revision = PromptTemplateRevision(
        id=revision_id,
        prompt_template_id=definition_id,
        version=1,
        schema_version=contract.schema_version,
        contract_json=payload,
        contract_sha256=digest,
    )
    try:
        session.add(definition)
        session.flush()
        session.add(revision)
        session.flush()
        definition.current_revision_id = revision.id
        session.commit()
    except IntegrityError as exc:
        session.rollback()
        retried = _exact_create_retry(
            session,
            definition_id=definition_id,
            revision_id=revision_id,
            name=normalized_name,
            description=normalized_description,
            digest=digest,
        )
        if retried is not None:
            return retried
        raise PromptLibraryError(
            "prompt-template-conflict",
            PROMPT_LIBRARY_CONFLICT,
            status_code=409,
        ) from exc
    return PromptTemplateWriteResult(definition, revision, False)


def _metadata_matches(
    definition: PromptTemplateDefinition,
    *,
    name: str | None,
    description: str | None,
    archived: bool | None,
) -> bool:
    return (
        (name is None or definition.name == name)
        and (description is None or definition.description == description)
        and (archived is None or definition.archived is archived)
    )


def update_prompt_template(
    session: Session,
    *,
    definition_id: str,
    expected_current_revision_id: object,
    name: object | None,
    description: object | None,
    archived: object | None,
    contract_value: object | None,
    idempotency_key: object | None,
    expected_engine: str,
) -> PromptTemplateWriteResult:
    if type(definition_id) is not str or not definition_id:
        _invalid()
    definition = session.get(PromptTemplateDefinition, definition_id)
    if definition is None:
        raise PromptLibraryError(
            "prompt-template-not-found",
            PROMPT_LIBRARY_NOT_FOUND,
            status_code=404,
        )
    if type(expected_current_revision_id) is not str or not expected_current_revision_id:
        _invalid()
    normalized_name = _text(name, maximum=200, allow_empty=False) if name is not None else None
    normalized_description = (
        _text(description, maximum=4_000, allow_empty=True) if description is not None else None
    )
    if type(archived) not in {bool, type(None)}:
        _invalid()
    normalized_archived = cast(bool | None, archived)
    if (
        normalized_name is None
        and normalized_description is None
        and normalized_archived is None
        and contract_value is None
    ):
        _invalid()

    contract: PromptTemplateContract | None = None
    payload: dict[str, object] | None = None
    digest: str | None = None
    revision_id: str | None = None
    if contract_value is not None:
        key = _idempotency_key(idempotency_key)
        contract, payload, digest = _canonical_contract(contract_value)
        validate_prompt_template_resources(session, contract, expected_engine=expected_engine)
        revision_id = _stable_id("ptrev", definition.id, key)
        prior = session.get(PromptTemplateRevision, revision_id)
        if prior is not None:
            if (
                prior.prompt_template_id != definition.id
                or prior.contract_sha256 != digest
                or definition.current_revision_id != prior.id
                or not _metadata_matches(
                    definition,
                    name=normalized_name,
                    description=normalized_description,
                    archived=normalized_archived,
                )
            ):
                _conflict()
            return PromptTemplateWriteResult(definition, prior, True)
    elif idempotency_key is not None:
        _invalid()

    current = _current_revision(session, definition)
    if definition.current_revision_id != expected_current_revision_id:
        _stale()
    if normalized_name is not None:
        definition.name = normalized_name
    if normalized_description is not None:
        definition.description = normalized_description
    if normalized_archived is not None:
        definition.archived = normalized_archived

    if digest is None or digest == current.contract_sha256:
        try:
            session.commit()
        except IntegrityError as exc:
            session.rollback()
            raise PromptLibraryError(
                "prompt-template-conflict",
                PROMPT_LIBRARY_CONFLICT,
                status_code=409,
            ) from exc
        return PromptTemplateWriteResult(definition, current, digest is not None)

    if contract is None or payload is None or revision_id is None:
        _invalid()
    revision = PromptTemplateRevision(
        id=revision_id,
        prompt_template_id=definition.id,
        version=current.version + 1,
        schema_version=contract.schema_version,
        contract_json=payload,
        contract_sha256=digest,
    )
    try:
        session.add(revision)
        session.flush()
        definition.current_revision_id = revision.id
        session.commit()
    except IntegrityError as exc:
        session.rollback()
        prior = session.get(PromptTemplateRevision, revision_id)
        definition = session.get(PromptTemplateDefinition, definition_id)
        if (
            prior is not None
            and definition is not None
            and prior.prompt_template_id == definition.id
            and prior.contract_sha256 == digest
            and definition.current_revision_id == prior.id
            and _metadata_matches(
                definition,
                name=normalized_name,
                description=normalized_description,
                archived=normalized_archived,
            )
        ):
            return PromptTemplateWriteResult(definition, prior, True)
        raise PromptLibraryError(
            "prompt-template-stale",
            PROMPT_LIBRARY_STALE,
            status_code=409,
        ) from exc
    return PromptTemplateWriteResult(definition, revision, False)


def restore_prompt_template_revision(
    session: Session,
    *,
    definition_id: str,
    revision_id: str,
    expected_current_revision_id: object,
    idempotency_key: object,
    expected_engine: str,
) -> PromptTemplateWriteResult:
    source = session.get(PromptTemplateRevision, revision_id)
    if source is None or source.prompt_template_id != definition_id:
        raise PromptLibraryError(
            "prompt-template-revision-not-found",
            PROMPT_LIBRARY_NOT_FOUND,
            status_code=404,
        )
    return update_prompt_template(
        session,
        definition_id=definition_id,
        expected_current_revision_id=expected_current_revision_id,
        name=None,
        description=None,
        archived=None,
        contract_value=source.contract_json,
        idempotency_key=idempotency_key,
        expected_engine=expected_engine,
    )

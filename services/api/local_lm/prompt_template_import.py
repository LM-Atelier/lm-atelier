"""Atomic persistence for fully authorized portable Prompt Template imports."""

from __future__ import annotations

import hashlib
import hmac
import json
import re
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import NoReturn, cast

from sqlalchemy.exc import IntegrityError, OperationalError
from sqlalchemy.orm import Session

from .models import (
    PromptTemplateDefinition,
    PromptTemplateImportWinner,
    PromptTemplateRevision,
)
from .prompt_library import PromptLibraryError, validate_prompt_template_resources
from .prompt_template_portability import (
    PORTABLE_AUTHORITY_RULE,
    PortablePromptTemplateBundle,
    PromptTemplatePortabilityError,
    bind_portable_prompt_template_contract,
    parse_portable_prompt_template_bundle,
    prompt_template_import_lora_authority,
    prompt_template_import_workflow_authority,
    verify_prompt_template_candidate_receipt,
    verify_prompt_template_import_receipt,
)
from .prompt_templates import (
    PromptTemplateContract,
    prompt_template_contract_payload,
    prompt_template_contract_sha256,
)

PROMPT_TEMPLATE_IMPORT_INVALID = "Prompt template import request is invalid."
PROMPT_TEMPLATE_IMPORT_CONFLICT = "Prompt template import conflicts with an existing import."
PROMPT_TEMPLATE_IMPORT_BUSY = "Prompt template import is busy. Try again."

MAX_IMPORT_WORKFLOWS = 16
MAX_IMPORT_LORAS = 64

_IDEMPOTENCY_KEY = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,199}", re.ASCII)
_BINDING_KEY = re.compile(r"workflow_[1-9][0-9]{0,2}", re.ASCII)
_LOCAL_REF = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,39}", re.ASCII)
_SHA256 = re.compile(r"[0-9a-f]{64}", re.ASCII)
_SEMANTIC_CONTEXT = b"lm-atelier-prompt-template-import-request-v1\0"


class PromptTemplateImportError(ValueError):
    """A stable non-echoing atomic-import failure."""

    def __init__(self, code: str, message: str, *, status_code: int) -> None:
        super().__init__(message)
        self.code = code
        self.status_code = status_code


@dataclass(frozen=True, slots=True)
class PromptTemplateImportResult:
    template_id: str
    revision_id: str
    contract_sha256: str
    idempotent: bool


@dataclass(frozen=True, slots=True)
class _PreparedImport:
    key: str
    destination_name: str
    bundle: PortablePromptTemplateBundle
    workflow_rows: tuple[tuple[str, str, str], ...]
    lora_rows: tuple[tuple[str, str], ...]
    contract: PromptTemplateContract
    contract_sha256: str
    request_sha256: str
    template_id: str
    revision_id: str


def _invalid() -> NoReturn:
    raise PromptTemplateImportError(
        "prompt-template-import-request-invalid",
        PROMPT_TEMPLATE_IMPORT_INVALID,
        status_code=422,
    )


def _conflict() -> NoReturn:
    raise PromptTemplateImportError(
        "prompt-template-import-conflict",
        PROMPT_TEMPLATE_IMPORT_CONFLICT,
        status_code=409,
    )


def _busy() -> NoReturn:
    raise PromptTemplateImportError(
        "prompt-template-import-busy",
        PROMPT_TEMPLATE_IMPORT_BUSY,
        status_code=409,
    )


def _canonical_json(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        ).encode("utf-8")
    except (RecursionError, UnicodeError, ValueError) as exc:
        raise PromptTemplateImportError(
            "prompt-template-import-request-invalid",
            PROMPT_TEMPLATE_IMPORT_INVALID,
            status_code=422,
        ) from exc


def _stable_id(prefix: str, *parts: str) -> str:
    encoded = "\0".join(("prompt-library-v1", prefix, *parts)).encode("utf-8")
    return f"{prefix}_{hashlib.sha256(encoded).hexdigest()[:32]}"


def _exact_rows(
    value: object,
    *,
    maximum: int,
    fields: frozenset[str],
) -> tuple[dict[str, object], ...]:
    if (
        isinstance(value, str | bytes)
        or not isinstance(value, Sequence)
        or len(value) > maximum
        or any(type(item) is not dict or set(item) != fields for item in value)
    ):
        _invalid()
    return tuple(cast(dict[str, object], item) for item in value)


def _prepare(
    *,
    idempotency_key: object,
    raw_bundle: object,
    confirmed_bundle_sha256: object,
    destination_name: object,
    workflow_bindings: object,
    lora_confirmations: object,
) -> _PreparedImport:
    if type(idempotency_key) is not str or _IDEMPOTENCY_KEY.fullmatch(idempotency_key) is None:
        _invalid()
    if type(destination_name) is not str or "\0" in destination_name or len(destination_name) > 200:
        _invalid()
    normalized_name = destination_name.strip()
    if not normalized_name:
        _invalid()
    bundle = parse_portable_prompt_template_bundle(raw_bundle)
    if (
        type(confirmed_bundle_sha256) is not str
        or _SHA256.fullmatch(confirmed_bundle_sha256) is None
        or not hmac.compare_digest(confirmed_bundle_sha256, bundle.bundle_sha256)
    ):
        _conflict()

    submitted_workflows = _exact_rows(
        workflow_bindings,
        maximum=MAX_IMPORT_WORKFLOWS,
        fields=frozenset({"binding_key", "local_ref", "candidate_receipt"}),
    )
    workflow_by_key: dict[str, tuple[str, str]] = {}
    for row in submitted_workflows:
        key, local_ref, receipt = (
            row["binding_key"],
            row["local_ref"],
            row["candidate_receipt"],
        )
        if (
            type(key) is not str
            or _BINDING_KEY.fullmatch(key) is None
            or type(local_ref) is not str
            or _LOCAL_REF.fullmatch(local_ref) is None
            or type(receipt) is not str
            or not receipt
            or len(receipt) > 2_048
            or key in workflow_by_key
        ):
            _invalid()
        workflow_by_key[key] = (local_ref, receipt)
    required_workflows = tuple(str(item["key"]) for item in bundle.workflow_requirements)
    if set(workflow_by_key) != set(required_workflows):
        _invalid()
    ordered_workflows = tuple(
        (key, workflow_by_key[key][0], workflow_by_key[key][1]) for key in required_workflows
    )

    submitted_loras = _exact_rows(
        lora_confirmations,
        maximum=MAX_IMPORT_LORAS,
        fields=frozenset({"sha256", "confirmation_receipt"}),
    )
    lora_by_digest: dict[str, str] = {}
    for row in submitted_loras:
        digest, receipt = row["sha256"], row["confirmation_receipt"]
        if (
            type(digest) is not str
            or _SHA256.fullmatch(digest) is None
            or type(receipt) is not str
            or not receipt
            or len(receipt) > 2_048
            or digest in lora_by_digest
        ):
            _invalid()
        lora_by_digest[digest] = receipt
    if set(lora_by_digest) != set(bundle.lora_requirements):
        _invalid()
    ordered_loras = tuple((digest, lora_by_digest[digest]) for digest in bundle.lora_requirements)

    contract = bind_portable_prompt_template_contract(
        bundle,
        {key: local_ref for key, local_ref, _receipt in ordered_workflows},
    )
    contract_sha256 = prompt_template_contract_sha256(contract)
    semantic = {
        "version": 1,
        "bundle_sha256": bundle.bundle_sha256,
        "authority_rule": PORTABLE_AUTHORITY_RULE,
        "destination_name": normalized_name,
        "workflow_bindings": [
            {"binding_key": key, "local_ref": local_ref}
            for key, local_ref, _receipt in ordered_workflows
        ],
        "lora_confirmations": [{"sha256": digest} for digest, _receipt in ordered_loras],
        "contract_sha256": contract_sha256,
    }
    request_sha256 = hashlib.sha256(_SEMANTIC_CONTEXT + _canonical_json(semantic)).hexdigest()
    template_id = _stable_id("ptdef", "import", idempotency_key)
    revision_id = _stable_id("ptrev", "import", template_id, idempotency_key)
    return _PreparedImport(
        key=idempotency_key,
        destination_name=normalized_name,
        bundle=bundle,
        workflow_rows=ordered_workflows,
        lora_rows=ordered_loras,
        contract=contract,
        contract_sha256=contract_sha256,
        request_sha256=request_sha256,
        template_id=template_id,
        revision_id=revision_id,
    )


def _winner_result(
    winner: PromptTemplateImportWinner | None,
    prepared: _PreparedImport,
    *,
    idempotent: bool,
) -> PromptTemplateImportResult | None:
    if winner is None:
        return None
    if (
        winner.request_sha256 != prepared.request_sha256
        or winner.bundle_sha256 != prepared.bundle.bundle_sha256
        or winner.authority_rule != PORTABLE_AUTHORITY_RULE
        or winner.prompt_template_id != prepared.template_id
        or winner.prompt_template_revision_id != prepared.revision_id
        or winner.contract_sha256 != prepared.contract_sha256
    ):
        _conflict()
    return PromptTemplateImportResult(
        template_id=winner.prompt_template_id,
        revision_id=winner.prompt_template_revision_id,
        contract_sha256=winner.contract_sha256,
        idempotent=idempotent,
    )


def _retry_after_database_error(
    session: Session,
    prepared: _PreparedImport,
) -> PromptTemplateImportResult:
    session.rollback()
    try:
        winner = session.get(PromptTemplateImportWinner, prepared.key)
        result = _winner_result(winner, prepared, idempotent=True)
    except OperationalError:
        session.rollback()
        _busy()
    except PromptTemplateImportError:
        session.rollback()
        raise
    session.rollback()
    if result is not None:
        return result
    _conflict()


def commit_prompt_template_import(
    session: Session,
    *,
    idempotency_key: object,
    raw_bundle: object,
    preview_receipt: object,
    confirmed_bundle_sha256: object,
    destination_name: object,
    workflow_bindings: object,
    lora_confirmations: object,
    expected_engine: str,
    signing_key: bytes,
    now: int | None = None,
    _stage_hook: Callable[[str], None] | None = None,
) -> PromptTemplateImportResult:
    """Commit one exact import or replay its immutable semantic winner."""

    prepared = _prepare(
        idempotency_key=idempotency_key,
        raw_bundle=raw_bundle,
        confirmed_bundle_sha256=confirmed_bundle_sha256,
        destination_name=destination_name,
        workflow_bindings=workflow_bindings,
        lora_confirmations=lora_confirmations,
    )
    if session.in_transaction() or session.new or session.dirty or session.deleted:
        _conflict()
    if session.get_bind().dialect.name != "sqlite":
        _conflict()

    try:
        early = _winner_result(
            session.get(PromptTemplateImportWinner, prepared.key),
            prepared,
            idempotent=True,
        )
        if early is not None:
            session.rollback()
            return early
        session.rollback()
        connection = session.connection()
        driver = connection.connection.driver_connection
        if bool(getattr(driver, "in_transaction", False)):
            session.rollback()
            _conflict()
        connection.exec_driver_sql("BEGIN IMMEDIATE")
    except OperationalError:
        return _retry_after_database_error(session, prepared)
    except PromptTemplateImportError:
        session.rollback()
        raise

    try:
        fenced = _winner_result(
            session.get(PromptTemplateImportWinner, prepared.key),
            prepared,
            idempotent=True,
        )
        if fenced is not None:
            session.rollback()
            return fenced

        reparsed = _prepare(
            idempotency_key=idempotency_key,
            raw_bundle=raw_bundle,
            confirmed_bundle_sha256=confirmed_bundle_sha256,
            destination_name=destination_name,
            workflow_bindings=workflow_bindings,
            lora_confirmations=lora_confirmations,
        )
        if reparsed.request_sha256 != prepared.request_sha256:
            _conflict()
        current = int(time.time()) if now is None else now
        expires_at = verify_prompt_template_import_receipt(
            preview_receipt,
            reparsed.bundle,
            signing_key=signing_key,
            now=current,
        )
        requirements = {
            str(item["key"]): cast(dict[str, object], item["descriptor"])
            for item in reparsed.bundle.workflow_requirements
        }
        for key, local_ref, receipt in reparsed.workflow_rows:
            authority = prompt_template_import_workflow_authority(
                session,
                local_ref=local_ref,
                expected_engine=expected_engine,
            )
            if authority is None or authority[0] != requirements[key]:
                _conflict()
            verify_prompt_template_candidate_receipt(
                receipt,
                {
                    "kind": "workflow",
                    "bundle_sha256": reparsed.bundle.bundle_sha256,
                    "binding_key": key,
                    "local_ref": local_ref,
                    "authority_sha256": authority[1],
                    "expires_at": expires_at,
                },
                signing_key=signing_key,
                now=current,
            )
        for digest, receipt in reparsed.lora_rows:
            authority_sha256 = prompt_template_import_lora_authority(
                session,
                sha256=digest,
            )
            if authority_sha256 is None:
                _conflict()
            verify_prompt_template_candidate_receipt(
                receipt,
                {
                    "kind": "lora",
                    "bundle_sha256": reparsed.bundle.bundle_sha256,
                    "sha256": digest,
                    "authority_sha256": authority_sha256,
                    "expires_at": expires_at,
                },
                signing_key=signing_key,
                now=current,
            )
        validate_prompt_template_resources(
            session,
            reparsed.contract,
            expected_engine=expected_engine,
        )
        template = cast(dict[str, object], reparsed.bundle.payload["template"])
        definition = PromptTemplateDefinition(
            id=reparsed.template_id,
            name=reparsed.destination_name,
            description=cast(str, template["description"]),
            archived=False,
        )
        revision = PromptTemplateRevision(
            id=reparsed.revision_id,
            prompt_template_id=reparsed.template_id,
            version=1,
            schema_version=reparsed.contract.schema_version,
            contract_json=prompt_template_contract_payload(reparsed.contract),
            contract_sha256=reparsed.contract_sha256,
        )
        session.add(definition)
        session.flush()
        if _stage_hook is not None:
            _stage_hook("definition")
        session.add(revision)
        session.flush()
        if _stage_hook is not None:
            _stage_hook("revision")
        definition.current_revision_id = revision.id
        session.flush()
        if _stage_hook is not None:
            _stage_hook("current")
        winner = PromptTemplateImportWinner(
            idempotency_key=reparsed.key,
            request_sha256=reparsed.request_sha256,
            bundle_sha256=reparsed.bundle.bundle_sha256,
            authority_rule=PORTABLE_AUTHORITY_RULE,
            prompt_template_id=reparsed.template_id,
            prompt_template_revision_id=reparsed.revision_id,
            contract_sha256=reparsed.contract_sha256,
        )
        session.add(winner)
        session.flush()
        if _stage_hook is not None:
            _stage_hook("winner")
        session.commit()
        return PromptTemplateImportResult(
            template_id=reparsed.template_id,
            revision_id=reparsed.revision_id,
            contract_sha256=reparsed.contract_sha256,
            idempotent=False,
        )
    except IntegrityError:
        return _retry_after_database_error(session, prepared)
    except OperationalError:
        return _retry_after_database_error(session, prepared)
    except (PromptLibraryError, PromptTemplateImportError, PromptTemplatePortabilityError):
        session.rollback()
        raise
    except BaseException:
        session.rollback()
        raise

from __future__ import annotations

import hashlib
import json
from collections.abc import AsyncIterator, Generator
from copy import deepcopy
from dataclasses import asdict, dataclass
from types import SimpleNamespace
from typing import Any, ClassVar, cast
from unittest.mock import AsyncMock

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from local_lm.adapters.base import MediaEvent, MediaRequest
from local_lm.adapters.mock import MockMediaAdapter
from local_lm.auxiliary_assets import (
    checkpoint_lora_extension,
    resolve_lora_stack,
    resolve_lora_stack_against_graph,
)
from local_lm.db import Base
from local_lm.domain import JobKind, JobStatus, MessageRole, MessageStatus, RunStatus, utcnow
from local_lm.models import (
    Chat,
    Job,
    Message,
    ModelAssetInstall,
    ModelInstall,
    Run,
    WorkflowActivation,
    WorkflowDefinition,
    WorkflowDependencyBinding,
    WorkflowDependencySlot,
    WorkflowRevision,
)
from local_lm.orchestrator import (
    WORKFLOW_LORA_REPLAY_FAILURE,
    ConversationOrchestrator,
)
from local_lm.workflow_bindings import (
    ResolvedWorkflowBinding,
    workflow_activation_binding_sha256,
    workflow_resource_identity_sha256,
)
from local_lm.workflow_dependencies import (
    WorkflowDependencyContract,
    WorkflowDependencyRequirement,
    WorkflowDependencySlotContract,
    workflow_dependency_contract_sha256,
    workflow_dependency_slot_payload,
    workflow_dependency_slot_sha256,
)
from local_lm.workflow_lora_admission import WorkflowLoraAdmissionError
from local_lm.workflow_lora_composition import compose_workflow_lora_graph
from local_lm.workflow_lora_execution import (
    MAX_WORKFLOW_LORA_REPLAY_JSON_DEPTH,
    MAX_WORKFLOW_LORA_REPLAY_JSON_NODES,
    build_workflow_lora_execution_authority,
    parse_workflow_lora_replay_receipt,
    workflow_lora_replay_payload,
)
from local_lm.workflow_lora_graph import WorkflowLoraGraphError
from local_lm.workflow_lora_overrides import (
    WorkflowLoraOverrideCatalog,
    WorkflowLoraOverrideLayer,
    parse_workflow_lora_overrides,
    resolve_workflow_lora_override_layers,
    workflow_lora_overrides_payload,
)
from local_lm.workflow_lora_settings import workflow_lora_override_resolution_as_overrides
from local_lm.workflow_loras import workflow_lora_controls

PRIVATE_LOADER_ID = "node-that-must-remain-private"
EMBEDDED_REFERENCE = "styles/embedded-detail.safetensors"
EMBEDDED_DIGEST = "a" * 64
ADDED_DIGEST = "b" * 64


class _ContainsGetTrap(dict[str, Any]):
    contains_calls: int
    get_calls: int

    def __init__(self, value: dict[str, Any]) -> None:
        super().__init__(value)
        self.contains_calls = 0
        self.get_calls = 0

    def __contains__(self, key: object) -> bool:
        self.contains_calls += 1
        raise AssertionError("hostile dict membership executed")

    def get(self, key: Any, default: Any = None) -> Any:
        self.get_calls += 1
        raise AssertionError("hostile dict get executed")


class _HashEqualityTrap(str):
    armed: ClassVar[bool] = False
    hash_calls: ClassVar[int] = 0
    equality_calls: ClassVar[int] = 0

    def __hash__(self) -> int:
        if type(self).armed:
            type(self).hash_calls += 1
            raise AssertionError("hostile string hash executed")
        return str.__hash__(self)

    def __eq__(self, other: object) -> bool:
        if type(self).armed:
            type(self).equality_calls += 1
            raise AssertionError("hostile string equality executed")
        result = str.__eq__(self, other)
        return False if result is NotImplemented else result

    @classmethod
    def arm(cls) -> None:
        cls.hash_calls = 0
        cls.equality_calls = 0
        cls.armed = True

    @classmethod
    def disarm(cls) -> None:
        cls.armed = False


@dataclass(frozen=True)
class SeededWorkflow:
    definition_id: str
    revision_id: str
    activation_id: str
    embedded_asset_id: str
    added_asset_id: str
    base_install_id: str


@pytest.fixture
def session_factory() -> Generator[sessionmaker[Session]]:
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    factory = sessionmaker(engine, expire_on_commit=False)
    yield factory
    engine.dispose()


def _slot(
    name: str, resource_kind: str, *, required: bool = True
) -> WorkflowDependencySlotContract:
    return WorkflowDependencySlotContract(
        name=name,
        resource_kind=cast(Any, resource_kind),
        required=required,
        satisfaction="any_of",
        requirements=(WorkflowDependencyRequirement("default", {}),),
    )


def _api_graph() -> dict[str, Any]:
    return {
        "source": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": "base.safetensors"},
        },
        PRIVATE_LOADER_ID: {
            "class_type": "LoraLoader",
            "inputs": {
                "model": ["source", 0],
                "clip": ["source", 1],
                "lora_name": EMBEDDED_REFERENCE,
                "strength_model": 0.75,
                "strength_clip": 0.5,
            },
        },
    }


def _ui_graph() -> dict[str, Any]:
    return {
        "version": 0.4,
        "nodes": [
            {
                "id": "source",
                "type": "CheckpointLoaderSimple",
                "mode": 0,
                "inputs": [],
                "outputs": [
                    {"name": "MODEL", "type": "MODEL", "links": [1]},
                    {"name": "CLIP", "type": "CLIP", "links": [2]},
                ],
                "widgets_values": ["base.safetensors"],
                "properties": {"cnr_id": "comfy-core"},
            },
            {
                "id": PRIVATE_LOADER_ID,
                "type": "LoraLoader",
                "mode": 0,
                "inputs": [
                    {"name": "model", "type": "MODEL", "link": 1},
                    {"name": "clip", "type": "CLIP", "link": 2},
                ],
                "outputs": [],
                "widgets_values": [EMBEDDED_REFERENCE, 0.75, 0.5],
                "properties": {"cnr_id": "comfy-core", "ver": "0.28.0"},
            },
        ],
        "links": [
            [1, "source", 0, PRIVATE_LOADER_ID, 0, "MODEL"],
            [2, "source", 1, PRIVATE_LOADER_ID, 1, "CLIP"],
        ],
    }


def _runtime_binding(*, build: str) -> ResolvedWorkflowBinding:
    identity = {
        "kind": "runtime",
        "engine": "comfyui",
        "runtime_build": build,
        "adapter_contract_version": 1,
        "launch_contract_version": "v1",
    }
    return ResolvedWorkflowBinding(
        slot_name="comfy-runtime",
        requirement_key="default",
        resource_kind="runtime",
        identity=identity,
        resource_identity_sha256=workflow_resource_identity_sha256("runtime", identity),
        mount={},
    )


def _asset_binding() -> ResolvedWorkflowBinding:
    identity = {
        "kind": "model_asset",
        "asset_kind": "lora",
        "runtime_reference": EMBEDDED_REFERENCE,
        "sha256": EMBEDDED_DIGEST,
    }
    return ResolvedWorkflowBinding(
        slot_name="style",
        requirement_key="default",
        resource_kind="model_asset",
        identity=identity,
        resource_identity_sha256=workflow_resource_identity_sha256("model_asset", identity),
        mount={},
    )


def _base_binding() -> ResolvedWorkflowBinding:
    identity = {
        "kind": "model_install",
        "engine": "comfyui",
        "family": "test",
    }
    return ResolvedWorkflowBinding(
        slot_name="base-model",
        requirement_key="default",
        resource_kind="model_install",
        identity=identity,
        resource_identity_sha256=workflow_resource_identity_sha256("model_install", identity),
        mount={},
    )


def _seed_workflow(session: Session, *, suffix: str = "replay") -> SeededWorkflow:
    base = ModelInstall(
        id=f"model_{suffix}",
        name="Replay base model",
        role="image",
        engine="comfyui",
        local_path="C:/private/models/base",
        manifest_json={"family": "test"},
        active=True,
    )
    definition = WorkflowDefinition(
        id=f"workflow_{suffix}",
        name=f"Replay workflow {suffix}",
        operation="text_to_image",
    )
    session.add_all([base, definition])
    session.flush()
    contract = WorkflowDependencyContract(
        version=1,
        slots=(
            _slot("comfy-runtime", "runtime"),
            _slot("style", "model_asset", required=False),
            _slot("base-model", "model_install"),
        ),
    )
    graph = _api_graph()
    extension = checkpoint_lora_extension(graph)
    assert extension is not None
    revision = WorkflowRevision(
        id=f"wfrev_{suffix}",
        workflow_id=definition.id,
        version=1,
        engine="comfyui",
        ui_graph_json=_ui_graph(),
        api_graph_json=graph,
        input_schema_json={
            "type": "object",
            "properties": {
                "loras": {"type": "array", "default": [], "maxItems": 8},
            },
        },
        dependencies_json={"extensions": {"lora": extension}},
        dependency_contract_sha256=workflow_dependency_contract_sha256(contract),
        trusted=True,
    )
    session.add(revision)
    session.flush()
    definition.current_revision_id = revision.id
    slots: dict[str, WorkflowDependencySlot] = {}
    for ordinal, contract_slot in enumerate(contract.slots):
        payload = workflow_dependency_slot_payload(contract_slot)
        row = WorkflowDependencySlot(
            id=f"wfslot_{suffix}_{ordinal}",
            workflow_revision_id=revision.id,
            name=contract_slot.name,
            resource_kind=contract_slot.resource_kind,
            required=contract_slot.required,
            satisfaction=contract_slot.satisfaction,
            requirements_json=payload["requirements"],
            contract_sha256=workflow_dependency_slot_sha256(contract_slot),
            ordinal=ordinal,
        )
        session.add(row)
        slots[row.name] = row
    embedded = ModelAssetInstall(
        id=f"asset_embedded_{suffix}",
        name="Embedded private LoRA",
        kind="lora",
        family="test",
        local_path="C:/private/models/embedded-detail.safetensors",
        size_bytes=17,
        manifest_json={"sha256": EMBEDDED_DIGEST, "comfy_name": EMBEDDED_REFERENCE},
        active=True,
        verified_at=utcnow(),
    )
    added = ModelAssetInstall(
        id=f"asset_added_{suffix}",
        name="Added replay LoRA",
        kind="lora",
        family="test",
        local_path="C:/private/models/added-detail.safetensors",
        size_bytes=19,
        manifest_json={
            "sha256": ADDED_DIGEST,
            "comfy_name": "styles/added-detail.safetensors",
        },
        active=True,
        verified_at=utcnow(),
    )
    session.add_all([embedded, added])
    session.flush()
    activation = _add_activation(
        session,
        suffix=suffix,
        revision=revision,
        contract=contract,
        slots=slots,
        embedded_asset_id=embedded.id,
        base_install_id=base.id,
        activation_suffix="recorded",
        runtime_build="ComfyUI recorded",
        launch_sha256="e" * 64,
        active=True,
    )
    session.commit()
    return SeededWorkflow(
        definition.id,
        revision.id,
        activation.id,
        embedded.id,
        added.id,
        base.id,
    )


def _add_activation(
    session: Session,
    *,
    suffix: str,
    revision: WorkflowRevision,
    contract: WorkflowDependencyContract,
    slots: dict[str, WorkflowDependencySlot],
    embedded_asset_id: str,
    base_install_id: str,
    activation_suffix: str,
    runtime_build: str,
    launch_sha256: str,
    active: bool,
) -> WorkflowActivation:
    bindings = (_runtime_binding(build=runtime_build), _asset_binding(), _base_binding())
    activation = WorkflowActivation(
        id=f"wfact_{suffix}_{activation_suffix}",
        workflow_revision_id=revision.id,
        resolver_version="workflow-activation-v1",
        dependency_contract_sha256=workflow_dependency_contract_sha256(contract),
        binding_sha256=workflow_activation_binding_sha256(contract, bindings),
        state="ready",
        is_active=active,
        details_json={"launch_sha256": launch_sha256},
    )
    session.add(activation)
    session.flush()
    for index, binding in enumerate(bindings):
        session.add(
            WorkflowDependencyBinding(
                id=f"wfbind_{suffix}_{activation_suffix}_{index}",
                workflow_revision_id=revision.id,
                workflow_activation_id=activation.id,
                workflow_dependency_slot_id=slots[binding.slot_name].id,
                requirement_key=binding.requirement_key,
                runtime_key=("comfyui-current" if binding.resource_kind == "runtime" else None),
                model_asset_install_id=(
                    embedded_asset_id if binding.resource_kind == "model_asset" else None
                ),
                model_install_id=(
                    base_install_id if binding.resource_kind == "model_install" else None
                ),
                mount_json=binding.mount,
                resource_identity_json=binding.identity,
                resource_identity_sha256=binding.resource_identity_sha256,
            )
        )
    session.flush()
    return activation


def _admission_receipt(
    session: Session,
    seeded: SeededWorkflow,
    *,
    model_strength: float = 0.25,
    added_loras: object | None = None,
) -> dict[str, object]:
    revision = session.get(WorkflowRevision, seeded.revision_id)
    assert revision is not None
    projection = workflow_lora_controls(
        session,
        revision_id=seeded.revision_id,
    )
    assert projection.override_target is not None
    slot = projection.slots[0]
    raw_target = {
        **asdict(projection.override_target),
        "overrides": [
            {
                "slot_id": slot.slot_id,
                "loader_contract": slot.loader_contract,
                "loader_authority_sha256": slot.loader_authority_sha256,
                "changes": {"model_strength": model_strength},
            }
        ],
    }
    overrides = parse_workflow_lora_overrides({"version": 1, "targets": [raw_target]})
    catalog = WorkflowLoraOverrideCatalog(projection.override_target, projection.slots)
    resolution = resolve_workflow_lora_override_layers(
        catalog=catalog,
        layers=(WorkflowLoraOverrideLayer("turn", overrides),),
    )
    authority = build_workflow_lora_execution_authority(
        session,
        revision,
        activation_id=seeded.activation_id,
    )
    composition = compose_workflow_lora_graph(
        session,
        revision,
        added_loras=[] if added_loras is None else added_loras,
        workflow_activation_id=seeded.activation_id,
        override_catalog=authority.catalog,
        slot_extraction=authority.slot_extraction,
        override_resolution=resolution,
    )
    return workflow_lora_replay_payload(composition, resolution)


def _workflow_witness(session: Session, seeded: SeededWorkflow) -> dict[str, object]:
    revision = session.get(WorkflowRevision, seeded.revision_id)
    activation = session.get(WorkflowActivation, seeded.activation_id)
    assert revision is not None and activation is not None
    return {
        "definition_id": seeded.definition_id,
        "revision_id": seeded.revision_id,
        "activation": {
            "id": activation.id,
            "resolver_version": activation.resolver_version,
            "dependency_contract_sha256": activation.dependency_contract_sha256,
            "binding_sha256": activation.binding_sha256,
            "launch_sha256": activation.details_json["launch_sha256"],
        },
    }


_ABSENT = object()


def _workflow_witness(session: Session, seeded: SeededWorkflow) -> dict[str, object]:
    revision = session.get(WorkflowRevision, seeded.revision_id)
    activation = session.get(WorkflowActivation, seeded.activation_id)
    assert revision is not None and activation is not None
    return {
        "definition_id": seeded.definition_id,
        "revision_id": seeded.revision_id,
        "activation": {
            "id": activation.id,
            "resolver_version": activation.resolver_version,
            "dependency_contract_sha256": activation.dependency_contract_sha256,
            "binding_sha256": activation.binding_sha256,
            "launch_sha256": activation.details_json["launch_sha256"],
        },
    }


def _setting_for(receipt: dict[str, object]) -> dict[str, object]:
    """Return the reserved setting admission stores beside this receipt."""

    parsed = parse_workflow_lora_replay_receipt(receipt)
    return workflow_lora_overrides_payload(
        workflow_lora_override_resolution_as_overrides(parsed.override_resolution)
    )


def _add_run(
    session: Session,
    seeded: SeededWorkflow,
    *,
    tag: str,
    receipt: object = _ABSENT,
    settings: dict[str, Any] | None = None,
    provenance_updates: dict[str, Any] | None = None,
) -> tuple[str, str]:
    chat = Chat(id=f"chat_{tag}", title="Replay dispatch")
    user = Message(
        id=f"msg_user_{tag}",
        chat_id=chat.id,
        role=MessageRole.USER.value,
        status=MessageStatus.COMPLETE.value,
    )
    assistant = Message(
        id=f"msg_assistant_{tag}",
        chat_id=chat.id,
        role=MessageRole.ASSISTANT.value,
        status=MessageStatus.PENDING.value,
    )
    provenance: dict[str, Any] = {
        "workflow": _workflow_witness(session, seeded),
        **(provenance_updates or {}),
    }
    if receipt is not _ABSENT:
        provenance["workflow_lora"] = receipt
    run = Run(
        id=f"run_{tag}",
        chat_id=chat.id,
        user_message_id=user.id,
        assistant_message_id=assistant.id,
        operation="text_to_image",
        status=RunStatus.QUEUED.value,
        standalone_prompt="A replay-safe scene",
        workflow_revision_id=seeded.revision_id,
        settings_json=settings or {},
        provenance_json=provenance,
    )
    job = Job(
        id=f"job_{tag}",
        kind=JobKind.IMAGE.value,
        status=JobStatus.QUEUED.value,
        run_id=run.id,
        queue_resource="media_compute",
        queue_group="primary",
    )
    session.add_all([chat, user, assistant, run, job])
    session.commit()
    return run.id, job.id


def _orchestrator(factory: sessionmaker[Session] | Any) -> ConversationOrchestrator:
    orchestrator = ConversationOrchestrator(
        engines=cast(
            Any,
            SimpleNamespace(
                settings=SimpleNamespace(media_engine="mock"),
                media=MockMediaAdapter(),
            ),
        ),
        artifacts=cast(Any, SimpleNamespace(resolve=lambda _artifact: None)),
        events=cast(Any, SimpleNamespace(publish=AsyncMock())),
        scheduler=cast(Any, SimpleNamespace(publish_job=AsyncMock())),
        processes=cast(Any, SimpleNamespace()),
        session_factory=factory,
    )
    orchestrator.__setattr__("_require_phase", AsyncMock())
    return orchestrator


_CLAIM = cast(Any, SimpleNamespace(token="replay-claim", attempt=1))


async def _dispatch(
    factory: sessionmaker[Session] | Any,
    monkeypatch: pytest.MonkeyPatch,
    *,
    run_id: str,
    job_id: str,
) -> list[MediaRequest]:
    requests: list[MediaRequest] = []

    async def capture(
        _adapter: MockMediaAdapter,
        request: MediaRequest,
    ) -> AsyncIterator[MediaEvent]:
        requests.append(request)
        yield MediaEvent(type="cancelled")

    monkeypatch.setattr(MockMediaAdapter, "generate", capture)
    await _orchestrator(factory)._execute_media(job_id, run_id, _CLAIM)
    return requests


def _composition_digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode()
    ).hexdigest()


def test_recorded_execution_authority_hides_private_targets_and_receipt_is_public_safe(
    session_factory: sessionmaker[Session],
) -> None:
    with session_factory() as session:
        seeded = _seed_workflow(session)
        revision = session.get(WorkflowRevision, seeded.revision_id)
        assert revision is not None
        authority = build_workflow_lora_execution_authority(
            session,
            revision,
            activation_id=seeded.activation_id,
        )
        receipt = _admission_receipt(session, seeded)

    authority_repr = repr(authority)
    encoded = json.dumps(receipt, sort_keys=True)
    assert PRIVATE_LOADER_ID not in authority_repr
    assert "C:/private" not in authority_repr
    assert PRIVATE_LOADER_ID not in encoded
    assert "C:/private" not in encoded
    assert "effective_api_graph_json" not in encoded
    parsed = parse_workflow_lora_replay_receipt(receipt)
    assert parsed.composition_sha256 == receipt["composition_sha256"]


async def test_dispatch_replays_recorded_activation_after_current_swap_and_detaches_graph(
    session_factory: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with session_factory() as session:
        seeded = _seed_workflow(session)
        receipt = _admission_receipt(session, seeded)
        recorded = session.get(WorkflowActivation, seeded.activation_id)
        revision = session.get(WorkflowRevision, seeded.revision_id)
        assert recorded is not None and revision is not None
        recorded.is_active = False
        contract_rows = sorted(revision.dependency_slots, key=lambda row: row.ordinal)
        contract = WorkflowDependencyContract(
            version=1,
            slots=tuple(
                WorkflowDependencySlotContract(
                    name=row.name,
                    resource_kind=cast(Any, row.resource_kind),
                    required=row.required,
                    satisfaction=cast(Any, row.satisfaction),
                    requirements=tuple(
                        WorkflowDependencyRequirement(item["key"], item["constraints"])
                        for item in row.requirements_json
                    ),
                )
                for row in contract_rows
            ),
        )
        _add_activation(
            session,
            suffix="replay",
            revision=revision,
            contract=contract,
            slots={row.name: row for row in contract_rows},
            embedded_asset_id=seeded.embedded_asset_id,
            base_install_id=seeded.base_install_id,
            activation_suffix="current",
            runtime_build="ComfyUI current replacement",
            launch_sha256="d" * 64,
            active=True,
        )
        run_id, job_id = _add_run(
            session,
            seeded,
            tag="recorded_swap",
            receipt=receipt,
            settings={"steps": 12, "workflow_lora_overrides": _setting_for(receipt)},
        )

    requests = await _dispatch(session_factory, monkeypatch, run_id=run_id, job_id=job_id)

    assert len(requests) == 1
    request = requests[0]
    assert request.workflow[PRIVATE_LOADER_ID]["inputs"]["strength_model"] == 0.25
    assert "workflow_lora_overrides" not in request.parameters
    request.parameters["steps"] = 99
    request.workflow[PRIVATE_LOADER_ID]["inputs"]["strength_model"] = 99
    with session_factory() as session:
        revision = session.get(WorkflowRevision, seeded.revision_id)
        run = session.get(Run, run_id)
        assert revision is not None and run is not None
        assert revision.api_graph_json[PRIVATE_LOADER_ID]["inputs"]["strength_model"] == 0.75
        assert run.settings_json["steps"] == 12


_PAIR_CELLS = [
    "both_absent",
    "setting_only",
    "receipt_only",
    "exact_pair",
    "changed_setting",
    "malformed_setting",
    "changed_receipt",
    "malformed_receipt",
]


def _pair_cell_run(session: Session, cell: str) -> tuple[SeededWorkflow, str, str]:
    seeded = _seed_workflow(session, suffix=f"pair_{_PAIR_CELLS.index(cell)}")
    receipt: Any = deepcopy(_admission_receipt(session, seeded))
    setting: Any = _setting_for(receipt)
    if cell == "changed_setting":
        target = cast(list[dict[str, Any]], setting["targets"])[0]
        target["overrides"][0]["changes"]["model_strength"] = 0.5
    elif cell == "malformed_setting":
        setting = {"version": 1, "targets": "not a list"}
    elif cell == "changed_receipt":
        receipt = deepcopy(_admission_receipt(session, seeded, model_strength=0.5))
    elif cell == "malformed_receipt":
        receipt = {"present": "but malformed"}
    settings: dict[str, Any] = {"steps": 12}
    if cell not in {"both_absent", "receipt_only"}:
        settings["workflow_lora_overrides"] = setting
    run_id, job_id = _add_run(
        session,
        seeded,
        tag=f"pair_{cell}",
        receipt=(_ABSENT if cell in {"both_absent", "setting_only"} else receipt),
        settings=settings,
    )
    return seeded, run_id, job_id


@pytest.mark.parametrize("cell", _PAIR_CELLS)
async def test_dispatch_accepts_only_no_edits_or_an_exact_setting_and_receipt_pair(
    session_factory: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
    cell: str,
) -> None:
    with session_factory() as session:
        seeded, run_id, job_id = _pair_cell_run(session, cell)

    if cell in {"both_absent", "exact_pair"}:
        requests = await _dispatch(session_factory, monkeypatch, run_id=run_id, job_id=job_id)
        assert len(requests) == 1
        expected_strength = 0.25 if cell == "exact_pair" else 0.75
        assert requests[0].workflow[PRIVATE_LOADER_ID]["inputs"]["strength_model"] == (
            expected_strength
        )
        assert "workflow_lora_overrides" not in requests[0].parameters
        return

    with pytest.raises(RuntimeError, match="^" + WORKFLOW_LORA_REPLAY_FAILURE + "$"):
        await _dispatch(session_factory, monkeypatch, run_id=run_id, job_id=job_id)


@pytest.mark.parametrize("cell", _PAIR_CELLS)
def test_retry_preflight_uses_the_same_pairing_rule_before_anything_changes(
    session_factory: sessionmaker[Session],
    cell: str,
) -> None:
    with session_factory() as session:
        _, run_id, _ = _pair_cell_run(session, cell)

    with session_factory() as session:
        run = session.get(Run, run_id)
        assert run is not None
        before = (deepcopy(run.settings_json), deepcopy(run.provenance_json), run.status)
        if cell in {"both_absent", "exact_pair"}:
            ConversationOrchestrator.preflight_workflow_lora_replay(session, run)
        else:
            with pytest.raises(WorkflowLoraAdmissionError) as refused:
                ConversationOrchestrator.preflight_workflow_lora_replay(session, run)
            assert refused.value.conflict is True
            assert refused.value.code == "workflow-lora-replay-stale"
            assert str(refused.value) == WORKFLOW_LORA_REPLAY_FAILURE
        assert (run.settings_json, run.provenance_json, run.status) == before


@pytest.mark.parametrize(
    "cell",
    [cell for cell in _PAIR_CELLS if cell not in {"both_absent", "exact_pair"}],
)
def test_activation_scope_refuses_an_unpaired_run_before_any_worker_start(
    session_factory: sessionmaker[Session],
    cell: str,
) -> None:
    with session_factory() as session:
        _, run_id, _ = _pair_cell_run(session, cell)

    orchestrator = _orchestrator(session_factory)
    with session_factory() as session:
        run = session.get(Run, run_id)
        assert run is not None
        with pytest.raises(RuntimeError, match="^" + WORKFLOW_LORA_REPLAY_FAILURE + "$"):
            orchestrator._media_activation_scope(session, run)


@pytest.mark.parametrize(
    "tamper",
    [
        "digest",
        "receipt_graph",
        "receipt_added",
        "stored_added",
        "stored_added_equivalent",
        "revision",
        "dependency",
        "binding",
        "launch",
        "source_graph",
        "graph_variant",
    ],
)
async def test_every_replay_tamper_refuses_before_media_adapter(
    session_factory: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
    tamper: str,
) -> None:
    with session_factory() as session:
        seeded = _seed_workflow(
            session,
            suffix=(
                "tamper_added_eq" if tamper == "stored_added_equivalent" else f"tamper_{tamper}"
            ),
        )
        normalized_added = [
            {
                "asset_id": seeded.added_asset_id,
                "model_strength": 1.0,
                "clip_strength": 1.0,
                "enabled": True,
            }
        ]
        receipt: Any = deepcopy(
            _admission_receipt(
                session,
                seeded,
                added_loras=(normalized_added if tamper == "stored_added_equivalent" else []),
            )
        )
        settings: dict[str, Any] = {"workflow_lora_overrides": _setting_for(receipt)}
        witness_updates: dict[str, Any] = {}
        if tamper == "digest":
            receipt["composition_sha256"] = "0" * 64
        elif tamper == "receipt_graph":
            receipt["composition"]["effective_graph_sha256"] = "0" * 64
            receipt["composition_sha256"] = _composition_digest(receipt["composition"])
        elif tamper == "receipt_added":
            receipt["composition"]["added_settings"] = [
                {
                    "asset_id": seeded.added_asset_id,
                    "model_strength": 0.5,
                    "clip_strength": 0.5,
                    "enabled": True,
                }
            ]
            receipt["composition_sha256"] = _composition_digest(receipt["composition"])
        elif tamper == "stored_added":
            settings["loras"] = [
                {
                    "asset_id": seeded.added_asset_id,
                    "model_strength": 0.5,
                    "clip_strength": 0.5,
                    "enabled": True,
                }
            ]
        elif tamper == "stored_added_equivalent":
            settings["loras"] = [{"asset_id": seeded.added_asset_id}]
        elif tamper in {"revision", "dependency", "binding", "launch"}:
            witness = _workflow_witness(session, seeded)
            if tamper == "revision":
                witness_updates["workflow"] = {**witness, "revision_id": "other"}
            else:
                activation = deepcopy(witness["activation"])
                assert isinstance(activation, dict)
                activation[
                    {
                        "dependency": "dependency_contract_sha256",
                        "binding": "binding_sha256",
                        "launch": "launch_sha256",
                    }[tamper]
                ] = "0" * 64
                witness_updates["workflow"] = {**witness, "activation": activation}
        elif tamper == "source_graph":
            revision = session.get(WorkflowRevision, seeded.revision_id)
            assert revision is not None
            changed = deepcopy(revision.api_graph_json)
            changed[PRIVATE_LOADER_ID]["inputs"]["strength_clip"] = 0.125
            revision.api_graph_json = changed
            session.commit()
        elif tamper == "graph_variant":
            revision = session.get(WorkflowRevision, seeded.revision_id)
            assert revision is not None
            changed = deepcopy(revision.api_graph_json)
            changed[PRIVATE_LOADER_ID] = {
                "class_type": "LoraLoaderModelOnly",
                "inputs": {
                    "model": ["source", 0],
                    "lora_name": EMBEDDED_REFERENCE,
                    "strength_model": 0.75,
                },
            }
            revision.api_graph_json = changed
            session.commit()
        run_id, job_id = _add_run(
            session,
            seeded,
            tag=f"dispatch_{tamper}",
            receipt=receipt,
            settings=settings,
            provenance_updates=witness_updates,
        )

    requests: list[MediaRequest] = []

    async def capture(
        _adapter: MockMediaAdapter,
        request: MediaRequest,
    ) -> AsyncIterator[MediaEvent]:
        requests.append(request)
        yield MediaEvent(type="cancelled")

    monkeypatch.setattr(MockMediaAdapter, "generate", capture)
    with pytest.raises(RuntimeError):
        await _orchestrator(session_factory)._execute_media(job_id, run_id, _CLAIM)
    assert requests == []


async def test_legacy_added_only_dispatch_is_unchanged_without_setting_or_receipt(
    session_factory: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with session_factory() as session:
        seeded = _seed_workflow(session, suffix="legacy")
        revision = session.get(WorkflowRevision, seeded.revision_id)
        assert revision is not None
        resolved = resolve_lora_stack(
            session,
            revision,
            [
                {
                    "asset_id": seeded.added_asset_id,
                    "model_strength": 0.6,
                    "clip_strength": 0.4,
                    "enabled": True,
                }
            ],
        )
        run_id, job_id = _add_run(
            session,
            seeded,
            tag="legacy_added",
            settings={"loras": resolved.settings},
            provenance_updates={
                "auxiliary_assets": {"effective_graph_sha256": resolved.graph_sha256}
            },
        )
        expected = resolve_lora_stack_against_graph(
            session,
            revision,
            resolved.settings,
            base_api_graph=revision.api_graph_json,
        ).graph

    requests = await _dispatch(session_factory, monkeypatch, run_id=run_id, job_id=job_id)

    assert len(requests) == 1
    assert requests[0].workflow == expected
    assert requests[0].parameters["loras"] == resolved.settings
    assert "workflow_lora_overrides" not in requests[0].parameters


@pytest.mark.parametrize(
    "case",
    [
        "root_dict",
        "workflow_dict",
        "activation_dict",
        "provenance_key",
        "activation_key",
        "receipt_key",
        "settings_key",
    ],
)
def test_hostile_run_json_is_refused_without_running_its_methods(
    session_factory: sessionmaker[Session],
    case: str,
) -> None:
    with session_factory() as session:
        seeded = _seed_workflow(session, suffix=f"hostile_{case[:4]}")
        receipt = _admission_receipt(session, seeded)
        run_id, _ = _add_run(
            session,
            seeded,
            tag=f"hostile_{case}",
            receipt=receipt,
            settings={"workflow_lora_overrides": _setting_for(receipt)},
        )

    traps: list[_ContainsGetTrap] = []
    with session_factory() as session:
        run = session.get(Run, run_id)
        assert run is not None
        provenance: dict[str, Any] = deepcopy(run.provenance_json)
        settings: dict[str, Any] = deepcopy(run.settings_json)
        replacement_provenance: object = provenance
        replacement_settings: object = settings
        try:
            if case == "root_dict":
                trap = _ContainsGetTrap(provenance)
                traps.append(trap)
                replacement_provenance = trap
            elif case == "workflow_dict":
                trap = _ContainsGetTrap(provenance["workflow"])
                traps.append(trap)
                provenance["workflow"] = trap
            elif case == "activation_dict":
                trap = _ContainsGetTrap(provenance["workflow"]["activation"])
                traps.append(trap)
                provenance["workflow"]["activation"] = trap
            elif case == "provenance_key":
                provenance[_HashEqualityTrap("workflow_lora")] = provenance.pop("workflow_lora")
            elif case == "activation_key":
                activation = provenance["workflow"]["activation"]
                activation[_HashEqualityTrap("id")] = activation.pop("id")
            elif case == "receipt_key":
                replay = provenance["workflow_lora"]
                replay[_HashEqualityTrap("composition")] = replay.pop("composition")
            else:
                settings[_HashEqualityTrap("workflow_lora_overrides")] = settings.pop(
                    "workflow_lora_overrides"
                )
            _HashEqualityTrap.arm()
            run.__dict__["provenance_json"] = replacement_provenance
            run.__dict__["settings_json"] = replacement_settings
            with pytest.raises(WorkflowLoraAdmissionError):
                ConversationOrchestrator.preflight_workflow_lora_replay(session, run)
            assert all(trap.contains_calls == trap.get_calls == 0 for trap in traps)
            assert _HashEqualityTrap.hash_calls == 0
            assert _HashEqualityTrap.equality_calls == 0
        finally:
            _HashEqualityTrap.disarm()


@pytest.mark.parametrize("shape", ["cycle", "shared", "depth", "nodes"])
def test_bounded_run_json_refuses_cycles_sharing_depth_and_node_exhaustion(
    session_factory: sessionmaker[Session],
    shape: str,
) -> None:
    with session_factory() as session:
        seeded = _seed_workflow(session, suffix=f"bound_{shape}")
        receipt = _admission_receipt(session, seeded)
        run_id, _ = _add_run(
            session,
            seeded,
            tag=f"bound_{shape}",
            receipt=receipt,
            settings={"workflow_lora_overrides": _setting_for(receipt)},
        )

    with session_factory() as session:
        run = session.get(Run, run_id)
        assert run is not None
        provenance = deepcopy(run.provenance_json)
        assert type(provenance) is dict
        if shape == "cycle":
            provenance["hostile"] = provenance
        elif shape == "shared":
            shared: dict[str, object] = {"value": "shared"}
            provenance["hostile_a"] = shared
            provenance["hostile_b"] = shared
        elif shape == "depth":
            nested: object = None
            for _ in range(MAX_WORKFLOW_LORA_REPLAY_JSON_DEPTH + 2):
                nested = [nested]
            provenance["hostile"] = nested
        else:
            provenance["hostile"] = [None] * MAX_WORKFLOW_LORA_REPLAY_JSON_NODES
        run.__dict__["provenance_json"] = provenance
        with pytest.raises(WorkflowLoraAdmissionError):
            ConversationOrchestrator.preflight_workflow_lora_replay(session, run)


async def test_replay_refusal_never_echoes_private_graph_detail(
    session_factory: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with session_factory() as session:
        seeded = _seed_workflow(session, suffix="private_detail")
        receipt = _admission_receipt(session, seeded)
        run_id, job_id = _add_run(
            session,
            seeded,
            tag="private_detail",
            receipt=receipt,
            settings={"workflow_lora_overrides": _setting_for(receipt)},
        )

    def graph_variant_error(*_args: object, **_kwargs: object) -> None:
        raise WorkflowLoraGraphError(
            "graph_variant_drift",
            "private graph variant detail must not escape",
        )

    monkeypatch.setattr("local_lm.orchestrator.compose_workflow_lora_graph", graph_variant_error)
    with pytest.raises(RuntimeError) as refused:
        await _dispatch(session_factory, monkeypatch, run_id=run_id, job_id=job_id)
    assert str(refused.value) == WORKFLOW_LORA_REPLAY_FAILURE
    assert "private" not in str(refused.value)

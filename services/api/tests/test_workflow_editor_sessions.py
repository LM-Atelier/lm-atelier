from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from itertools import count
from threading import Lock

import pytest

from local_lm.workflow_editor_sessions import (
    WorkflowEditorSession,
    WorkflowEditorSessionError,
    WorkflowEditorSessions,
    workflow_api_graph_sha256,
    workflow_ui_graph_sha256,
)


def _graph(
    *,
    node_type: str = "KSampler",
    filename: str = "base.safetensors",
    extra_nodes: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    nodes: list[dict[str, object]] = [
        {
            "id": 1,
            "type": node_type,
            "inputs": [],
            "outputs": [],
            "widgets_values": [filename],
        }
    ]
    nodes.extend(extra_nodes or [])
    return {"version": 0.4, "nodes": nodes, "links": []}


def _prompt(*, node_type: str = "KSampler") -> dict[str, object]:
    return {"1": {"class_type": node_type, "inputs": {}}}


class _Clock:
    def __init__(self) -> None:
        self.value = datetime(2026, 8, 3, 12, tzinfo=UTC)

    def __call__(self) -> datetime:
        return self.value


class _Tokens:
    def __init__(self) -> None:
        self._values = count(1)
        self._lock = Lock()

    def __call__(self, size: int) -> str:
        with self._lock:
            return f"token-{size}-{next(self._values)}"


def _sessions(
    *, ttl: timedelta = timedelta(minutes=5), max_active: int = 8
) -> tuple[WorkflowEditorSessions, _Clock]:
    clock = _Clock()
    return WorkflowEditorSessions(
        ttl=ttl,
        max_active=max_active,
        clock=clock,
        token_factory=_Tokens(),
    ), clock


def _start(
    manager: WorkflowEditorSessions,
    *,
    graph: dict[str, object] | None = None,
) -> WorkflowEditorSession:
    return manager.start(
        workflow_id="workflow_one",
        base_revision_id="revision_one",
        base_ui_graph=graph or _graph(),
        base_api_graph=_prompt(),
        runtime_identity="runtime_one",
    )


def test_workflow_graph_digest_is_canonical_and_rejects_non_finite_values() -> None:
    assert workflow_ui_graph_sha256(_graph()) == workflow_ui_graph_sha256(
        {"links": [], "nodes": _graph()["nodes"], "version": 0.4}
    )
    with pytest.raises(WorkflowEditorSessionError, match="non-finite number") as rejected:
        workflow_ui_graph_sha256({**_graph(), "value": float("nan")})
    assert rejected.value.code == "workflow-editor-non_finite_number"


def test_session_is_consumed_once_and_reports_unchanged_graph() -> None:
    manager, _clock = _sessions()
    session = _start(manager)

    returned = manager.consume(
        session_id=session.id,
        nonce=session.nonce,
        workflow_id=session.workflow_id,
        base_revision_id=session.base_revision_id,
        current_revision_id=session.base_revision_id,
        returned_ui_graph=_graph(),
        returned_api_graph=_prompt(),
        runtime_identity=session.runtime_identity,
    )

    assert not returned.changed
    assert not returned.forked
    assert returned.validated_return_id.startswith("wfreturn_")
    assert manager.validated_return_count == 1
    with pytest.raises(WorkflowEditorSessionError) as replay:
        manager.consume(
            session_id=session.id,
            nonce=session.nonce,
            workflow_id=session.workflow_id,
            base_revision_id=session.base_revision_id,
            current_revision_id=session.base_revision_id,
            returned_ui_graph=_graph(),
            returned_api_graph=_prompt(),
            runtime_identity=session.runtime_identity,
        )
    assert replay.value.code == "workflow-editor-session-not-found"


def test_unchanged_graph_requires_its_bound_prompt_and_preserves_retry() -> None:
    manager, _clock = _sessions()
    session = _start(manager)

    with pytest.raises(WorkflowEditorSessionError, match="does not match") as rejected:
        manager.consume(
            session_id=session.id,
            nonce=session.nonce,
            workflow_id=session.workflow_id,
            base_revision_id=session.base_revision_id,
            current_revision_id=session.base_revision_id,
            returned_ui_graph=_graph(),
            returned_api_graph=_prompt(node_type="VAEDecode"),
            runtime_identity=session.runtime_identity,
        )
    assert rejected.value.code == "workflow-editor-prompt-mismatch"

    returned = manager.consume(
        session_id=session.id,
        nonce=session.nonce,
        workflow_id=session.workflow_id,
        base_revision_id=session.base_revision_id,
        current_revision_id=session.base_revision_id,
        returned_ui_graph=_graph(),
        returned_api_graph=_prompt(),
        runtime_identity=session.runtime_identity,
    )
    assert not returned.changed


def test_invalid_prompt_is_bounded_without_consuming_the_session() -> None:
    manager, _clock = _sessions()
    session = _start(manager)

    with pytest.raises(WorkflowEditorSessionError) as rejected:
        manager.consume(
            session_id=session.id,
            nonce=session.nonce,
            workflow_id=session.workflow_id,
            base_revision_id=session.base_revision_id,
            current_revision_id=session.base_revision_id,
            returned_ui_graph=_graph(),
            returned_api_graph={"value": float("nan")},
            runtime_identity=session.runtime_identity,
        )
    assert rejected.value.code == "workflow-editor-non_finite_number"

    manager.cancel(
        session_id=session.id,
        nonce=session.nonce,
        workflow_id=session.workflow_id,
    )


def test_changed_return_is_marked_as_a_fork_when_current_revision_advanced() -> None:
    manager, _clock = _sessions()
    session = _start(manager)

    returned = manager.consume(
        session_id=session.id,
        nonce=session.nonce,
        workflow_id=session.workflow_id,
        base_revision_id=session.base_revision_id,
        current_revision_id="revision_two",
        returned_ui_graph=_graph(node_type="VAEDecode"),
        returned_api_graph=_prompt(node_type="VAEDecode"),
        runtime_identity=session.runtime_identity,
    )

    assert returned.changed
    assert returned.forked
    assert returned.current_revision_id == "revision_two"
    assert manager.validated_return_count == 1


def test_unchanged_return_does_not_claim_a_fork_when_current_revision_advanced() -> None:
    manager, _clock = _sessions()
    session = _start(manager)

    returned = manager.consume(
        session_id=session.id,
        nonce=session.nonce,
        workflow_id=session.workflow_id,
        base_revision_id=session.base_revision_id,
        current_revision_id="revision_two",
        returned_ui_graph=_graph(),
        returned_api_graph=_prompt(),
        runtime_identity=session.runtime_identity,
    )

    assert not returned.changed
    assert not returned.forked


def test_changed_return_reports_server_derived_graph_and_asset_delta() -> None:
    manager, _clock = _sessions()
    session = _start(manager)
    returned_graph = _graph(
        node_type="VAEDecode",
        filename="replacement.safetensors",
        extra_nodes=[
            {
                "id": 2,
                "type": "CLIPTextEncode",
                "inputs": [],
                "outputs": [],
                "widgets_values": ["prompt"],
            }
        ],
    )

    returned = manager.consume(
        session_id=session.id,
        nonce=session.nonce,
        workflow_id=session.workflow_id,
        base_revision_id=session.base_revision_id,
        current_revision_id=session.base_revision_id,
        returned_ui_graph=returned_graph,
        returned_api_graph=_prompt(node_type="VAEDecode"),
        runtime_identity=session.runtime_identity,
    )

    assert returned.returned_graph_sha256 == workflow_ui_graph_sha256(returned_graph)
    assert returned.base_prompt_sha256 == workflow_api_graph_sha256(_prompt())
    assert returned.returned_prompt_sha256 == workflow_api_graph_sha256(
        _prompt(node_type="VAEDecode")
    )
    assert returned.delta.node_count_delta == 1
    assert returned.delta.link_count_delta == 0
    assert returned.delta.added_node_types == ("CLIPTextEncode", "VAEDecode")
    assert returned.delta.removed_node_types == ("KSampler",)
    assert returned.delta.added_asset_filenames == ("replacement.safetensors",)
    assert returned.delta.removed_asset_filenames == ("base.safetensors",)


def test_invalid_returned_graph_does_not_consume_the_session() -> None:
    manager, _clock = _sessions()
    session = _start(manager)

    with pytest.raises(WorkflowEditorSessionError, match="contains no nodes") as rejected:
        manager.consume(
            session_id=session.id,
            nonce=session.nonce,
            workflow_id=session.workflow_id,
            base_revision_id=session.base_revision_id,
            current_revision_id=session.base_revision_id,
            returned_ui_graph={"version": 0.4, "nodes": [], "links": []},
            returned_api_graph=_prompt(),
            runtime_identity=session.runtime_identity,
        )
    assert rejected.value.code == "workflow-editor-empty_workflow"

    assert not manager.consume(
        session_id=session.id,
        nonce=session.nonce,
        workflow_id=session.workflow_id,
        base_revision_id=session.base_revision_id,
        current_revision_id=session.base_revision_id,
        returned_ui_graph=_graph(),
        returned_api_graph=_prompt(),
        runtime_identity=session.runtime_identity,
    ).changed


@pytest.mark.parametrize(
    ("field", "value", "code"),
    [
        ("nonce", "wrong", "workflow-editor-session-authentication-failed"),
        ("workflow_id", "workflow_two", "workflow-editor-session-mismatch"),
        ("base_revision_id", "revision_two", "workflow-editor-session-mismatch"),
    ],
)
def test_rejected_return_does_not_consume_the_valid_session(
    field: str, value: str, code: str
) -> None:
    manager, _clock = _sessions()
    session = _start(manager)

    with pytest.raises(WorkflowEditorSessionError) as rejected:
        manager.consume(
            session_id=session.id,
            nonce=value if field == "nonce" else session.nonce,
            workflow_id=value if field == "workflow_id" else session.workflow_id,
            base_revision_id=(value if field == "base_revision_id" else session.base_revision_id),
            current_revision_id=session.base_revision_id,
            returned_ui_graph=_graph(),
            returned_api_graph=_prompt(),
            runtime_identity=session.runtime_identity,
        )
    assert rejected.value.code == code

    assert not manager.consume(
        session_id=session.id,
        nonce=session.nonce,
        workflow_id=session.workflow_id,
        base_revision_id=session.base_revision_id,
        current_revision_id=session.base_revision_id,
        returned_ui_graph=_graph(),
        returned_api_graph=_prompt(),
        runtime_identity=session.runtime_identity,
    ).changed


def test_authorize_return_is_retryable_and_preserves_authentication_order() -> None:
    manager, _clock = _sessions()
    session = _start(manager)

    with pytest.raises(WorkflowEditorSessionError) as unauthenticated:
        manager.authorize_return(
            session_id=session.id,
            nonce="wrong",
            workflow_id="workflow_two",
            base_revision_id=session.base_revision_id,
            runtime_identity=session.runtime_identity,
        )
    assert unauthenticated.value.code == "workflow-editor-session-authentication-failed"
    with pytest.raises(WorkflowEditorSessionError) as mismatched:
        manager.authorize_return(
            session_id=session.id,
            nonce=session.nonce,
            workflow_id="workflow_two",
            base_revision_id=session.base_revision_id,
            runtime_identity=session.runtime_identity,
        )
    assert mismatched.value.code == "workflow-editor-session-mismatch"

    authorized = manager.authorize_return(
        session_id=session.id,
        nonce=session.nonce,
        workflow_id=session.workflow_id,
        base_revision_id=session.base_revision_id,
        runtime_identity=session.runtime_identity,
    )
    assert authorized.id == session.id
    assert manager.active_count == 1
    assert not manager.consume(
        session_id=session.id,
        nonce=session.nonce,
        workflow_id=session.workflow_id,
        base_revision_id=session.base_revision_id,
        current_revision_id=session.base_revision_id,
        returned_ui_graph=_graph(),
        returned_api_graph=_prompt(),
        runtime_identity=session.runtime_identity,
    ).changed


def test_expiry_purges_authority_and_releases_capacity() -> None:
    manager, clock = _sessions(ttl=timedelta(seconds=30), max_active=1)
    first = _start(manager)
    with pytest.raises(WorkflowEditorSessionError) as capacity:
        _start(manager)
    assert capacity.value.code == "workflow-editor-capacity"

    clock.value += timedelta(seconds=30)
    second = _start(manager)
    assert second.id != first.id
    with pytest.raises(WorkflowEditorSessionError) as expired:
        manager.cancel(
            session_id=first.id,
            nonce=first.nonce,
            workflow_id=first.workflow_id,
        )
    assert expired.value.code == "workflow-editor-session-not-found"


def test_new_runtime_authority_discards_sessions_from_the_previous_launch() -> None:
    manager, _clock = _sessions(max_active=1)
    first = _start(manager)

    second = manager.start(
        workflow_id="workflow_two",
        base_revision_id="revision_two",
        base_ui_graph=_graph(),
        base_api_graph=_prompt(),
        runtime_identity="runtime_two",
    )

    assert manager.active_count == 1
    assert second.runtime_identity == "runtime_two"
    with pytest.raises(WorkflowEditorSessionError) as stale:
        manager.cancel(
            session_id=first.id,
            nonce=first.nonce,
            workflow_id=first.workflow_id,
        )
    assert stale.value.code == "workflow-editor-session-not-found"


def test_validated_return_is_bounded_by_capacity_expiry_runtime_and_clear() -> None:
    manager, clock = _sessions(ttl=timedelta(seconds=30), max_active=1)
    session = _start(manager)
    manager.consume(
        session_id=session.id,
        nonce=session.nonce,
        workflow_id=session.workflow_id,
        base_revision_id=session.base_revision_id,
        current_revision_id=session.base_revision_id,
        returned_ui_graph=_graph(),
        returned_api_graph=_prompt(),
        runtime_identity=session.runtime_identity,
    )
    assert manager.active_count == 0
    assert manager.validated_return_count == 1
    with pytest.raises(WorkflowEditorSessionError) as capacity:
        _start(manager)
    assert capacity.value.code == "workflow-editor-capacity"

    clock.value += timedelta(seconds=30)
    second = _start(manager)
    assert manager.validated_return_count == 0
    manager.consume(
        session_id=second.id,
        nonce=second.nonce,
        workflow_id=second.workflow_id,
        base_revision_id=second.base_revision_id,
        current_revision_id=second.base_revision_id,
        returned_ui_graph=_graph(),
        returned_api_graph=_prompt(),
        runtime_identity=second.runtime_identity,
    )
    assert manager.validated_return_count == 1
    manager.clear()
    assert manager.validated_return_count == 0


def test_validated_return_receipt_retains_exact_canonical_graph_pair() -> None:
    manager, _clock = _sessions()
    session = _start(manager)
    ui_graph = _graph(node_type="VAEDecode", filename="edited.safetensors")
    api_graph = _prompt(node_type="VAEDecode")
    result = manager.consume(
        session_id=session.id,
        nonce=session.nonce,
        workflow_id=session.workflow_id,
        base_revision_id=session.base_revision_id,
        current_revision_id=session.base_revision_id,
        returned_ui_graph=ui_graph,
        returned_api_graph=api_graph,
        runtime_identity=session.runtime_identity,
    )

    validated = manager.validated_return(
        validated_return_id=result.validated_return_id,
        workflow_id=session.workflow_id,
        runtime_identity=session.runtime_identity,
    )
    assert json.loads(validated.returned_ui_graph_json) == ui_graph
    assert json.loads(validated.returned_api_graph_json) == api_graph
    assert validated.result == result

    with pytest.raises(WorkflowEditorSessionError) as wrong_workflow:
        manager.validated_return(
            validated_return_id=result.validated_return_id,
            workflow_id="workflow_two",
            runtime_identity=session.runtime_identity,
        )
    assert wrong_workflow.value.code == "workflow-editor-validated-return-mismatch"
    with pytest.raises(WorkflowEditorSessionError) as wrong_runtime:
        manager.validated_return(
            validated_return_id=result.validated_return_id,
            workflow_id=session.workflow_id,
            runtime_identity="runtime_two",
        )
    assert wrong_runtime.value.code == "workflow-editor-runtime-changed"


def test_runtime_change_refuses_return_without_consuming_the_session() -> None:
    manager, _clock = _sessions()
    session = _start(manager)

    with pytest.raises(WorkflowEditorSessionError) as changed:
        manager.consume(
            session_id=session.id,
            nonce=session.nonce,
            workflow_id=session.workflow_id,
            base_revision_id=session.base_revision_id,
            current_revision_id=session.base_revision_id,
            returned_ui_graph=_graph(),
            returned_api_graph=_prompt(),
            runtime_identity="runtime_two",
        )
    assert changed.value.code == "workflow-editor-runtime-changed"

    manager.cancel(
        session_id=session.id,
        nonce=session.nonce,
        workflow_id=session.workflow_id,
    )


def test_clear_invalidates_every_session() -> None:
    manager, _clock = _sessions()
    session = _start(manager)

    manager.clear()

    assert manager.active_count == 0
    with pytest.raises(WorkflowEditorSessionError) as missing:
        manager.cancel(
            session_id=session.id,
            nonce=session.nonce,
            workflow_id=session.workflow_id,
        )
    assert missing.value.code == "workflow-editor-session-not-found"


def test_cancel_requires_the_nonce_and_is_one_use() -> None:
    manager, _clock = _sessions()
    session = _start(manager)
    with pytest.raises(WorkflowEditorSessionError) as rejected:
        manager.cancel(
            session_id=session.id,
            nonce="wrong",
            workflow_id=session.workflow_id,
        )
    assert rejected.value.code == "workflow-editor-session-authentication-failed"

    manager.cancel(
        session_id=session.id,
        nonce=session.nonce,
        workflow_id=session.workflow_id,
    )
    with pytest.raises(WorkflowEditorSessionError) as replay:
        manager.cancel(
            session_id=session.id,
            nonce=session.nonce,
            workflow_id=session.workflow_id,
        )
    assert replay.value.code == "workflow-editor-session-not-found"


def test_cancel_authenticates_before_revealing_a_workflow_mismatch() -> None:
    manager, _clock = _sessions()
    session = _start(manager)

    with pytest.raises(WorkflowEditorSessionError) as unauthenticated:
        manager.cancel(
            session_id=session.id,
            nonce="wrong",
            workflow_id="workflow_two",
        )
    assert unauthenticated.value.code == "workflow-editor-session-authentication-failed"

    with pytest.raises(WorkflowEditorSessionError) as mismatched:
        manager.cancel(
            session_id=session.id,
            nonce=session.nonce,
            workflow_id="workflow_two",
        )
    assert mismatched.value.code == "workflow-editor-session-mismatch"

    manager.cancel(
        session_id=session.id,
        nonce=session.nonce,
        workflow_id=session.workflow_id,
    )


@pytest.mark.parametrize("nonce", ["", "has whitespace", "line\\nbreak", "x" * 201])
def test_invalid_nonce_is_rejected_without_consuming_the_session(nonce: str) -> None:
    manager, _clock = _sessions()
    session = _start(manager)

    with pytest.raises(WorkflowEditorSessionError) as rejected:
        manager.cancel(
            session_id=session.id,
            nonce=nonce,
            workflow_id=session.workflow_id,
        )
    assert rejected.value.code == "invalid-workflow-editor-session"

    manager.cancel(
        session_id=session.id,
        nonce=session.nonce,
        workflow_id=session.workflow_id,
    )


def test_concurrent_returns_allow_exactly_one_consumer() -> None:
    manager, _clock = _sessions()
    session = _start(manager)

    def consume() -> str:
        try:
            manager.consume(
                session_id=session.id,
                nonce=session.nonce,
                workflow_id=session.workflow_id,
                base_revision_id=session.base_revision_id,
                current_revision_id=session.base_revision_id,
                returned_ui_graph=_graph(node_type="VAEDecode"),
                returned_api_graph=_prompt(node_type="VAEDecode"),
                runtime_identity=session.runtime_identity,
            )
        except WorkflowEditorSessionError as exc:
            return exc.code
        return "consumed"

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(consume) for _ in range(2)]
        results = sorted(future.result() for future in futures)

    assert results == ["consumed", "workflow-editor-session-not-found"]
    assert manager.validated_return_count == 1


def test_concurrent_consume_and_cancel_allow_one_terminal_winner() -> None:
    manager, _clock = _sessions()
    session = _start(manager)

    def consume() -> str:
        try:
            manager.consume(
                session_id=session.id,
                nonce=session.nonce,
                workflow_id=session.workflow_id,
                base_revision_id=session.base_revision_id,
                current_revision_id=session.base_revision_id,
                returned_ui_graph=_graph(),
                returned_api_graph=_prompt(),
                runtime_identity=session.runtime_identity,
            )
        except WorkflowEditorSessionError as exc:
            return exc.code
        return "consumed"

    def cancel() -> str:
        try:
            manager.cancel(
                session_id=session.id,
                nonce=session.nonce,
                workflow_id=session.workflow_id,
            )
        except WorkflowEditorSessionError as exc:
            return exc.code
        return "cancelled"

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(consume), pool.submit(cancel)]
        results = sorted(future.result() for future in futures)

    assert "workflow-editor-session-not-found" in results
    assert len({"consumed", "cancelled"} & set(results)) == 1
    assert manager.validated_return_count == (1 if "consumed" in results else 0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"ttl": timedelta(0)},
        {"max_active": 0},
        {"clock": lambda: datetime(2026, 8, 3, 12)},
    ],
)
def test_invalid_manager_or_clock_configuration_is_rejected(kwargs: dict[str, object]) -> None:
    if "clock" in kwargs:
        manager = WorkflowEditorSessions(token_factory=_Tokens(), **kwargs)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="aware datetime"):
            _start(manager)
        return
    with pytest.raises(ValueError):
        WorkflowEditorSessions(**kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "graph",
    [
        {"version": 0.4, "nodes": [], "links": []},
        {"version": 0.3, "nodes": _graph()["nodes"], "links": []},
        {**_graph(), "invalid": "x" * 65_537},
    ],
)
def test_invalid_or_oversized_graphs_are_rejected(graph: dict[str, object]) -> None:
    manager, _clock = _sessions()
    with pytest.raises(WorkflowEditorSessionError) as rejected:
        _start(manager, graph=graph)
    assert rejected.value.code.startswith("workflow-editor-")

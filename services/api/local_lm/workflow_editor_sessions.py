from __future__ import annotations

import hashlib
import hmac
import secrets
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from .comfy_workflow_packages import (
    WorkflowPackageError,
    analyze_comfyui_workflow_package,
    validate_bounded_workflow_json,
)
from .workflow_trust import canonical_graph

WORKFLOW_EDITOR_PROTOCOL_VERSION = 1
DEFAULT_EDITOR_SESSION_TTL = timedelta(minutes=5)
DEFAULT_MAX_EDITOR_SESSIONS = 128
_OPAQUE_TOKEN_CHARACTERS = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-."
)
_IDENTIFIER_CHARACTERS = _OPAQUE_TOKEN_CHARACTERS | {":"}

Clock = Callable[[], datetime]
TokenFactory = Callable[[int], str]


class WorkflowEditorSessionError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True, slots=True)
class WorkflowEditorGraphSummary:
    sha256: str
    node_count: int
    link_count: int
    required_node_types: tuple[str, ...]
    asset_filenames: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class WorkflowEditorGraphDelta:
    node_count_delta: int
    link_count_delta: int
    added_node_types: tuple[str, ...]
    removed_node_types: tuple[str, ...]
    added_asset_filenames: tuple[str, ...]
    removed_asset_filenames: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class WorkflowEditorSession:
    id: str
    workflow_id: str
    base_revision_id: str
    base_graph: WorkflowEditorGraphSummary
    base_prompt_sha256: str
    runtime_identity: str
    nonce: str
    created_at: datetime
    expires_at: datetime
    protocol_version: int = WORKFLOW_EDITOR_PROTOCOL_VERSION

    @property
    def base_graph_sha256(self) -> str:
        return self.base_graph.sha256


@dataclass(frozen=True, slots=True)
class WorkflowEditorReturn:
    session_id: str
    workflow_id: str
    base_revision_id: str
    current_revision_id: str
    base_graph_sha256: str
    returned_graph_sha256: str
    base_prompt_sha256: str
    returned_prompt_sha256: str
    changed: bool
    forked: bool
    delta: WorkflowEditorGraphDelta


def workflow_ui_graph_sha256(graph: Mapping[str, Any]) -> str:
    return _workflow_ui_graph_summary(graph).sha256


def workflow_api_graph_sha256(graph: Mapping[str, Any]) -> str:
    try:
        validate_bounded_workflow_json(graph)
        canonical = canonical_graph(graph)
    except WorkflowPackageError as exc:
        raise WorkflowEditorSessionError(f"workflow-editor-{exc.code}", str(exc)) from exc
    except (RecursionError, TypeError, ValueError) as exc:
        raise WorkflowEditorSessionError(
            "workflow-editor-invalid-prompt",
            "The workflow editor prompt is not valid JSON",
        ) from exc
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class WorkflowEditorSessions:
    """Keep short-lived native-editor authority in memory and consume it once."""

    def __init__(
        self,
        *,
        ttl: timedelta = DEFAULT_EDITOR_SESSION_TTL,
        max_active: int = DEFAULT_MAX_EDITOR_SESSIONS,
        clock: Clock | None = None,
        token_factory: TokenFactory | None = None,
    ) -> None:
        if ttl <= timedelta(0):
            raise ValueError("workflow editor session TTL must be positive")
        if max_active < 1:
            raise ValueError("workflow editor session capacity must be positive")
        self._ttl = ttl
        self._max_active = max_active
        self._clock = clock or (lambda: datetime.now(UTC))
        self._token_factory = token_factory or secrets.token_urlsafe
        self._sessions: dict[str, WorkflowEditorSession] = {}
        self._lock = threading.Lock()

    def start(
        self,
        *,
        workflow_id: str,
        base_revision_id: str,
        base_ui_graph: Mapping[str, Any],
        base_api_graph: Mapping[str, Any],
        runtime_identity: str,
    ) -> WorkflowEditorSession:
        workflow_id = _identifier(workflow_id, "workflow")
        base_revision_id = _identifier(base_revision_id, "workflow revision")
        base_graph = _workflow_ui_graph_summary(base_ui_graph)
        base_prompt_sha256 = workflow_api_graph_sha256(base_api_graph)
        runtime_identity = _identifier(runtime_identity, "workflow editor runtime")
        now = _aware_utc(self._clock())
        with self._lock:
            self._purge_expired(now)
            self._purge_other_runtimes(runtime_identity)
            if len(self._sessions) >= self._max_active:
                raise WorkflowEditorSessionError(
                    "workflow-editor-capacity",
                    "Too many workflow editor sessions are already open",
                )
            session_id = self._new_session_id()
            session = WorkflowEditorSession(
                id=session_id,
                workflow_id=workflow_id,
                base_revision_id=base_revision_id,
                base_graph=base_graph,
                base_prompt_sha256=base_prompt_sha256,
                runtime_identity=runtime_identity,
                nonce=_nonce(self._token_factory(32)),
                created_at=now,
                expires_at=now + self._ttl,
            )
            self._sessions[session.id] = session
            return session

    def consume(
        self,
        *,
        session_id: str,
        nonce: str,
        workflow_id: str,
        base_revision_id: str,
        current_revision_id: str,
        returned_ui_graph: Mapping[str, Any],
        returned_api_graph: Mapping[str, Any],
        runtime_identity: str,
    ) -> WorkflowEditorReturn:
        session_id = _identifier(session_id, "workflow editor session")
        nonce = _nonce(nonce)
        workflow_id = _identifier(workflow_id, "workflow")
        base_revision_id = _identifier(base_revision_id, "workflow revision")
        current_revision_id = _identifier(current_revision_id, "current workflow revision")
        returned_graph = _workflow_ui_graph_summary(returned_ui_graph)
        returned_prompt_sha256 = workflow_api_graph_sha256(returned_api_graph)
        runtime_identity = _identifier(runtime_identity, "workflow editor runtime")
        now = _aware_utc(self._clock())
        with self._lock:
            self._purge_expired(now)
            session = self._sessions.get(session_id)
            if session is None:
                raise WorkflowEditorSessionError(
                    "workflow-editor-session-not-found",
                    "The workflow editor session is unavailable or expired",
                )
            if not hmac.compare_digest(session.nonce, nonce):
                raise WorkflowEditorSessionError(
                    "workflow-editor-session-authentication-failed",
                    "The workflow editor session could not be authenticated",
                )
            if session.workflow_id != workflow_id or session.base_revision_id != base_revision_id:
                raise WorkflowEditorSessionError(
                    "workflow-editor-session-mismatch",
                    "The workflow editor session does not match this workflow revision",
                )
            if session.runtime_identity != runtime_identity:
                raise WorkflowEditorSessionError(
                    "workflow-editor-runtime-changed",
                    "The media runtime changed while the workflow editor was open",
                )
            changed = not hmac.compare_digest(
                session.base_graph_sha256,
                returned_graph.sha256,
            )
            if not changed and not hmac.compare_digest(
                session.base_prompt_sha256,
                returned_prompt_sha256,
            ):
                raise WorkflowEditorSessionError(
                    "workflow-editor-prompt-mismatch",
                    "The workflow editor returned a prompt that does not match the graph",
                )
            del self._sessions[session_id]

        return WorkflowEditorReturn(
            session_id=session.id,
            workflow_id=session.workflow_id,
            base_revision_id=session.base_revision_id,
            current_revision_id=current_revision_id,
            base_graph_sha256=session.base_graph_sha256,
            returned_graph_sha256=returned_graph.sha256,
            base_prompt_sha256=session.base_prompt_sha256,
            returned_prompt_sha256=returned_prompt_sha256,
            changed=changed,
            forked=changed and current_revision_id != session.base_revision_id,
            delta=_graph_delta(session.base_graph, returned_graph),
        )

    def cancel(self, *, session_id: str, nonce: str, workflow_id: str) -> None:
        session_id = _identifier(session_id, "workflow editor session")
        nonce = _nonce(nonce)
        workflow_id = _identifier(workflow_id, "workflow")
        now = _aware_utc(self._clock())
        with self._lock:
            self._purge_expired(now)
            session = self._sessions.get(session_id)
            if session is None:
                raise WorkflowEditorSessionError(
                    "workflow-editor-session-not-found",
                    "The workflow editor session is unavailable or expired",
                )
            if not hmac.compare_digest(session.nonce, nonce):
                raise WorkflowEditorSessionError(
                    "workflow-editor-session-authentication-failed",
                    "The workflow editor session could not be authenticated",
                )
            if session.workflow_id != workflow_id:
                raise WorkflowEditorSessionError(
                    "workflow-editor-session-mismatch",
                    "The workflow editor session does not match this workflow",
                )
            del self._sessions[session_id]

    def clear(self) -> None:
        """Invalidate every outstanding authority during application shutdown."""

        with self._lock:
            self._sessions.clear()

    @property
    def active_count(self) -> int:
        now = _aware_utc(self._clock())
        with self._lock:
            self._purge_expired(now)
            return len(self._sessions)

    def _new_session_id(self) -> str:
        for _attempt in range(8):
            candidate = f"wfedit_{self._token_factory(24)}"
            candidate = _identifier(candidate, "workflow editor session")
            if candidate not in self._sessions:
                return candidate
        raise WorkflowEditorSessionError(
            "workflow-editor-session-id-collision",
            "A unique workflow editor session could not be created",
        )

    def _purge_expired(self, now: datetime) -> None:
        expired = [
            session_id
            for session_id, session in self._sessions.items()
            if session.expires_at <= now
        ]
        for session_id in expired:
            del self._sessions[session_id]

    def _purge_other_runtimes(self, runtime_identity: str) -> None:
        stale = [
            session_id
            for session_id, session in self._sessions.items()
            if session.runtime_identity != runtime_identity
        ]
        for session_id in stale:
            del self._sessions[session_id]


def _identifier(value: str, label: str) -> str:
    if (
        not value
        or len(value) > 200
        or any(character not in _IDENTIFIER_CHARACTERS for character in value)
    ):
        raise WorkflowEditorSessionError(
            "invalid-workflow-editor-session",
            f"The {label} identifier is invalid",
        )
    return value


def _nonce(value: str) -> str:
    if (
        not value
        or len(value) > 200
        or any(character not in _OPAQUE_TOKEN_CHARACTERS for character in value)
    ):
        raise WorkflowEditorSessionError(
            "invalid-workflow-editor-session",
            "The workflow editor session nonce is invalid",
        )
    return value


def _workflow_ui_graph_summary(graph: Mapping[str, Any]) -> WorkflowEditorGraphSummary:
    try:
        analysis = analyze_comfyui_workflow_package(graph)
    except WorkflowPackageError as exc:
        raise WorkflowEditorSessionError(f"workflow-editor-{exc.code}", str(exc)) from exc
    return WorkflowEditorGraphSummary(
        sha256=hashlib.sha256(canonical_graph(graph).encode("utf-8")).hexdigest(),
        node_count=analysis.node_count,
        link_count=analysis.link_count,
        required_node_types=analysis.required_node_types,
        asset_filenames=tuple(
            sorted(
                {reference.filename for reference in analysis.asset_references}, key=str.casefold
            )
        ),
    )


def _graph_delta(
    base: WorkflowEditorGraphSummary,
    returned: WorkflowEditorGraphSummary,
) -> WorkflowEditorGraphDelta:
    base_nodes = set(base.required_node_types)
    returned_nodes = set(returned.required_node_types)
    base_assets = set(base.asset_filenames)
    returned_assets = set(returned.asset_filenames)
    return WorkflowEditorGraphDelta(
        node_count_delta=returned.node_count - base.node_count,
        link_count_delta=returned.link_count - base.link_count,
        added_node_types=tuple(sorted(returned_nodes - base_nodes, key=str.casefold)),
        removed_node_types=tuple(sorted(base_nodes - returned_nodes, key=str.casefold)),
        added_asset_filenames=tuple(sorted(returned_assets - base_assets, key=str.casefold)),
        removed_asset_filenames=tuple(sorted(base_assets - returned_assets, key=str.casefold)),
    )


def _aware_utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("workflow editor session clock must return an aware datetime")
    return value.astimezone(UTC)

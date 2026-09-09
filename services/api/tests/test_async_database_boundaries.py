from __future__ import annotations

import ast
from pathlib import Path

SOURCE_ROOT = Path(__file__).parents[1] / "local_lm"

# Each audited exception commits its current transaction immediately before
# entering the awaited helper. Keep this list narrow: a new entry requires a
# concurrency regression proving another writer can make progress.
AUDITED_AWAITS = {
    ("orchestrator.py", "_execute_chat", "self._prepare_chat_context"),
    ("orchestrator.py", "_execute_media", "self.artifacts.browser_video_proxy"),
    ("orchestrator.py", "_execute_media", "self.artifacts.video_poster"),
    # The preparation lifecycle owns commit/rollback on a session that must
    # span its downloads and assembly. The session opens fresh with no prior
    # writes, so it enters the await holding no SQLite lock; the concurrency
    # regression in test_workflow_package_preparation.py proves another
    # writer makes progress mid-preparation.
    (
        "workflow_package_preparation.py",
        "prepare_workflow_package",
        "prepare_comfy_registry_install",
    ),
    (
        "workflow_package_preparation.py",
        "prepare_workflow_package",
        "renew_comfy_registry_install_environment",
    ),
}


def _expression_name(expression: ast.expr) -> str:
    if isinstance(expression, ast.Call):
        return _expression_name(expression.func)
    if isinstance(expression, ast.Attribute):
        prefix = _expression_name(expression.value)
        return f"{prefix}.{expression.attr}" if prefix else expression.attr
    if isinstance(expression, ast.Name):
        return expression.id
    return type(expression).__name__


def _binds_database_session(statement: ast.With) -> bool:
    for item in statement.items:
        context_name = _expression_name(item.context_expr)
        if context_name.endswith(("SessionLocal", "session_factory")):
            return True
    return False


def _awaits_inside_database_sessions() -> set[tuple[str, str, str]]:
    findings: set[tuple[str, str, str]] = set()
    for path in SOURCE_ROOT.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        parents: dict[ast.AST, ast.AST] = {}
        for node in ast.walk(tree):
            for child in ast.iter_child_nodes(node):
                parents[child] = node
        for awaited in (node for node in ast.walk(tree) if isinstance(node, ast.Await)):
            function_name = ""
            nested_in_session = False
            parent = parents.get(awaited)
            while parent is not None:
                if isinstance(parent, (ast.AsyncFunctionDef, ast.FunctionDef)):
                    function_name = parent.name
                    break
                if isinstance(parent, ast.With) and _binds_database_session(parent):
                    nested_in_session = True
                parent = parents.get(parent)
            if nested_in_session:
                findings.add(
                    (
                        path.relative_to(SOURCE_ROOT).as_posix(),
                        function_name,
                        _expression_name(awaited.value),
                    )
                )
    return findings


def test_async_database_session_boundaries_remain_explicit() -> None:
    assert _awaits_inside_database_sessions() == AUDITED_AWAITS


# The regenerate path writes a baseline ResponseRevision whose flush takes SQLite's
# single writer lock, and holds it until the turn commits. Everything between those
# two points runs with every other writer blocked, and db.py sets busy_timeout to
# five seconds, so an await there is not a latency problem but a correctness one:
# the scheduler heartbeat's write raises OperationalError, the claim stops being
# renewed, and an unrelated running generation is interrupted.
#
# The audit above cannot see this. It recognises a session only from a `with`
# statement whose callee ends in SessionLocal or session_factory, and this session
# arrives as a parameter. Widening it was measured on 2026-09-03 and rejected: the
# receives-a-session rule matches 172 sites. So this pins the one function instead.
WRITER_LOCK_SPANS = {
    (
        "orchestrator.py",
        "_create_new_turn",
        "self._ensure_response_revision",
    ),
}


def _awaits_between_write_and_commit(path: Path, function_name: str, write_call: str) -> list[int]:
    """Line numbers of awaits between the named write and the next commit.

    Both endpoints are located by walking the function body in source order, so a
    call moved to a different block is still found; only its position matters.
    """

    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    function = next(
        (
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.AsyncFunctionDef | ast.FunctionDef)
            and node.name == function_name
        ),
        None,
    )
    assert function is not None, f"{function_name} is missing from {path.name}"

    write_line = next(
        (
            node.lineno
            for node in ast.walk(function)
            if isinstance(node, ast.Call) and _expression_name(node) == write_call
        ),
        None,
    )
    assert write_line is not None, f"{write_call} is missing from {function_name}"

    commit_line = next(
        (
            node.lineno
            for node in ast.walk(function)
            if isinstance(node, ast.Call)
            and node.lineno > write_line
            and _expression_name(node).endswith("session.commit")
        ),
        None,
    )
    assert commit_line is not None, f"{function_name} never commits after {write_call}"

    return sorted(
        node.lineno
        for node in ast.walk(function)
        if isinstance(node, ast.Await) and write_line < node.lineno < commit_line
    )


def test_no_await_separates_a_writer_lock_from_its_commit() -> None:
    """A write and its commit must not be separated by an await.

    Regression for the regenerate path, where the baseline revision insert sat above
    the planner, the visual-prompt compile and the engine probe - roughly twenty-two
    seconds of held writer lock against a five-second busy timeout.
    """

    for module, function_name, write_call in sorted(WRITER_LOCK_SPANS):
        awaits = _awaits_between_write_and_commit(SOURCE_ROOT / module, function_name, write_call)
        assert not awaits, (
            f"{module}:{function_name} awaits at {awaits} while holding the writer "
            f"lock taken by {write_call}; move the write below every await, or commit "
            "before the await and record why the partial state is safe to keep"
        )

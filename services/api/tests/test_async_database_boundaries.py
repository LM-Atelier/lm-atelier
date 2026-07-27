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

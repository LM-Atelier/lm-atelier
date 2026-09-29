"""Send the actual browser's saved-edit serialization through real HTTP admission."""

from __future__ import annotations

import asyncio
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from httpx2 import AsyncClient
from test_source_fit_acceptance import prepared_turn
from test_workflow_revision_review import reviewed_runtime as reviewed_runtime

from local_lm.accepted_turn_context import accepted_context
from local_lm.db import SessionLocal
from local_lm.models import Run

ROOT = Path(__file__).resolve().parents[3]

_SERIALIZE = """
import fs from "node:fs";
import { pathToFileURL } from "node:url";
const [web, input, vite, mode] = process.argv.slice(2);
const { createServer } = await import(pathToFileURL(vite).href);
const server = await createServer({
  configFile: false, root: web, server: { middlewareMode: true },
  optimizeDeps: { noDiscovery: true, include: [] },
});
try {
  const module = await server.ssrLoadModule("/src/priorTurnEditDraft.ts");
  const source = JSON.parse(fs.readFileSync(input, "utf8"));
  const draft = module.initializePriorTurnEditDraft(source);
  if (mode !== "inherit") draft.editor.mode = mode;
  const request = module.buildPriorTurnEditRequest(draft);
  console.log("SOURCE_CANVAS_WIRE:" + JSON.stringify(request));
} finally {
  await server.close();
}
"""


async def browser_request(source: dict[str, Any], mode: str, tmp_path: Path) -> dict[str, Any]:
    node = shutil.which("node")
    vite = ROOT / "node_modules/vite/dist/node/index.js"
    if node is None or not vite.is_file():
        pytest.skip("Browser serialization integration requires the repository npm dependencies.")
    script = tmp_path / "source-canvas-serialize.mjs"
    payload = tmp_path / "source-canvas-input.json"
    script.write_text(_SERIALIZE, encoding="utf-8")
    payload.write_text(json.dumps(source), encoding="utf-8")
    completed = await asyncio.to_thread(
        subprocess.run,
        [node, str(script), str(ROOT / "apps/web"), str(payload), str(vite), mode],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
    messages = [
        line.removeprefix("SOURCE_CANVAS_WIRE:")
        for line in completed.stdout.splitlines()
        if line.startswith("SOURCE_CANVAS_WIRE:")
    ]
    assert len(messages) == 1
    result = json.loads(messages[0])
    assert isinstance(result, dict)
    return result


@pytest.mark.parametrize("mode", ["image", "auto"])
async def test_browser_source_canvas_edit_is_accepted_by_the_actual_api(
    app: FastAPI,
    client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    mode: str,
) -> None:
    chat_id, request = await prepared_turn(app, client, monkeypatch)
    fit = {"mode": "extend", "width": 4, "height": 5}
    async with app.state.services.scheduler.lease("primary"):
        created = await client.post(
            f"/api/chats/{chat_id}/turns", json={**request, "source_fit": fit}
        )
        assert created.status_code == 202, created.text
        message_id = created.json()["user_message"]["id"]
        source = await client.get(f"/api/messages/{message_id}/edit-source")
        assert source.status_code == 200, source.text
        payload = await browser_request(source.json(), mode, tmp_path)
        # Confirmation belongs to the existing confirmation flow; it changes no
        # selected source, workflow, canvas or serializer-produced field.
        payload["confirm_media"] = True
        edited = await client.post(f"/api/messages/{message_id}/edits", json=payload)
        assert edited.status_code == 202, edited.text
        replay = await client.post(f"/api/messages/{message_id}/edits", json=payload)
        assert replay.status_code == 202, replay.text
        assert replay.json()["run"]["id"] == edited.json()["run"]["id"]
        with SessionLocal() as session:
            run = session.get(Run, edited.json()["run"]["id"])
            assert run is not None
            context = accepted_context(session, run)
            assert context is not None and context.source_fit is not None
            assert context.source_fit.image.source_artifact_id == request["input_artifact_ids"][0]
            assert context.workflow_revision_id == request["workflow_revision_id"]
            assert context.profile_id == request["profile_id"]
            assert (context.source_fit.canvas_width, context.source_fit.canvas_height) == (4, 5)
        if mode == "auto":
            next_message = edited.json()["user_message"]["id"]
            restored = await client.get(f"/api/messages/{next_message}/edit-source")
            assert restored.status_code == 200, restored.text
            assert restored.json()["original_mode"] == "auto"
            assert restored.json()["source_fit"] == fit
            next_request = await browser_request(restored.json(), "inherit", tmp_path)
            assert next_request["mode"] == "auto"
            assert next_request["source_fit"] == fit

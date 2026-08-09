from __future__ import annotations

import hmac
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from html import escape
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from local_lm.comfy_editor_bridge import stage_comfy_editor_bridge

APP_ORIGIN_ENV = "LM_ATELIER_E2E_APP_ORIGIN"
ATTACKER_ORIGIN_ENV = "LM_ATELIER_E2E_ATTACKER_ORIGIN"
FIXTURE_ROOT_ENV = "LM_ATELIER_E2E_FIXTURE_ROOT"
READY_TOKEN_ENV = "LM_ATELIER_E2E_COMFY_READY_TOKEN"

_attacker_origin: str | None = None
_bridge_root: Path | None = None


def _required_environment(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(
            f"{name} is required for the synthetic browser protocol fixture"
        )
    return value


@asynccontextmanager
async def _lifespan(_app: FastAPI) -> AsyncIterator[None]:
    global _attacker_origin, _bridge_root
    fixture_root = Path(_required_environment(FIXTURE_ROOT_ENV)).resolve()
    _attacker_origin = _required_environment(ATTACKER_ORIGIN_ENV)
    _bridge_root = stage_comfy_editor_bridge(
        fixture_root / "custom_nodes",
        coordinator_origins=(_required_environment(APP_ORIGIN_ENV),),
    )
    try:
        yield
    finally:
        _attacker_origin = None
        _bridge_root = None


app = FastAPI(lifespan=_lifespan, docs_url=None, redoc_url=None, openapi_url=None)


@app.get("/ready/{token}")
async def ready(token: str) -> dict[str, str]:
    expected = _required_environment(READY_TOKEN_ENV)
    if not hmac.compare_digest(token, expected):
        raise HTTPException(404, "synthetic browser protocol fixture not found")
    return {"token": expected}


@app.get("/object_info")
async def object_info() -> dict[str, object]:
    return {
        "Source": {
            "display_name": "Source image",
            "input": {
                "required": {
                    "label": ["STRING", {"default": "camera"}],
                    "seed": ["INT", {"default": 1, "control_after_generate": True}],
                }
            },
            "input_order": {"required": ["label", "seed"]},
            "output": ["IMAGE"],
        }
    }


@app.get("/")
async def editor() -> HTMLResponse:
    if _bridge_root is None or _attacker_origin is None:
        raise HTTPException(503, "synthetic browser protocol fixture is not ready")
    bridge_name = escape(_bridge_root.name, quote=True)
    attacker_source = escape(f"{_attacker_origin}/", quote=True)
    return HTMLResponse(
        "\n".join(
            (
                "<!doctype html>",
                '<html lang="en">',
                "<head>",
                '  <meta charset="utf-8">',
                "  <title>Synthetic browser protocol editor</title>",
                "</head>",
                "<body>",
                (
                    '  <main id="synthetic-comfy-editor" '
                    'aria-label="Synthetic browser protocol editor">'
                ),
                "    <h1>Synthetic browser protocol editor</h1>",
                '    <div id="workflow-controls"></div>',
                '    <div id="action-bar" aria-label="Workflow actions"></div>',
                (
                    '    <iframe title="Hostile protocol probe" '
                    f'src="{attacker_source}"></iframe>'
                ),
                "  </main>",
                (
                    '  <script type="module" src="/extensions/'
                    f'{bridge_name}/lm_atelier_workflow_editor.js"></script>'
                ),
                "</body>",
                "</html>",
            )
        ),
        headers={"Cache-Control": "no-store"},
    )


@app.get("/scripts/app.js")
async def comfy_app_module() -> Response:
    return Response(
        content=_COMFY_APP_MODULE,
        media_type="application/javascript",
        headers={"Cache-Control": "no-store"},
    )


@app.get("/extensions/{bridge_name}/{asset_name}")
async def bridge_asset(bridge_name: str, asset_name: str) -> FileResponse:
    root = _bridge_root
    if root is None or bridge_name != root.name:
        raise HTTPException(404, "bridge fixture not found")
    if asset_name not in {
        "lm_atelier_workflow_editor.js",
        "lm_atelier_workflow_editor_config.js",
    }:
        raise HTTPException(404, "bridge asset not found")
    path = root / "js" / asset_name
    if not path.is_file():
        raise HTTPException(404, "bridge asset not found")
    return FileResponse(
        path,
        media_type="application/javascript",
        headers={"Cache-Control": "no-store"},
    )


@app.exception_handler(HTTPException)
async def http_exception(_request: Request, exc: HTTPException) -> JSONResponse:
    return JSONResponse({"detail": exc.detail}, status_code=exc.status_code)


_COMFY_APP_MODULE = """
let workflow = null;

function clone(value) {
  return structuredClone(value);
}

function sourceNode() {
  return workflow?.nodes?.find((node) => String(node.id) === "1") ?? null;
}

function renderWorkflow() {
  const controls = document.getElementById("workflow-controls");
  controls.replaceChildren();
  const node = sourceNode();
  if (!node) return;

  const label = document.createElement("label");
  label.textContent = "Source label";
  const input = document.createElement("input");
  input.setAttribute("aria-label", "Source label");
  input.value = String(node.widgets_values?.[0] ?? "");
  input.addEventListener("input", () => {
    node.widgets_values[0] = input.value;
  });
  label.append(input);
  controls.append(label);
}

export const app = {
  async loadGraphData(graph) {
    workflow = clone(graph);
    renderWorkflow();
  },

  async graphToPrompt() {
    const node = sourceNode();
    if (!workflow || !node) throw new Error("workflow not loaded");
    return {
      workflow: clone(workflow),
      output: {
        "1": {
          inputs: {
            label: String(node.widgets_values?.[0] ?? ""),
            seed: Number(node.widgets_values?.[1] ?? 1),
          },
          class_type: "Source",
          _meta: { title: "Source image" },
        },
      },
    };
  },

  registerExtension(extension) {
    const actionBar = document.getElementById("action-bar");
    for (const action of extension.actionBarButtons ?? []) {
      const button = document.createElement("button");
      button.type = "button";
      button.textContent = action.label;
      button.setAttribute("aria-label", action.label);
      button.addEventListener("click", () => action.onClick());
      actionBar.append(button);
    }
  },
};
""".strip()

"""A media engine the product starts and supervises, for browser certification.

The isolated runner used to start its ComfyUI stand-in beside the product and
point the product at it with an address. That makes the engine REACHABLE but not
MANAGED, and several product paths ask for a managed one: a workflow revision can
only be reviewed against node information fetched from a media worker the
supervisor itself launched, is running, has marked ready, and can name a process
for. An external listener satisfies none of that, so nothing in a browser run
could ever become trusted, and every surface gated on trust was untestable.

So this is not another server the runner starts. It is a program the PRODUCT
starts, through its ordinary media launch, by being pointed at an interpreter and
a directory the way a real ComfyUI installation is. The launch is unchanged; only
what sits at the end of it is synthetic.

That launch passes flags this program has no use for - a model paths file, an
output directory, a preview method, custom node switches. They are accepted and
ignored rather than rejected, because the point is to stand where the real engine
stands, and a stand-in that refuses the real command line is testing its own
argument parser.

It is copied to the staged directory under the name the product requires, and it
runs with none of the repository on its path, so it must not import from it.

NOTHING OUTSIDE THE STANDARD LIBRARY EITHER, and that is not a style preference.
The product resolves the configured interpreter with `Path.resolve(strict=True)`
before launching it. On Linux a virtual environment's `bin/python` is a SYMLINK
to the base interpreter, so resolving it silently discards the environment and
the program starts under a Python that has none of the project's packages. On
Windows the launcher is a real file, so the same code keeps the environment and
the difference never shows locally. This ran on uvicorn and passed on Windows
while failing on Ubuntu with `No module named 'uvicorn'`; the standard library
is what makes it independent of which interpreter survives that resolution.
"""

from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlsplit

# Every class a certification graph may name, each reported the way the real
# engine reports a built-in: python_module "nodes". A review grants core trust
# on that label alone and only when no installed package also claims the class,
# so this is the whole of what makes a synthetic workflow reviewable.
_CORE_MODULE = "nodes"


def _node(
    display_name: str,
    required: dict[str, Any],
    output: list[str],
    *,
    output_node: bool = False,
    category: str = "_for_testing",
) -> dict[str, Any]:
    return {
        "display_name": display_name,
        "description": "",
        "python_module": _CORE_MODULE,
        "category": category,
        "output_node": output_node,
        "input": {"required": required},
        "input_order": {"required": list(required)},
        "output": output,
        "output_is_list": [False] * len(output),
        "output_name": output,
    }


_OBJECT_INFO: dict[str, Any] = {
    "CheckpointLoaderSimple": _node(
        "Load Checkpoint",
        {"ckpt_name": [["model.safetensors"], {}]},
        ["MODEL", "CLIP", "VAE"],
        category="loaders",
    ),
    "CLIPTextEncode": _node(
        "CLIP Text Encode (Prompt)",
        {"text": ["STRING", {"multiline": True}], "clip": ["CLIP", {}]},
        ["CONDITIONING"],
        category="conditioning",
    ),
    "EmptyLatentImage": _node(
        "Empty Latent Image",
        {
            "width": ["INT", {"default": 512, "min": 16, "max": 16384, "step": 8}],
            "height": ["INT", {"default": 512, "min": 16, "max": 16384, "step": 8}],
            "batch_size": ["INT", {"default": 1, "min": 1, "max": 4096}],
        },
        ["LATENT"],
        category="latent",
    ),
    "KSampler": _node(
        "KSampler",
        {
            "model": ["MODEL", {}],
            "seed": ["INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}],
            "steps": ["INT", {"default": 20, "min": 1, "max": 10000}],
            "cfg": ["FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}],
            "sampler_name": [["euler"], {}],
            "scheduler": [["normal"], {}],
            "positive": ["CONDITIONING", {}],
            "negative": ["CONDITIONING", {}],
            "latent_image": ["LATENT", {}],
            "denoise": ["FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0}],
        },
        ["LATENT"],
        category="sampling",
    ),
    "VAEDecode": _node(
        "VAE Decode",
        {"samples": ["LATENT", {}], "vae": ["VAE", {}]},
        ["IMAGE"],
        category="latent",
    ),
    "SaveImage": _node(
        "Save Image",
        {"images": ["IMAGE", {}], "filename_prefix": ["STRING", {"default": "ComfyUI"}]},
        [],
        output_node=True,
        category="image",
    ),
    "PreviewImage": _node(
        "Preview Image",
        {"images": ["IMAGE", {}]},
        [],
        output_node=True,
        category="image",
    ),
    "LoadImage": _node(
        "Load Image",
        {"image": [["example.png"], {"image_upload": True}]},
        ["IMAGE", "MASK"],
        category="image",
    ),
    "ImageScaleBy": _node(
        "Upscale Image By",
        {
            "image": ["IMAGE", {}],
            "upscale_method": [["nearest-exact"], {}],
            "scale_by": ["FLOAT", {"default": 1.0, "min": 0.01, "max": 8.0}],
        },
        ["IMAGE"],
        category="image",
    ),
}

_ROUTES: dict[str, Any] = {
    # What readiness is decided on. The supervisor asks for any success here and
    # then checks the listener belongs to the process it started, so the body is
    # not what proves anything - the process answering is. It is shaped like the
    # real one so a reader is not misled into thinking the shape carries weight.
    "/system_stats": {
        "system": {
            "os": "synthetic",
            "comfyui_version": "0.0.0-certification",
            "python_version": "certification",
            "embedded_python": False,
        },
        "devices": [],
    },
    "/object_info": _OBJECT_INFO,
    "/embeddings": [],
    "/queue": {"queue_running": [], "queue_pending": []},
}

_PER_CLASS = "/object_info/"


class _Engine(BaseHTTPRequestHandler):
    """Only the verbs the product actually uses, and nothing else."""

    protocol_version = "HTTP/1.1"

    def do_GET(self) -> None:  # noqa: N802 - the base class names it
        path = urlsplit(self.path).path
        if path in _ROUTES:
            self._send(200, _ROUTES[path])
            return
        if path.startswith(_PER_CLASS):
            named = path[len(_PER_CLASS) :]
            described = _OBJECT_INFO.get(named)
            if described is None:
                self._send(404, {"error": "unknown node class"})
                return
            self._send(200, {named: described})
            return
        self._send(404, {"error": "not found"})

    def _send(self, status: int, body: Any) -> None:
        payload = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format: str, *args: Any) -> None:
        """Silence per-request logging; the health probe polls continuously."""


def _address() -> tuple[str, int]:
    """Read the address out of the real media launch command.

    Unknown flags are kept rather than refused: the product passes a model paths
    file, an output directory, a preview method and custom node switches, and a
    stand-in that died on any of them would be asserting its own argument parser
    rather than standing where the engine stands.
    """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--listen", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8188)
    known, _ignored = parser.parse_known_args()
    return known.listen, known.port


def main() -> None:
    host, port = _address()
    # Threading: the supervisor's readiness probe and the review's object_info
    # fetch can overlap, and a single-threaded server would serialise them into
    # a timeout that looked like an unready engine.
    ThreadingHTTPServer((host, port), _Engine).serve_forever()


if __name__ == "__main__":
    main()

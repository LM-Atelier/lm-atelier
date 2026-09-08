from __future__ import annotations

from typing import Literal

InstalledAssetKind = Literal[
    "checkpoint",
    "clip_vision",
    "controlnet",
    "diffusion_model",
    "embedding",
    "gguf_model",
    "ip_adapter",
    "lora",
    "text_encoder",
    "upscaler",
    "vae",
]

BoundWorkflowAssetKind = Literal["checkpoint", "embedding", "lora", "upscaler", "vae"]
WorkflowAssetKind = Literal[BoundWorkflowAssetKind, "configuration"]

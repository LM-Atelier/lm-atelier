from __future__ import annotations

from typing import Literal

InstalledAssetKind = Literal[
    "background_removal",
    "checkpoint",
    "clip_vision",
    "controlnet",
    "diffusion_model",
    "embedding",
    "gguf_model",
    "ip_adapter",
    "latent_upscaler",
    "lora",
    "text_encoder",
    "upscaler",
    "vae",
]

BoundWorkflowAssetKind = Literal[
    "background_removal",
    "checkpoint",
    "embedding",
    "lora",
    "upscaler",
    "vae",
]
WorkflowAssetKind = Literal[BoundWorkflowAssetKind, "configuration"]

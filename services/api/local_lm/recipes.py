from __future__ import annotations

from .schemas import DownloadRequest, RecipeFile, RecipeHardware, ReferenceRecipe

_WAN_REPOSITORY = "Comfy-Org/Wan_2.1_ComfyUI_repackaged"
_WAN_REVISION = "06e001fc51048fb03433a6fb25334de7836704a5"


REFERENCE_RECIPES: tuple[ReferenceRecipe, ...] = (
    ReferenceRecipe(
        id="qwen3-8b-q4-k-m",
        version=1,
        name="Qwen3 8B Q4_K_M",
        summary="Balanced general-purpose chat model for CPU-first local use.",
        role="chat",
        engine="llama.cpp",
        operations=["text"],
        license_id="Apache-2.0",
        status="reference-candidate",
        certified=False,
        remote_id="Qwen/Qwen3-8B-GGUF",
        revision="7c41481f57cb95916b40956ab2f0b139b296d974",
        files=[
            RecipeFile(
                path="Qwen3-8B-Q4_K_M.gguf",
                size_bytes=5_027_783_488,
                sha256="d98cdcbd03e17ce47681435b5150e34c1417f50b5c0019dd560e4882c5745785",
            )
        ],
        total_size_bytes=5_027_783_488,
        hardware=RecipeHardware(
            tier="cpu",
            minimum_ram_gb=8,
            recommended_ram_gb=16,
            guidance="Runs on a modern CPU; GPU offload is optional and improves speed.",
        ),
        default_settings={
            "context_length": 32_768,
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": 20,
            "min_p": 0.0,
            "repeat_penalty": 1.0,
        },
        notes=["Thinking mode is controlled through the Qwen chat template."],
    ),
    ReferenceRecipe(
        id="flux1-schnell-fp8",
        version=1,
        name="FLUX.1 Schnell FP8",
        summary="Fast, high-quality text-to-image generation in a single ComfyUI checkpoint.",
        role="image",
        engine="comfyui",
        operations=["text_to_image", "image_to_image"],
        license_id="Apache-2.0",
        status="reference-candidate",
        certified=False,
        remote_id="Comfy-Org/flux1-schnell",
        revision="7d679837b018bfeb28eca55734b335efcd0e7100",
        files=[
            RecipeFile(
                path="flux1-schnell-fp8.safetensors",
                size_bytes=17_200_000_000,
                sha256="ead426278b49030e9da5df862994f25ce94ab2ee4df38b556ddddb3db093bf72",
            )
        ],
        total_size_bytes=17_200_000_000,
        hardware=RecipeHardware(
            tier="midrange-gpu",
            minimum_ram_gb=24,
            recommended_ram_gb=32,
            minimum_vram_gb=12,
            recommended_vram_gb=16,
            guidance="FP8 reduces memory pressure; lower resolutions can run with model offload.",
        ),
        default_settings={
            "width": 1024,
            "height": 1024,
            "steps": 4,
            "cfg": 1.0,
            "sampler": "euler",
            "scheduler": "simple",
        },
        node_policy="ComfyUI core nodes only",
        notes=["A native checkpoint workflow must be selected or imported before generation."],
    ),
    ReferenceRecipe(
        id="wan21-t2v-13b",
        version=1,
        name="Wan 2.1 T2V 1.3B",
        summary="The affordable 480p reference path for local text-to-video generation.",
        role="video",
        engine="comfyui",
        operations=["text_to_video"],
        license_id="Apache-2.0",
        status="reference-candidate",
        certified=False,
        remote_id=_WAN_REPOSITORY,
        revision=_WAN_REVISION,
        files=[
            RecipeFile(
                path="split_files/diffusion_models/wan2.1_t2v_1.3B_fp16.safetensors",
                size_bytes=2_838_303_560,
                sha256="be531024cd9018cb5b48c40cfbb6a6191645b1c792eb8bf4f8c1c6e10f924dc5",
            ),
            RecipeFile(
                path="split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors",
                size_bytes=6_735_906_897,
                sha256="c3355d30191f1f066b26d93fba017ae9809dce6c627dda5f6a66eaa651204f68",
            ),
            RecipeFile(
                path="split_files/vae/wan_2.1_vae.safetensors",
                size_bytes=253_815_318,
                sha256="2fc39d31359a4b0a64f55876d8ff7fa8d780956ae2cb13463b0223e15148976b",
            ),
            RecipeFile(
                path="example workflows_Wan2.1/text_to_video_wan.json",
                size_bytes=8_280,
                sha256="ddb573b668360baa29e303eacb427ac99e551ec4f0307babe707a71e966c0a61",
            ),
        ],
        total_size_bytes=9_828_034_055,
        hardware=RecipeHardware(
            tier="midrange-gpu",
            minimum_ram_gb=16,
            recommended_ram_gb=32,
            minimum_vram_gb=8,
            recommended_vram_gb=12,
            guidance=(
                "Designed for 480p generation; generation time grows quickly with frame count."
            ),
        ),
        default_settings={
            "width": 832,
            "height": 480,
            "frames": 81,
            "fps": 16,
            "steps": 30,
            "guidance": 6.0,
        },
        workflow_path="example workflows_Wan2.1/text_to_video_wan.json",
        node_policy="ComfyUI core nodes only",
        notes=["Uses the official Comfy-Org safe-tensor repack instead of upstream .pth files."],
    ),
    ReferenceRecipe(
        id="wan21-i2v-14b-480p-fp8",
        version=1,
        name="Wan 2.1 I2V 14B 480p FP8",
        summary="High-end image-to-video reference recipe with an FP8 diffusion model.",
        role="video",
        engine="comfyui",
        operations=["image_to_video"],
        license_id="Apache-2.0",
        status="reference-candidate",
        certified=False,
        remote_id=_WAN_REPOSITORY,
        revision=_WAN_REVISION,
        files=[
            RecipeFile(
                path="split_files/diffusion_models/wan2.1_i2v_480p_14B_fp8_e4m3fn.safetensors",
                size_bytes=16_397_245_448,
                sha256="0ca75338e7a47ca7cacddb7e626647e65829c497387f718ecb6ea0bae456944a",
            ),
            RecipeFile(
                path="split_files/clip_vision/clip_vision_h.safetensors",
                size_bytes=1_264_219_396,
                sha256="64a7ef761bfccbadbaa3da77366aac4185a6c58fa5de5f589b42a65bcc21f161",
            ),
            RecipeFile(
                path="split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors",
                size_bytes=6_735_906_897,
                sha256="c3355d30191f1f066b26d93fba017ae9809dce6c627dda5f6a66eaa651204f68",
            ),
            RecipeFile(
                path="split_files/vae/wan_2.1_vae.safetensors",
                size_bytes=253_815_318,
                sha256="2fc39d31359a4b0a64f55876d8ff7fa8d780956ae2cb13463b0223e15148976b",
            ),
            RecipeFile(
                path="example workflows_Wan2.1/image_to_video_wan_480p_example.json",
                size_bytes=11_881,
                sha256="edff65aafefe1e94841628250240e0cab5714873486fbae442de18a8a62f89ba",
            ),
        ],
        total_size_bytes=24_651_198_940,
        hardware=RecipeHardware(
            tier="high-end-gpu",
            minimum_ram_gb=32,
            recommended_ram_gb=64,
            minimum_vram_gb=20,
            recommended_vram_gb=24,
            guidance="Intended for a high-memory discrete GPU; aggressive offload will be slow.",
        ),
        default_settings={
            "width": 832,
            "height": 480,
            "frames": 81,
            "fps": 16,
            "steps": 30,
            "guidance": 5.0,
        },
        workflow_path="example workflows_Wan2.1/image_to_video_wan_480p_example.json",
        node_policy="ComfyUI core nodes only",
        notes=["Uses the official Comfy-Org safe-tensor repack instead of upstream .pth files."],
    ),
)


def list_reference_recipes() -> list[ReferenceRecipe]:
    return list(REFERENCE_RECIPES)


def get_reference_recipe(recipe_id: str) -> ReferenceRecipe | None:
    return next((recipe for recipe in REFERENCE_RECIPES if recipe.id == recipe_id), None)


def recipe_download_request(recipe: ReferenceRecipe) -> DownloadRequest:
    comfy_paths: dict[str, str] = {}
    if recipe.id == "flux1-schnell-fp8":
        comfy_paths = {"checkpoints": "."}
    elif recipe.id.startswith("wan21-"):
        comfy_paths = {
            "diffusion_models": "split_files/diffusion_models",
            "text_encoders": "split_files/text_encoders",
            "vae": "split_files/vae",
            "clip_vision": "split_files/clip_vision",
        }
    return DownloadRequest(
        remote_id=recipe.remote_id,
        revision=recipe.revision,
        role=recipe.role,
        engine=recipe.engine,
        allow_patterns=[file.path for file in recipe.files],
        expected_sha256={file.path: file.sha256 for file in recipe.files if file.sha256},
        recipe_id=recipe.id,
        recipe_version=recipe.version,
        comfy_paths=comfy_paths,
        workflow_path=recipe.workflow_path,
        default_settings=recipe.default_settings,
    )

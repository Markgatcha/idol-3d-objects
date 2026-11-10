"""ComfyUI nodes for Idol: one image in, one 3D Gaussian-splat .ply out.

Idol runs in-process. Before loading its weights the node asks ComfyUI to unload its own
models, because Idol needs ~5 GB VRAM and ComfyUI may still be holding Qwen's ~6.7 GB.
After the run the pipeline is dropped so the next queued prompt gets its VRAM back.
"""
import gc
import json
import os
import sys
import time

import numpy as np
import torch
from PIL import Image

IDOL_SDK = os.environ.get("IDOL_SDK", r"C:\idol\idol-sdk")
IDOL_TOOLS = os.environ.get("IDOL_TOOLS", r"C:\idol\tools")
for p in (IDOL_SDK, IDOL_TOOLS):
    if p and p not in sys.path:
        sys.path.insert(0, p)

_PIPE = None
_PIPE_KEY = None


def _free_comfy_vram():
    try:
        import comfy.model_management as mm
        mm.unload_all_models()
        mm.soft_empty_cache()
    except Exception:
        pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _drop_pipeline():
    global _PIPE, _PIPE_KEY
    _PIPE = None
    _PIPE_KEY = None
    _free_comfy_vram()


def _get_pipeline(weights, dtype):
    global _PIPE, _PIPE_KEY
    key = (weights, dtype)
    if _PIPE is not None and _PIPE_KEY == key:
        return _PIPE
    _drop_pipeline()
    from idol_loader import load_pipeline
    _PIPE = load_pipeline(weights, dtype=dtype, optimize=False)
    _PIPE_KEY = key
    return _PIPE


def _to_pil(image_tensor):
    """ComfyUI IMAGE [B,H,W,C] float 0..1 -> PIL (RGBA when 4 channels)."""
    arr = (image_tensor[0].detach().float().cpu().numpy() * 255.0).round()
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.shape[2] == 4:
        return Image.fromarray(arr, "RGBA")
    return Image.fromarray(arr[:, :, :3], "RGB")


def _to_tensor(pil_image):
    arr = np.array(pil_image.convert("RGB")).astype(np.float32) / 255.0
    return torch.from_numpy(arr)[None]


class IdolImageTo3D:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mode": (["synthetic", "real"], {"default": "synthetic"}),
                "steps": ("INT", {"default": 50, "min": 10, "max": 100, "step": 1}),
                "num_gaussians": ("INT", {"default": 262144, "min": 32768,
                                          "max": 1048576, "step": 32}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 2147483647}),
                "filename_prefix": ("STRING", {"default": "3d/object"}),
            },
            "optional": {
                "weights": ("STRING", {"default": r"C:\idol\idol-bf16"}),
                "dtype": (["bf16", "fp16"], {"default": "bf16"}),
                "keep_loaded": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("STRING", "IMAGE", "STRING")
    RETURN_NAMES = ("ply_path", "preprocessed", "camera_json")
    FUNCTION = "run"
    CATEGORY = "3D/Idol"
    OUTPUT_NODE = True

    def run(self, image, mode="synthetic", steps=50, num_gaussians=262144, seed=42,
            filename_prefix="3d/object", weights=r"C:\idol\idol-bf16", dtype="bf16",
            keep_loaded=False):
        import folder_paths

        pil = _to_pil(image)
        pipeline = _get_pipeline(weights, dtype)
        t0 = time.time()
        out = pipeline(pil, mode=mode, seed=seed, steps=steps, num_gaussians=num_gaussians)
        dt = time.time() - t0

        out_dir = os.path.join(folder_paths.get_output_directory(),
                               os.path.dirname(filename_prefix) or "3d")
        os.makedirs(out_dir, exist_ok=True)
        stem = os.path.basename(filename_prefix) or "object"
        ply_path = os.path.join(out_dir, f"{stem}_{seed}.ply")
        out.gaussian.save_ply(ply_path)
        out.preprocessed_image.save(os.path.join(out_dir, f"{stem}_{seed}_input.png"))
        with open(os.path.join(out_dir, f"{stem}_{seed}_camera.json"), "w", encoding="utf-8") as f:
            json.dump(out.camera, f, indent=2)

        print(f"[Idol] {dt:.1f}s -> {ply_path} ({num_gaussians} gaussians)")
        if not keep_loaded:
            _drop_pipeline()

        return {
            "ui": {"text": [f"{ply_path}  ({dt:.1f}s, {num_gaussians} gaussians)"]},
            "result": (ply_path, _to_tensor(out.preprocessed_image), json.dumps(out.camera)),
        }


NODE_CLASS_MAPPINGS = {"IdolImageTo3D": IdolImageTo3D}
NODE_DISPLAY_NAME_MAPPINGS = {"IdolImageTo3D": "Idol — Image to 3D (.ply)"}

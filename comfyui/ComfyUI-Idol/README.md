# ComfyUI-Idol

A ComfyUI node that turns an image into a **3D Gaussian-splat `.ply`** with
[Idol](https://huggingface.co/speridlabs/idol), in-process.

## Install

1. Copy (or junction) this folder into `ComfyUI/custom_nodes/ComfyUI-Idol`.
2. Idol needs a few packages that ComfyUI does not ship. Idol itself is installed from
   [idol-sdk](https://github.com/speridlabs/idol-sdk):

```bat
python_embeded\python.exe -m pip install diffusers tensordict plyfile pyvers cloudpickle accelerate
python_embeded\python.exe -m pip install -e C:\idol\idol-sdk
```

3. Point ComfyUI at the existing Hugging Face cache so it does not re-download the gated
   encoders (DINOv3 + FLUX.2 VAE):

```bat
set HF_HOME=C:\idol\hf-cache
```

4. Accept the licence on `facebook/dinov3-vith16plus-pretrain-lvd1689m` and
   `black-forest-labs/FLUX.2-dev`, and be logged in (`hf auth login`).

If the paths differ on your machine, set `IDOL_SDK` / `IDOL_TOOLS` environment variables.

## Node: **Idol — Image to 3D (.ply)**

| input | |
|---|---|
| `image` | any IMAGE; RGBA is used as a mask (no background removal needed) |
| `mode` | `synthetic` (renders/generated images) or `real` (photos, looser crop) |
| `steps` | 50 default; fewer is faster and loses detail |
| `num_gaussians` | 32,768 – 1,048,576 (multiple of 32) |
| `seed` | fixed seed = same object |
| `filename_prefix` | output subfolder/name inside ComfyUI `output/` |
| `weights` | local folder or HF repo id (default `C:\idol\idol-bf16`) |
| `dtype` | `bf16` or `fp16` |
| `keep_loaded` | keep Idol in VRAM between runs (see below) |

Outputs: `ply_path` (string, also shown in the node's text output), `preprocessed` (the
1024² image Idol actually saw) and `camera_json`.

## VRAM

Idol needs ~5 GB. Before loading it the node calls
`comfy.model_management.unload_all_models()` and empties the CUDA cache, so a Qwen model
sitting in VRAM is evicted first. By default the pipeline is dropped after each run, which
returns the VRAM to ComfyUI at the cost of a reload (~20–40 s warm). Set `keep_loaded` on a
dedicated image→3D workflow to skip that.

## Workflows

- `workflows/image_to_3d_api.json` — LoadImage → Idol.
- `workflows/image_to_transparent_3d_api.json` — LoadImage → Qwen Image 2.1 edit
  (reference image + RGBA prompt) → Idol.

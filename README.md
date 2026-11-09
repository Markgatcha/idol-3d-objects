# Idol 3D Objects

Prompt → **2K transparent PNG** → **3D Gaussian-splat `.ply`**.

A ComfyUI workflow generates a single object on a transparent background with
Qwen-Image-2.1-Turbo, then [Idol](https://huggingface.co/speridlabs/idol) reconstructs
it as a 3D Gaussian splat from that one image.

Tested on a laptop with an **8 GB RTX 5050 (sm_120)**, Windows 11, 16 GB RAM.

![preview](docs/preview.png)

## Samples

| object | 2K RGBA | 3D splat |
|---|---|---|
| smartphone | `samples/images/smartphone.png` | `samples/ply/smartphone.ply` |
| laptop | `samples/images/laptop.png` | `samples/ply/laptop.ply` |
| wireless headphones | `samples/images/wireless_headphones.png` | `samples/ply/wireless_headphones.ply` |
| coffee mug | `samples/images/coffee_mug.png` | `samples/ply/coffee_mug.ply` |
| running shoe | `samples/images/running_shoe.png` | `samples/ply/running_shoe.ply` |
| armchair | `samples/images/armchair.png` | `samples/ply/armchair.ply` |

Open a `.ply` in any 3D Gaussian splat viewer (SuperSplat, antimatter15/splat, Polycam web viewer).

## How it works

```
prompt
  |  ComfyUI  ·  Qwen-Image-2.1-Turbo Q5_K_M (GGUF)  ·  8 steps, CFG 1
  v
2048 x 2048 RGBA PNG          (transparency comes from the prompt, not a matte)
  |  Idol  ·  BiRefNet background removal  ·  DINOv3 + FLUX.2 VAE conditioning
  v
.ply  ·  262,144 3D Gaussians  ·  standard 3DGS format
```

Qwen Image 2.1 supports RGBA natively: wrapping the prompt as
*"This is an RGBA format image with transparency. … The image has an alpha channel and a
transparent background."* makes the model emit a real alpha channel. No segmentation or
matting step is involved.

Idol detects that alpha and skips its own background removal, so the two stages stay clean.

## Setup

### 1. ComfyUI + Qwen Image 2.1

Use ComfyUI portable (v0.44.0 tested) and the GGUF loader:

```bash
git clone https://github.com/leejet/ComfyUI-GGUF ComfyUI/custom_nodes/ComfyUI-GGUF
python_embeded/python.exe -m pip install -r ComfyUI/custom_nodes/ComfyUI-GGUF/requirements.txt
```

Models (Qwen Research License, see `NOTICE`):

| file | folder | size |
|---|---|---|
| `qwen-image-2.1-turbo-Q5_K_M.gguf` | `models/diffusion_models/` | 4.6 GB |
| `qwen3vl_8b_int8_convrot.safetensors` | `models/text_encoders/` | 8.7 GB |
| `qwen_image_2.1_vae_bf16.safetensors` | `models/vae/` | 0.6 GB |

From `abenzerps/Qwen-Image-2.1-Uncensored-GGUF` on Hugging Face.

On 8 GB VRAM the new async/pinned offload path fails with
`hostbuf_file_reader_read failed`; start ComfyUI with:

```
python_embeded\python.exe -s ComfyUI\main.py --port 8188 ^
  --disable-async-offload --disable-dynamic-vram --disable-pinned-memory
```

Even with those flags, reloading the 8.7 GB text encoder between prompts can crash with an
access violation on a 16 GB machine. `scripts/generate_with_restart.py` works around it by
starting a fresh ComfyUI process per image. Slower, but it never fails.

### 2. Idol

```bash
pip install -e idol-sdk        # Python 3.10+, PyTorch 2.5+, CUDA GPU
```

You also need access to the two gated encoders (`facebook/dinov3-vith16plus-pretrain-lvd1689m`,
`black-forest-labs/FLUX.2-dev`) and `hf auth login`.

## Run

```bash
# 1. images (ComfyUI must be running on :8188)
python scripts/generate_images.py \
  --workflow workflows/qwen_2.1_turbo_api.json \
  --objects prompts/objects.txt --template prompts/prompt_template.txt \
  --out samples/images --size 2048

# 1b. or, on 8 GB / 16 GB RAM, one ComfyUI process per image (robust)
python scripts/generate_with_restart.py --comfy C:\ComfyUI_windows_portable \
  --workflow workflows/qwen_2.1_turbo_api.json \
  --objects prompts/objects.txt --template prompts/prompt_template.txt \
  --out samples/images --size 2048

# 2. 3D
python scripts/image_to_ply.py --images samples/images --out samples/ply \
  --weights speridlabs/idol --gaussians 262144 --steps 50

# or both
python scripts/run_all.py --size 2048
```

The workflow file is the ComfyUI **API** format (what `/prompt` accepts). To use it in the
ComfyUI UI, load the official template *Qwen Image 2.1 Turbo — Text to Image* from the
Template Library and swap `Load Diffusion Model` for `Unet Loader (GGUF)`.

## Measured

| stage | 1024² | 2048² |
|---|---|---|
| Qwen generation (8 steps) | 122 s first run | 212–248 s |
| Idol → 262k Gaussians | 145 s | 140 s (input resized to 1024 internally) |
| peak VRAM | ~6.7 GB | ~6.7 GB |

2K numbers include a fresh ComfyUI process per image (the restart workaround); sampling itself is
~140 s of that. Idol is a separate process and reuses the bf16 weights (load ~70 s, then ~140 s
per object). Six objects end to end: about 40 minutes on this laptop.

## Licences

- **Code and workflow in this repository: MIT** (see `LICENSE`).
- **Sample images / 3D assets: Qwen Research License**, non-commercial only, because they were
  generated with Qwen-Image-2.1-Turbo. See `NOTICE` — keep it with any redistribution.
- **Idol** (`speridlabs/idol`, `idol-sdk`): Apache-2.0.
- Model weights are not included here; download them from their publishers.

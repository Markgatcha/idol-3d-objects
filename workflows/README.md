# Workflows

| file | what it does |
|---|---|
| `qwen_2.1_turbo_api.json` | text prompt → 2K RGBA image |
| `image_to_3d_api.json` | **any image → Idol `.ply`** |
| `image_to_transparent_3d_api.json` | **any image → Qwen edit (transparent) → Idol `.ply`** |

All three are ComfyUI **API-format** workflows (the shape `/prompt` accepts).
`scripts/run_image.py` uploads a local image and runs either of the last two.

## `qwen_2.1_turbo_api.json`

Used by `scripts/generate_images.py`. It is a minimal version of the official
*Qwen Image 2.1 Turbo — Text to Image* template with the prompt-enhancer branch removed:

| node | type | purpose |
|---|---|---|
| 1 | `UnetLoaderGGUF` | `qwen-image-2.1-turbo-Q5_K_M.gguf` |
| 2 | `CLIPLoader` | `qwen3vl_8b_int8_convrot.safetensors`, type `qwen_image` |
| 3 | `VAELoader` | `qwen_image_2.1_vae_bf16.safetensors` |
| 5 | `TextEncodeQwenImage21` | positive + negative prompt; `images: {}` |
| 6 | `EmptyLatentImage` | width/height (1024 or 2048) |
| 10 | `ManualSigmas` | the 8-step Turbo schedule |
| 11 | `SamplerCustomAdvanced` | euler, CFG 1 |
| 13 | `SaveImage` | PNG, keeps the alpha channel |

The `_node_ids` block at the top maps the fields the runner rewrites
(prompt, latent size, seed, output prefix). ComfyUI ignores it because it is removed
before the request is sent.

## Using it in the UI

Load the official template *Qwen Image 2.1 Turbo — Text to Image* from the ComfyUI Template
Library, then replace `Load Diffusion Model` with **Unet Loader (GGUF)** and point it at the
Q5_K_M file. Keep the `ManualSigmas` node — a normal KSampler with 8 steps does **not**
reproduce the Turbo schedule.

## Transparency prompt

Qwen Image 2.1 emits a real alpha channel when the prompt says so:

```
This is an RGBA format image with transparency. A single {object}, shown whole and centred,
three-quarter view, studio product photograph, soft even lighting, sharp focus, true-to-life
materials and colours, no cast shadow, nothing else in the frame. The image has an alpha
channel and a transparent background.
```

RGB under fully transparent pixels is arbitrary (often purple); it is hidden by alpha 0 and
is ignored by the 3D step, which composites on black using the mask.

## `image_to_3d_api.json`

`LoadImage` → `IdolImageTo3D`. One image in, one `.ply` out. The node lives in
`comfyui/ComfyUI-Idol` (see its README for install).

## `image_to_transparent_3d_api.json`

`LoadImage` → `TextEncodeQwenImage21` (reference image + RGBA prompt) → sampler → `VAEDecode`
→ `IdolImageTo3D`, plus a `SaveImage` so the transparent PNG is kept too.

The reference image goes into the autogrow `images` input. **In API format the key is flat:**

```json
"images.image_1": ["1", 0]
```

not `"images": {"image_1": ...}`. A nested dict is silently ignored, and the model then
invents a subject from the text prompt alone (we got a cat instead of the laptop this way).

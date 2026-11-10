"""Turn RGBA object images into 3D Gaussian-splat PLYs with Idol (one image -> one .ply).

Usage:
    python image_to_ply.py --images ../samples/images --out ../samples/ply \
        --weights C:\\idol\\idol-bf16 --gaussians 262144 --steps 50
"""
import argparse
import json
import os
import sys
import time

from PIL import Image

sys.path.insert(0, r"C:\idol\tools")
from idol_loader import load_pipeline  # noqa: E402

IMAGE_EXT = (".png", ".jpg", ".jpeg", ".webp")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--weights", default=r"C:\idol\idol-bf16")
    ap.add_argument("--gaussians", type=int, default=262144)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--mode", choices=["synthetic", "real"], default="synthetic")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16"])
    ap.add_argument("--batch-cfg", action="store_true", default=False,
                    help="run guided and unguided passes as one batch (same math; measured no gain here)")
    ap.add_argument("--no-batch-cfg", dest="batch_cfg", action="store_false")
    args = ap.parse_args()

    paths = ([os.path.join(args.images, n) for n in sorted(os.listdir(args.images))
              if n.lower().endswith(IMAGE_EXT)]
             if os.path.isdir(args.images) else [args.images])
    os.makedirs(args.out, exist_ok=True)

    pipe = load_pipeline(args.weights, dtype=args.dtype, optimize=False)
    if args.batch_cfg:
        from contextlib import contextmanager

        @contextmanager
        def _no_settings(self):
            yield

        # batch_cfg batches the guided and unguided passes; identical math, ~1.5x on the denoiser.
        # We skip the rest of upstream's optimize bundle (cuDNN attention / TF32), which changes
        # numerics and hangs on some sm_120 drivers.
        pipe.optimize = True
        pipe._sampling_settings = _no_settings.__get__(pipe, type(pipe))
    log = []
    for i, path in enumerate(paths):
        name = os.path.splitext(os.path.basename(path))[0]
        ply = os.path.join(args.out, f"{name}.ply")
        if os.path.exists(ply):
            print(f"[{i+1}/{len(paths)}] skip {name}")
            continue
        image = Image.open(path).convert("RGBA")
        t0 = time.time()
        out = pipe(image, mode=args.mode, seed=args.seed, steps=args.steps,
                   num_gaussians=args.gaussians)
        out.gaussian.save_ply(ply)
        out.preprocessed_image.save(os.path.join(args.out, f"{name}_input.png"))
        with open(os.path.join(args.out, f"{name}_camera.json"), "w", encoding="utf-8") as f:
            json.dump(out.camera, f, indent=2)
        dt = time.time() - t0
        print(f"[{i+1}/{len(paths)}] {name}: {dt:.1f}s -> {ply}")
        log.append({"object": name, "ply": ply, "seconds": round(dt, 2),
                    "gaussians": args.gaussians, "camera": out.camera})

    with open(os.path.join(args.out, "generation_log.json"), "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2)


if __name__ == "__main__":
    main()

"""Run the full pipeline: ComfyUI images -> Idol PLYs."""
import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default="http://127.0.0.1:8188")
    ap.add_argument("--size", type=int, default=2048)
    ap.add_argument("--objects", default=os.path.join(HERE, "..", "prompts", "objects.txt"))
    ap.add_argument("--images", default=os.path.join(HERE, "..", "samples", "images"))
    ap.add_argument("--ply", default=os.path.join(HERE, "..", "samples", "ply"))
    ap.add_argument("--weights", default="speridlabs/idol")
    ap.add_argument("--gaussians", type=int, default=262144)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--only", default=None)
    args = ap.parse_args()

    gen = [sys.executable, os.path.join(HERE, "generate_images.py"),
           "--server", args.server, "--size", str(args.size),
           "--workflow", os.path.join(HERE, "..", "workflows", "qwen_2.1_turbo_api.json"),
           "--objects", args.objects,
           "--template", os.path.join(HERE, "..", "prompts", "prompt_template.txt"),
           "--out", args.images]
    if args.only:
        gen += ["--only", args.only]
    subprocess.run(gen, check=True)

    subprocess.run([sys.executable, os.path.join(HERE, "image_to_ply.py"),
                    "--images", args.images, "--out", args.ply,
                    "--weights", args.weights,
                    "--gaussians", str(args.gaussians), "--steps", str(args.steps)],
                   check=True)


if __name__ == "__main__":
    main()

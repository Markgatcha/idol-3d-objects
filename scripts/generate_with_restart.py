"""Generate images one per ComfyUI process.

On 8 GB VRAM / 16 GB RAM, ComfyUI can crash with an access violation when it unloads and
re-reads the 8.7 GB text encoder between prompts. Restarting the server for each image is
slower but deterministic.

Usage:
    python generate_with_restart.py --comfy C:\\ComfyUI_windows_portable \
        --workflow ../workflows/qwen_2.1_turbo_api.json \
        --objects ../prompts/objects.txt --template ../prompts/prompt_template.txt \
        --out ../samples/images --size 2048 --only "laptop,coffee mug"
"""
import argparse
import os
import subprocess
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
SERVER = "http://127.0.0.1:8188"


def wait_up(timeout=240):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen(SERVER + "/system_stats", timeout=3) as r:
                r.read()
            return True
        except Exception:
            time.sleep(2)
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--comfy", default=r"C:\ComfyUI_windows_portable")
    ap.add_argument("--port", type=int, default=8188)
    ap.add_argument("--workflow", required=True)
    ap.add_argument("--objects", required=True)
    ap.add_argument("--template", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--only", default=None)
    ap.add_argument("--flags", default="--disable-async-offload --disable-dynamic-vram --disable-pinned-memory")
    args = ap.parse_args()

    objects = [l.strip() for l in open(args.objects, encoding="utf-8") if l.strip()]
    if args.only:
        wanted = {s.strip().lower() for s in args.only.split(",")}
        objects = [o for o in objects if o.lower() in wanted]

    py = os.path.join(args.comfy, "python_embeded", "python.exe")
    main_py = os.path.join(args.comfy, "ComfyUI", "main.py")
    os.makedirs(args.out, exist_ok=True)

    for i, obj in enumerate(objects):
        name = obj.lower().replace(" ", "_")
        if os.path.exists(os.path.join(args.out, f"{name}.png")):
            print(f"skip {obj}")
            continue
        log = open(os.path.join(args.out, f"_server_{name}.log"), "w")
        cmd = [py, "-s", main_py, "--port", str(args.port), "--listen", "127.0.0.1"] + args.flags.split()
        srv = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
        try:
            if not wait_up():
                print(f"server did not start for {obj}; see log")
                continue
            subprocess.run([sys.executable, os.path.join(HERE, "generate_images.py"),
                            "--server", SERVER, "--workflow", args.workflow,
                            "--objects", args.objects, "--template", args.template,
                            "--out", args.out, "--size", str(args.size),
                            "--seed", str(args.seed + i), "--only", obj], check=True)
        finally:
            srv.terminate()
            try:
                srv.wait(timeout=20)
            except subprocess.TimeoutExpired:
                srv.kill()
            log.close()
            time.sleep(3)


if __name__ == "__main__":
    main()

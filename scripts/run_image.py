"""Upload any image to ComfyUI, run a workflow that ends in Idol, print the resulting .ply path.

Usage:
    python run_image.py --image photo.jpg --workflow workflows/image_to_3d_api.json
    python run_image.py --image photo.jpg --workflow workflows/image_to_transparent_3d_api.json \
        --prompt "This is an RGBA format image with transparency. ..."
"""
import argparse
import json
import os
import time
import urllib.parse
import urllib.request
import uuid


def upload_image(server, path):
    boundary = uuid.uuid4().hex
    name = os.path.basename(path)
    with open(path, "rb") as f:
        data = f.read()
    body = b"".join([
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"image\"; filename=\"{name}\"\r\n"
        f"Content-Type: application/octet-stream\r\n\r\n".encode(),
        data, b"\r\n",
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"overwrite\"\r\n\r\ntrue\r\n".encode(),
        f"--{boundary}--\r\n".encode(),
    ])
    req = urllib.request.Request(
        server + "/upload/image", data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())


def post_json(server, path, payload):
    req = urllib.request.Request(server + path, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())


def get_json(server, path):
    with urllib.request.urlopen(server + path, timeout=60) as r:
        return json.loads(r.read())


def wait(server, prompt_id, timeout):
    t0 = time.time()
    while time.time() - t0 < timeout:
        hist = get_json(server, f"/history/{prompt_id}")
        if prompt_id in hist:
            entry = hist[prompt_id]
            status = entry.get("status", {})
            if status.get("status_str") == "error":
                raise RuntimeError(json.dumps(status.get("messages", status))[:1200])
            if status.get("completed") or entry.get("outputs"):
                return entry
        time.sleep(2)
    raise TimeoutError(f"{timeout}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default="http://127.0.0.1:8188")
    ap.add_argument("--image", required=True)
    ap.add_argument("--workflow", required=True)
    ap.add_argument("--prompt", default=None, help="override the text prompt (edit workflow)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--gaussians", type=int, default=262144)
    ap.add_argument("--mode", default=None, choices=["synthetic", "real"])
    ap.add_argument("--timeout", type=int, default=3600)
    ap.add_argument("--out", default=None, help="download returned images here")
    args = ap.parse_args()

    wf = json.load(open(args.workflow, encoding="utf-8"))
    ids = wf.pop("_node_ids")

    up = upload_image(args.server, args.image)
    img_name = up["name"] if not up.get("subfolder") else f'{up["subfolder"]}/{up["name"]}'
    wf[ids["load"]]["inputs"]["image"] = img_name

    if "text" in ids:
        wf[ids["text"]]["inputs"]["prompt"] = args.prompt if args.prompt else wf[ids["text"]]["inputs"]["prompt"]
    if "noise" in ids:
        wf[ids["noise"]]["inputs"]["noise_seed"] = args.seed
    if "idol" in ids and ids["idol"] in wf:
        idol = wf[ids["idol"]]["inputs"]
        idol["seed"] = args.seed
        idol["steps"] = args.steps
        idol["num_gaussians"] = args.gaussians
        if args.mode:
            idol["mode"] = args.mode

    resp = post_json(args.server, "/prompt", {"prompt": wf, "client_id": str(uuid.uuid4())})
    entry = wait(args.server, resp["prompt_id"], args.timeout)

    for node_id, out in entry.get("outputs", {}).items():
        for line in out.get("text", []):
            print(f"node {node_id}: {line}")
        for img in out.get("images", []):
            print(f"node {node_id}: image {img.get('subfolder','')}/{img['filename']}")
            if args.out:
                os.makedirs(args.out, exist_ok=True)
                q = urllib.parse.urlencode({"filename": img["filename"],
                                            "subfolder": img.get("subfolder", ""),
                                            "type": img.get("type", "output")})
                with urllib.request.urlopen(args.server + "/view?" + q, timeout=120) as r:
                    data = r.read()
                with open(os.path.join(args.out, img["filename"]), "wb") as f:
                    f.write(data)


if __name__ == "__main__":
    main()

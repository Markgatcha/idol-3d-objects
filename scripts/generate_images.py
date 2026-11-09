"""Batch-generate RGBA object images from a running ComfyUI server.

Reads a prompt per line, submits the API workflow, waits for completion and saves
the PNG (alpha preserved) into --out.

Usage:
    python generate_images.py --server http://127.0.0.1:8188 \
        --workflow ../workflows/qwen_2.1_turbo_api.json \
        --objects ../prompts/objects.txt --template ../prompts/prompt_template.txt \
        --out ../samples/images --size 1024 --seed 42
"""
import argparse
import json
import os
import time
import urllib.parse
import urllib.request
import uuid


def post_json(server, path, payload):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(server + path, data=data,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())


def get_json(server, path):
    with urllib.request.urlopen(server + path, timeout=60) as r:
        return json.loads(r.read())


def get_bytes(server, query):
    with urllib.request.urlopen(server + "/view?" + query, timeout=120) as r:
        return r.read()


def wait_for(server, prompt_id, timeout=1800, poll=2.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        hist = get_json(server, f"/history/{prompt_id}")
        if prompt_id in hist:
            entry = hist[prompt_id]
            status = entry.get("status", {})
            if status.get("completed") or entry.get("outputs"):
                return entry
            if status.get("status_str") == "error":
                raise RuntimeError(f"ComfyUI error: {json.dumps(status)[:800]}")
        time.sleep(poll)
    raise TimeoutError(f"prompt {prompt_id} did not finish in {timeout}s")


def build(base, node_ids, prompt, negative, width, height, seed, prefix):
    wf = json.loads(json.dumps(base))
    wf[node_ids["text_encode"]]["inputs"]["prompt"] = prompt
    wf[node_ids["text_encode"]]["inputs"]["negative_prompt"] = negative
    wf[node_ids["text_encode"]]["inputs"]["resolution"] = width
    wf[node_ids["latent"]]["inputs"]["width"] = width
    wf[node_ids["latent"]]["inputs"]["height"] = height
    wf[node_ids["noise"]]["inputs"]["noise_seed"] = seed
    wf[node_ids["save"]]["inputs"]["filename_prefix"] = prefix
    return wf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default="http://127.0.0.1:8188")
    ap.add_argument("--workflow", required=True)
    ap.add_argument("--objects", required=True)
    ap.add_argument("--template", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--negative", default="")
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--only", default=None, help="comma separated subset of object names")
    ap.add_argument("--keep-loaded", action="store_true",
                    help="do not unload models between images (needs enough RAM)")
    args = ap.parse_args()

    base = json.load(open(args.workflow, encoding="utf-8"))
    node_ids = base.pop("_node_ids")
    template = open(args.template, encoding="utf-8").read().strip()
    objects = [l.strip() for l in open(args.objects, encoding="utf-8") if l.strip()]
    if args.only:
        wanted = {s.strip().lower() for s in args.only.split(",")}
        objects = [o for o in objects if o.lower() in wanted]

    os.makedirs(args.out, exist_ok=True)
    client_id = str(uuid.uuid4())
    results = []

    for i, obj in enumerate(objects):
        name = obj.lower().replace(" ", "_")
        target = os.path.join(args.out, f"{name}.png")
        if os.path.exists(target):
            print(f"[{i+1}/{len(objects)}] skip {obj} (exists)")
            results.append({"object": obj, "file": target, "skipped": True})
            continue
        prompt = template.format(object=obj)
        wf = build(base, node_ids, prompt, args.negative, args.size, args.size,
                   args.seed + i, f"idol3d/{name}")
        t0 = time.time()
        resp = post_json(args.server, "/prompt", {"prompt": wf, "client_id": client_id})
        entry = wait_for(args.server, resp["prompt_id"], timeout=args.timeout)
        images = []
        for node_out in entry.get("outputs", {}).values():
            images.extend(node_out.get("images", []))
        if not images:
            raise RuntimeError(f"no image returned for {obj}")
        img = images[0]
        query = urllib.parse.urlencode({"filename": img["filename"],
                                        "subfolder": img.get("subfolder", ""),
                                        "type": img.get("type", "output")})
        data = get_bytes(args.server, query)
        with open(target, "wb") as f:
            f.write(data)
        dt = time.time() - t0
        print(f"[{i+1}/{len(objects)}] {obj}: {dt:.1f}s -> {target}")
        results.append({"object": obj, "file": target, "seconds": round(dt, 2),
                        "seed": args.seed + i})
        if not args.keep_loaded:
            # 8 GB cards: unloading avoids the text-encoder reload crash on 16 GB RAM
            try:
                post_json(args.server, "/free", {"unload_models": True, "free_memory": True})
            except Exception:
                pass

    meta = os.path.join(args.out, "generation_log.json")
    log = json.load(open(meta)) if os.path.exists(meta) else []
    log.extend(results)
    json.dump(log, open(meta, "w", encoding="utf-8"), indent=2)
    print(f"\n{len(results)} images -> {args.out}")


if __name__ == "__main__":
    main()

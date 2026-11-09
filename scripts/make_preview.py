"""Composite sample RGBA images on a checkerboard and tile them into one preview PNG."""
import argparse
import os

import numpy as np
from PIL import Image

IMAGE_EXT = (".png", ".jpg", ".jpeg", ".webp")


def checker(w, h, size=16):
    yy, xx = np.mgrid[0:h, 0:w]
    c = ((yy // size + xx // size) % 2).astype(np.uint8)
    return np.where(c[..., None] == 0, 235, 205).astype(np.uint8).repeat(3, axis=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tile", type=int, default=512)
    ap.add_argument("--cols", type=int, default=3)
    args = ap.parse_args()

    paths = [os.path.join(args.images, n) for n in sorted(os.listdir(args.images))
             if n.lower().endswith(IMAGE_EXT)]
    if not paths:
        raise SystemExit("no images found")

    tiles = []
    for p in paths:
        im = Image.open(p).convert("RGBA").resize((args.tile, args.tile), Image.LANCZOS)
        bg = Image.fromarray(checker(args.tile, args.tile)).convert("RGB")
        bg.paste(im, mask=im.split()[3])
        tiles.append(bg)

    rows = (len(tiles) + args.cols - 1) // args.cols
    sheet = Image.new("RGB", (args.cols * args.tile, rows * args.tile), "white")
    for i, t in enumerate(tiles):
        sheet.paste(t, ((i % args.cols) * args.tile, (i // args.cols) * args.tile))
    sheet.save(args.out)
    print(f"{len(tiles)} tiles -> {args.out}")


if __name__ == "__main__":
    main()

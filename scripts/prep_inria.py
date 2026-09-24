

"""Prepare the Inria benchmark: tiling, the split and the index files."""

import argparse
import json
import os
import re
from pathlib import Path

import cv2
import numpy as np

CITIES = ["austin", "chicago", "kitsap", "tyrol-w", "vienna"]
STARTS = [0, 1024, 2048, 3072, 3976]
TILE = 1024


def which_split(idx):
    if idx <= 5:
        return "test"
    if idx <= 8:
        return "val"
    return "train"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help=".../AerialImageDataset/train")
    ap.add_argument("--dst", required=True)
    ap.add_argument("--splits-dir", default="splits")
    ap.add_argument("--data-root", default=".",
                    help="index paths are written relative to this directory")
    ap.add_argument("--limit", type=int, default=0, help="smoke test: process only the first N source images")
    a = ap.parse_args()

    src, dst = Path(a.src), Path(a.dst)
    for k in ("images", "masks", "inst"):
        (dst / k).mkdir(parents=True, exist_ok=True)

    names = sorted(p.name for p in (src / "images").glob("*.tif"))
    if a.limit:
        names = names[:a.limit]
    index, split = {}, {"train": [], "val": [], "test": [], "test_nooverlap": []}
    stats = {k: {"tiles": 0, "empty": 0, "bld_px": 0, "px": 0}
             for k in ("train", "val", "test", "test_nooverlap")}

    for n in names:
        m = re.match(r"([a-z\-]+?)(\d+)\.tif$", n)
        if not m:
            raise SystemExit(f"[FAIL] cannot parse the file name: {n}")
        city, idx = m.group(1), int(m.group(2))
        if city not in CITIES:
            raise SystemExit(f"[FAIL] unknown city {city} ({n})")
        sp = which_split(idx)
        rgb = cv2.imread(str(src / "images" / n), cv2.IMREAD_COLOR)
        gt = cv2.imread(str(src / "gt" / n), cv2.IMREAD_GRAYSCALE)
        if rgb is None or gt is None:
            raise SystemExit(f"[FAIL] cannot read {n}")
        if rgb.shape[:2] != (5000, 5000) or gt.shape != (5000, 5000):
            raise SystemExit(f"[FAIL] {n} is not 5000 by 5000: {rgb.shape} {gt.shape}")
        gt = (gt > 127).astype(np.uint8)

        for r, y0 in enumerate(STARTS):
            for c, x0 in enumerate(STARTS):
                stem = f"{city}{idx}_r{r}c{c}"
                ti = rgb[y0:y0 + TILE, x0:x0 + TILE]
                tm = gt[y0:y0 + TILE, x0:x0 + TILE]
                n_cc, lab = cv2.connectedComponents(tm, connectivity=8)
                if n_cc > 65535:
                    raise SystemExit(f"[FAIL] {stem} instance count exceeds uint16")
                p_rgb = dst / "images" / f"{stem}.png"
                p_msk = dst / "masks" / f"{stem}.png"
                p_ins = dst / "inst" / f"{stem}.png"
                cv2.imwrite(str(p_rgb), ti)
                cv2.imwrite(str(p_msk), tm * 255)
                cv2.imwrite(str(p_ins), lab.astype(np.uint16))
                rel = lambda p: os.path.relpath(p, a.data_root).replace(os.sep, "/")
                index[stem] = {"rgb": rel(p_rgb), "label": rel(p_msk),
                               "inst": rel(p_ins), "source": city,
                               "image": n, "grid": [r, c]}
                split[sp].append(stem)
                targets = [sp]
                if sp == "test" and r < 4 and c < 4:
                    split["test_nooverlap"].append(stem)
                    targets.append("test_nooverlap")
                for t in targets:
                    s = stats[t]
                    s["tiles"] += 1
                    s["empty"] += int(tm.max() == 0)
                    s["bld_px"] += int(tm.sum())
                    s["px"] += tm.size
        print(f"{n}: {sp}  cumulative {len(index)} tiles", flush=True)

    for k, s in stats.items():
        s["empty_frac"] = s["empty"] / max(s["tiles"], 1)
        s["building_px_frac"] = s["bld_px"] / max(s["px"], 1)

    sd = Path(a.splits_dir)
    sd.mkdir(parents=True, exist_ok=True)
    (sd / "inria_index.json").write_text(json.dumps(index, indent=0), encoding="utf-8")
    (sd / "inria_split.json").write_text(json.dumps(split, indent=0), encoding="utf-8")
    (sd / "inria_stats.json").write_text(json.dumps(stats, indent=1), encoding="utf-8")
    print(json.dumps({k: (len(v)) for k, v in split.items()}))
    for k, s in stats.items():
        print(f"{k:15s} tiles={s['tiles']:5d} empty={s['empty_frac']:.3f} "
              f"building_px={s['building_px_frac']:.4f}")


if __name__ == "__main__":
    main()

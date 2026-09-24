"""Prepare the WHU building dataset (aerial subset, COCO-format release).

Rasterises the COCO polygon annotations of each partition into a binary
mask and an instance map, and writes `splits/whu_index.json` and
`splits/whu_split.json` with paths relative to --data-root.

Expected input, i.e. the three official archives unpacked under the data root:

    <data-root>/whu_building/annotation/{train,validation,test}.json
    <data-root>/whu_building/{train,validation,test}/<stem>.TIF

Output:

    <data-root>/whu_building/converted/<partition>/masks/<stem>.png   0/255
    <data-root>/whu_building/converted/<partition>/inst/<stem>.png    uint16

Each annotated building is filled with its own identifier (its 1-based
position in the annotation list of the image), so the instances of the
ground truth are the annotated buildings and abutting buildings keep
separate identifiers. Predicted instances, by contrast, are connected
components (scripts/inst_metrics.py), which is why the merge rate has a
nonzero baseline when the ground truth is scored against itself.

    python scripts/prep_whu.py --data-root <data-root>
"""

import argparse
import json
import os

import cv2
import numpy as np

PARTITIONS = (("train", "train"), ("validation", "val"), ("test", "test"))


def rasterise(d, out_dir):
    by_img = {}
    for a in d["annotations"]:
        by_img.setdefault(a["image_id"], []).append(a)
    os.makedirs(os.path.join(out_dir, "masks"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "inst"), exist_ok=True)
    n_ann = n_empty = 0
    stems = []
    for k_img, im in enumerate(d["images"], 1):
        H, W = im["height"], im["width"]
        stem = os.path.splitext(os.path.basename(im["file_name"]))[0]
        anns = by_img.get(im["id"], [])
        sem = np.zeros((H, W), np.uint8)
        inst = np.zeros((H, W), np.uint16)
        for k, a in enumerate(anns, 1):
            segs = a["segmentation"]
            if not isinstance(segs, list):
                continue  # RLE is not used by this release
            polys = [np.asarray(s, np.float64).reshape(-1, 2).round().astype(np.int32)
                     for s in segs if len(s) >= 6]
            if not polys:
                continue
            cv2.fillPoly(sem, polys, 1)
            cv2.fillPoly(inst, polys, int(k))
        if len(anns) > 65535:
            raise SystemExit(f"[FAIL] {stem}: more than 65535 buildings")
        cv2.imwrite(os.path.join(out_dir, "masks", stem + ".png"), sem * 255)
        cv2.imwrite(os.path.join(out_dir, "inst", stem + ".png"), inst)
        stems.append(stem)
        n_ann += len(anns)
        n_empty += int(not anns)
        if k_img % 500 == 0:
            print(f"  {k_img}/{len(d['images'])}", flush=True)
    return stems, n_ann, n_empty


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=".",
                    help="directory containing whu_building/; index paths are relative to it")
    ap.add_argument("--splits-dir", default="splits")
    ap.add_argument("--partitions", nargs="+", default=[p for p, _ in PARTITIONS],
                    choices=[p for p, _ in PARTITIONS])
    a = ap.parse_args()

    base = os.path.join(a.data_root, "whu_building")
    index, split = {}, {"train": [], "val": [], "test": []}
    for part, key in PARTITIONS:
        if part not in a.partitions:
            continue
        ann = os.path.join(base, "annotation", part + ".json")
        if not os.path.exists(ann):
            raise SystemExit(f"[FAIL] missing {ann}")
        print(f"{part}: rasterising {ann}", flush=True)
        d = json.load(open(ann, encoding="utf-8"))
        stems, n_ann, n_empty = rasterise(d, os.path.join(base, "converted", part))
        for stem in sorted(stems):
            index[stem] = {
                "rgb": f"whu_building/{part}/{stem}.TIF",
                "label": f"whu_building/converted/{part}/masks/{stem}.png",
                "inst": f"whu_building/converted/{part}/inst/{stem}.png",
                "source": part,
            }
            split[key].append(stem)
        print(f"{part}: {len(stems)} images, {n_ann} buildings, "
              f"{n_empty} images without a building ({100 * n_empty / max(len(stems), 1):.1f}%)")

    if a.partitions == [p for p, _ in PARTITIONS]:
        os.makedirs(a.splits_dir, exist_ok=True)
        with open(os.path.join(a.splits_dir, "whu_index.json"), "w", encoding="utf-8") as f:
            json.dump(index, f, indent=0)
        with open(os.path.join(a.splits_dir, "whu_split.json"), "w", encoding="utf-8") as f:
            json.dump(split, f, indent=0)
        print(f"wrote {a.splits_dir}/whu_index.json ({len(index)} tiles) and whu_split.json")
    else:
        print("partial run: splits not written")


if __name__ == "__main__":
    main()

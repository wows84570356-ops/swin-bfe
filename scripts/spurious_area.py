"""Size and location of the false positives of a model (Sec. IV-B).

A spurious component is a predicted 8-connected component of at least 16 pixels
that covers no ground-truth building by more than 10 % (Eq. 12); this is the
same rule as scripts/inst_metrics.py. The per-image records in records/ count
spurious components and false-positive pixels but not the area of each
component, so the area is measured here from the predictions.

Two modes:

  measure (GPU; needs the checkpoint and the data root):
    python scripts/spurious_area.py --ckpt runs/whu_ctxnone_a05_s0/best.pt \
        --out records/whu_ctxnone_a05_s0_spurious.csv --data-root <data-root>

  summarize (CPU; reads the shipped records only):
    python scripts/spurious_area.py --summarize

The measure mode also checks itself: the number of spurious components and of
false-positive pixels it finds on every image must equal the per-image record
of the same run, otherwise the predictions are not the ones the paper used.
"""
import argparse
import csv
import glob
import json
import os
import re
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RECORDS = os.path.join(ROOT, "records")


def spurious_areas(pred, gt_inst, min_area=16):
    import cv2
    import numpy as np
    n_p, lab, stats, _ = cv2.connectedComponentsWithStats(pred.astype(np.uint8), connectivity=8)
    hit = set()
    for g in [i for i in np.unique(gt_inst) if i != 0]:
        gm = gt_inst == g
        if gm.sum() < min_area:
            continue
        ov = np.bincount(lab[gm], minlength=n_p)
        ov[0] = 0
        hit.update(int(p) for p in np.nonzero(ov)[0] if ov[p] / gm.sum() > 0.1)
    return [int(stats[p, cv2.CC_STAT_AREA]) for p in range(1, n_p)
            if stats[p, cv2.CC_STAT_AREA] >= min_area and p not in hit]


def measure(a):
    import cv2
    import numpy as np
    import torch
    sys.path.insert(0, HERE)
    sys.path.insert(0, ROOT)
    from eval_whu import build_net, predict
    from swinbfe.dataset import resolve_path
    index = json.load(open(a.index))
    stems = json.load(open(a.splitjson))["test"]
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    net = build_net(ck, 512, "cuda")
    ref = {r["stem"]: r for r in csv.DictReader(open(a.record))} if a.record else {}
    rows, mism = [], 0
    with torch.no_grad():
        for s in stems[: a.limit or None]:
            rec = index[s]
            rgb = cv2.cvtColor(cv2.imread(resolve_path(rec["rgb"], a.data_root), cv2.IMREAD_COLOR),
                               cv2.COLOR_BGR2RGB)
            gt = cv2.imread(resolve_path(rec["label"], a.data_root), cv2.IMREAD_GRAYSCALE) > 127
            inst = cv2.imread(resolve_path(rec["inst"], a.data_root), cv2.IMREAD_UNCHANGED)
            pred = predict(net, rgb, "cuda", tile=512) > 0.5
            areas = spurious_areas(pred, inst)
            fp = int(np.logical_and(pred, ~gt).sum())
            r = ref.get(s)
            if r is not None and (len(areas) != int(r["fp_inst"]) or fp != int(r["fp"])):
                mism += 1
            rows.append((s, int(inst.max() > 0), len(areas), sum(areas), fp,
                         r["fp_inst"] if r else "", r["fp"] if r else ""))
    with open(a.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["stem", "has_gt", "n_spur", "spur_area", "fp_px", "ref_fp_inst", "ref_fp_px"])
        w.writerows(rows)
    print("wrote %s: %d images, %d disagree with the record" % (a.out, len(rows), mism))


def summarize():
    runs = {}
    for p in sorted(glob.glob(os.path.join(RECORDS, "whu_*_spurious.csv"))):
        tag = re.match(r"(.+)_s\d_spurious\.csv$", os.path.basename(p)).group(1)
        rows = [r for r in csv.DictReader(open(p)) if r["has_gt"] == "1"]
        n = sum(int(r["n_spur"]) for r in rows)
        area = sum(int(r["spur_area"]) for r in rows)
        fp = sum(int(r["fp_px"]) for r in rows)
        runs.setdefault(tag, []).append(dict(n=n, mean_area=area / n, share=area / fp,
                                             attached=(fp - area) / len(rows)))
    print("nonempty WHU test images; mean +- SD over three seeds")
    for tag, v in runs.items():
        print("  %-18s spurious %6.0f  mean area %5.0f +- %4.0f px  share of FP px %.3f  "
              "attached FP px/img %5.0f +- %4.0f"
              % (tag, st.mean(x["n"] for x in v), st.mean(x["mean_area"] for x in v),
                 st.stdev(x["mean_area"] for x in v), st.mean(x["share"] for x in v),
                 st.mean(x["attached"] for x in v), st.stdev(x["attached"] for x in v)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--summarize", action="store_true")
    ap.add_argument("--ckpt")
    ap.add_argument("--out")
    ap.add_argument("--record", default=None,
                    help="the run's eval_test_perimage.csv, to check counts image by image")
    ap.add_argument("--data-root", default=os.environ.get("SWINBFE_DATA_ROOT", "."))
    ap.add_argument("--index", default=os.path.join(ROOT, "splits", "whu_index.json"))
    ap.add_argument("--splitjson", default=os.path.join(ROOT, "splits", "whu_split.json"))
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    if a.summarize:
        summarize()
    else:
        if not (a.ckpt and a.out):
            ap.error("--ckpt and --out are required unless --summarize is given")
        measure(a)

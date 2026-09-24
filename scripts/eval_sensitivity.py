

"""Sensitivity of the reported value to the scoring convention."""

import argparse
import csv
import json
import os
import sys

import cv2
import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))
from eval_whu import build_net, predict
from swinbfe.dataset import resolve_path
from inst_metrics import aggregate

TAUS = (0.1, 0.3, 0.5)
BANDS = (1, 2, 3)


def instance_stats_tau(pred_mask, gt_inst, tau, min_area=16):
    pred_mask = pred_mask.astype(np.uint8)
    n_p, lab_p, stats_p, _ = cv2.connectedComponentsWithStats(pred_mask, connectivity=8)
    gt_ids = [i for i in np.unique(gt_inst) if i != 0]
    gt_to_pred, pred_to_gt = {}, {}
    for g in gt_ids:
        gm = gt_inst == g
        area = int(gm.sum())
        if area < min_area:
            continue
        overl = np.bincount(lab_p[gm], minlength=n_p)
        overl[0] = 0
        hits = [p for p in np.nonzero(overl)[0] if overl[p] / max(area, 1) > tau]
        gt_to_pred[int(g)] = hits
        for p in hits:
            pred_to_gt.setdefault(p, []).append(int(g))
    n_gt = len(gt_to_pred)
    found = sum(1 for v in gt_to_pred.values() if v)
    splits = sum(max(0, len(v) - 1) for v in gt_to_pred.values())
    merges = sum(max(0, len(v) - 1) for v in pred_to_gt.values())
    fp = sum(1 for p in range(1, n_p)
             if stats_p[p, cv2.CC_STAT_AREA] >= min_area and p not in pred_to_gt)
    n_pred_inst = sum(1 for p in range(1, n_p)
                      if stats_p[p, cv2.CC_STAT_AREA] >= min_area)
    return dict(n_gt=n_gt, found=found, splits=splits, merges=merges,
                fp=fp, n_pred_inst=n_pred_inst, count_err=n_pred_inst - n_gt)


def boundary_iou_d(pred, gt, d):
    pred = pred.astype(np.uint8); gt = gt.astype(np.uint8)
    k = np.ones((2 * d + 1, 2 * d + 1), np.uint8)
    pb = pred - cv2.erode(pred, k, iterations=1)
    gb = gt - cv2.erode(gt, k, iterations=1)
    inter = np.logical_and(pb, gb).sum()
    union = np.logical_or(pb, gb).sum()
    return inter / union if union else 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--index", default="splits/whu_index.json")
    ap.add_argument("--splitjson", default="splits/whu_split.json")
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--tile", type=int, default=512)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--taus", type=float, nargs="+", default=list(TAUS))
    ap.add_argument("--bands", type=int, nargs="+", default=list(BANDS))
    a = ap.parse_args()

    index = json.load(open(a.index))
    stems = json.load(open(a.splitjson))[a.split]
    if a.limit:
        stems = stems[:a.limit]
    os.makedirs(a.out, exist_ok=True)
    run = os.path.basename(os.path.dirname(a.ckpt))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    net = build_net(ck, a.tile, device)
    print(f"[{run}] {len(stems)} images, tau={a.taus}, d={a.bands}, device={device}", flush=True)

    TP = FP = FN = 0
    rows = {t: [] for t in a.taus}
    bious = {d: [] for d in a.bands}
    per_img = []
    for k, s in enumerate(stems, 1):
        rec = index[s]
        bgr = cv2.imread(resolve_path(rec["rgb"], a.data_root), cv2.IMREAD_COLOR)
        if bgr is None:
            raise SystemExit(f"[FAIL] cannot read {rec['rgb']}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        gt = cv2.imread(resolve_path(rec["label"], a.data_root), cv2.IMREAD_GRAYSCALE) > 127
        inst = cv2.imread(resolve_path(rec["inst"], a.data_root), cv2.IMREAD_UNCHANGED)
        pred = predict(net, rgb, device, tile=a.tile) > 0.5
        tp = int(np.logical_and(pred, gt).sum())
        fp = int(np.logical_and(pred, ~gt).sum())
        fn = int(np.logical_and(~pred, gt).sum())
        TP += tp; FP += fp; FN += fn
        row = {"stem": s, "tp": tp, "fp": fp, "fn": fn}
        for t in a.taus:
            st = instance_stats_tau(pred, inst, t)
            rows[t].append(st)
            for kk in ("n_gt", "found", "merges", "splits", "fp", "n_pred_inst"):
                row[f"{kk}@tau{t}"] = st[kk]
        for d in a.bands:
            bi = boundary_iou_d(pred, gt, d) if inst.max() > 0 else float("nan")
            if inst.max() > 0:
                bious[d].append(bi)
            row[f"biou@d{d}"] = bi
        per_img.append(row)
        if k % 200 == 0:
            print(f"  {k}/{len(stems)}", flush=True)

    out = {"ckpt": a.ckpt, "run": run, "split": a.split, "n": len(stems),
           "IoU": TP / max(TP + FP + FN, 1),
           "by_tau": {str(t): aggregate(rows[t]) for t in a.taus},
           "boundary_iou_by_d": {str(d): float(np.mean(bious[d])) for d in a.bands}}
    json.dump(out, open(os.path.join(a.out, f"{run}_sens.json"), "w"), indent=1)
    with open(os.path.join(a.out, f"{run}_sens_perimage.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_img[0].keys()))
        w.writeheader(); w.writerows(per_img)
    print(json.dumps({"IoU": round(out["IoU"], 4),
                      **{f"fp_per_img@tau{t}": round(out['by_tau'][str(t)]['fp_per_img'], 3) for t in a.taus},
                      **{f"merges@tau{t}": round(out['by_tau'][str(t)]['merges_per_found'], 4) for t in a.taus},
                      **{f"biou@d{d}": round(out['boundary_iou_by_d'][str(d)], 4) for d in a.bands}},
                     ensure_ascii=False))
    print("wrote", os.path.join(a.out, f"{run}_sens.json"))


if __name__ == "__main__":
    main()

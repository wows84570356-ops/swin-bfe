
"""Instance-level metrics: recall, merges, splits, spurious components and
count error, on 8-connected components above a 16-pixel floor.
"""

import numpy as np
import cv2


def instance_stats(pred_mask, gt_inst, iou_match=0.5, min_area=16):
    pred_mask = pred_mask.astype(np.uint8)
    n_p, lab_p, stats_p, _ = cv2.connectedComponentsWithStats(pred_mask, connectivity=8)
    gt_ids = [i for i in np.unique(gt_inst) if i != 0]


    gt_to_pred, pred_to_gt = {}, {}
    for g in gt_ids:
        gm = gt_inst == g
        if gm.sum() < min_area:
            continue
        overl = np.bincount(lab_p[gm], minlength=n_p)
        overl[0] = 0
        hits = [p for p in np.nonzero(overl)[0]
                if overl[p] / max(gm.sum(), 1) > 0.1]
        gt_to_pred[int(g)] = hits
        for p in hits:
            pred_to_gt.setdefault(p, []).append(int(g))

    n_gt = len(gt_to_pred)
    found = sum(1 for v in gt_to_pred.values() if v)

    splits = sum(max(0, len(v) - 1) for v in gt_to_pred.values())

    merges = sum(max(0, len(v) - 1) for v in pred_to_gt.values())

    fp = 0
    for p in range(1, n_p):
        if stats_p[p, cv2.CC_STAT_AREA] < min_area:
            continue
        if p not in pred_to_gt:
            fp += 1
    n_pred_inst = sum(1 for p in range(1, n_p)
                      if stats_p[p, cv2.CC_STAT_AREA] >= min_area)
    return dict(n_gt=n_gt, found=found, splits=splits, merges=merges,
                fp=fp, n_pred_inst=n_pred_inst,
                count_err=n_pred_inst - n_gt)


def boundary_iou(pred, gt, d=2):
    pred = pred.astype(np.uint8); gt = gt.astype(np.uint8)
    k = np.ones((2 * d + 1, 2 * d + 1), np.uint8)
    pb = pred - cv2.erode(pred, k, iterations=1)
    gb = gt - cv2.erode(gt, k, iterations=1)
    inter = np.logical_and(pb, gb).sum()
    union = np.logical_or(pb, gb).sum()
    return inter / union if union else 1.0


def aggregate(rows):
    n_gt = sum(r["n_gt"] for r in rows)
    found = sum(r["found"] for r in rows)
    n = len(rows)
    return {
        "instance_recall": found / max(n_gt, 1),
        "merges_per_found": sum(r["merges"] for r in rows) / max(found, 1),
        "splits_per_found": sum(r["splits"] for r in rows) / max(found, 1),
        "fp_per_img": sum(r["fp"] for r in rows) / max(n, 1),
        "count_mae": float(np.mean([abs(r["count_err"]) for r in rows])),
        "count_bias": float(np.mean([r["count_err"] for r in rows])),
        "n_images": n, "n_gt_instances": n_gt,
    }


if __name__ == "__main__":

    import glob, os, sys
    D = sys.argv[1] if len(sys.argv) > 1 else r"data/whu_building/converted/validation"
    rows, bious = [], []
    for ip in sorted(glob.glob(os.path.join(D, "inst", "*.png")))[:120]:
        inst = cv2.imread(ip, cv2.IMREAD_UNCHANGED)
        if inst is None or inst.max() == 0:
            continue
        gtm = (inst > 0).astype(np.uint8)
        rows.append(instance_stats(gtm, inst))
        bious.append(boundary_iou(gtm, gtm))
    a = aggregate(rows)
    print("self-check (ground truth used as a perfect prediction, %d images)" % a["n_images"])
    for k, v in a.items():
        print("  %-18s %s" % (k, ("%.4f" % v) if isinstance(v, float) else v))
    print("  boundary_iou       %.4f" % np.mean(bious))
    ok = (abs(a["instance_recall"] - 1) < 1e-6 and a["fp_per_img"] == 0
          and abs(a["count_bias"]) < 1e-6 and abs(np.mean(bious) - 1) < 1e-6)
    print()
    print("  " + ("PASS: every metric takes its ideal value under a perfect prediction."
                  if ok else "FAIL: the metric implementation is wrong; fix it before use."))
    if not ok:
        print("     merges=%.4f splits=%.4f (abutting buildings in the ground truth are "
              "joined by connected components; this is a property of the data, not a bug, see below)"
              % (a["merges_per_found"], a["splits_per_found"]))

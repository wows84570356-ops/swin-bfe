"""Does the multiplier of the resolution rule (Eq. 14) decide any verdict? (Sec. IV-C)

A difference is reported as unresolved when it is smaller than twice the larger
of the two across-seed standard deviations. For every model pair on both
datasets and for the three metrics the conclusions use, this prints the ratio
|mean difference| / max(seed SD) and counts the comparisons whose verdict would
change for some multiplier between 1.5 and 3. Reads records/ only; no GPU.

    python scripts/rule_sensitivity.py
"""
import csv
import itertools
import math
import os
import statistics as st

RECORDS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "records")
DATASETS = {
    "WHU": {"Swin-BFE": "whu_ctxnone_a05", "Swin-BFE-Mamba": "whu_ours_a05",
            "Scan": "whu_vanilla_a05", "U-Net": "whu_unet_a05", "Reference": "whu_rescbam_a05"},
    "Inria": {"Swin-BFE": "inria_ctxnone_a05", "Swin-BFE-Mamba": "inria_ours_a05",
              "U-Net": "inria_unet_a05", "Reference": "inria_rescbam_a05"},
}
LOW, HIGH = 1.5, 3.0


def metrics(tag, seed):
    rows = list(csv.DictReader(open(os.path.join(RECORDS, "%s_s%d_eval_test_perimage.csv" % (tag, seed)))))
    tp = sum(int(r["tp"]) for r in rows)
    fp = sum(int(r["fp"]) for r in rows)
    fn = sum(int(r["fn"]) for r in rows)
    bi = [float(r["boundary_iou"]) for r in rows if r["boundary_iou"] not in ("", "nan")]
    bi = [b for b in bi if not math.isnan(b)]
    return {"pixel IoU": tp / (tp + fp + fn), "Boundary IoU": st.mean(bi),
            "spurious/img": sum(int(r["fp_inst"]) for r in rows) / len(rows)}


def main():
    rows = []
    for ds, arms in DATASETS.items():
        data = {k: [metrics(t, s) for s in (0, 1, 2)] for k, t in arms.items()}
        for m in ("pixel IoU", "Boundary IoU", "spurious/img"):
            for a, b in itertools.combinations(sorted(data), 2):
                va = [x[m] for x in data[a]]
                vb = [x[m] for x in data[b]]
                sd = max(st.stdev(va), st.stdev(vb))
                rows.append((ds, m, a, b, abs(st.mean(va) - st.mean(vb)) / sd))
    moves = [r for r in rows if LOW <= r[4] < HIGH]
    print("%d model-pair comparisons" % len(rows))
    print("  same verdict for every multiplier in [%.1f, %.1f]: %d" % (LOW, HIGH, len(rows) - len(moves)))
    print("  verdict depends on the multiplier:                %d" % len(moves))
    for ds, m, a, b, r in sorted(moves, key=lambda x: x[4]):
        print("      %-6s %-13s %-15s vs %-15s %5.2f" % (ds, m, a, b, r))


if __name__ == "__main__":
    main()

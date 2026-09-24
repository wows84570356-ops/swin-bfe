

"""The five scoring conventions of IoU compared in the paper (Sec. III-D), differing in how images
with no building of the target class are scored and at what level the
score is aggregated.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

ARM_LABEL = {
    "rescbam": "reference model (retrained)",
    "unet": "ResNet-34 U-Net",
    "ctxnone": "Swin-BFE",
    "ours": "Swin-BFE-Mamba",
}
READINGS = [
    ("R1", "convention C: per-image macro, absent class scored 1.0 (torchmetrics default)"),
    ("R2", "convention D: per-image macro, empty-GT images dropped"),
    ("R3", "convention E: per-image macro, absent class scored 0.0"),
    ("R4", "convention B: macro computed once over pooled TP/FP/FN/TN"),
    ("R5", "convention A: building class only, pooled (used in the main tables)"),
]


def class_iou(t, f_p, f_n, absent_value):
    d = t + f_p + f_n
    return absent_value if d == 0 else t / d


def macro_per_image(tp, fp, fn, tn, absent_value):
    ib = class_iou(tp, fp, fn, absent_value)
    ig = class_iou(tn, fn, fp, absent_value)
    return 0.5 * (ib + ig)


def load_counts(csv_path, image_px):
    rows = []
    with open(csv_path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            tp, fp, fn = int(r["tp"]), int(r["fp"]), int(r["fn"])
            tn = image_px - tp - fp - fn
            if tn < 0:
                raise SystemExit(f"[FAIL] {csv_path}: tn<0 on {r['stem']}, "
                                 f"--image-px is wrong")
            rows.append((tp, fp, fn, tn))
    return np.asarray(rows, dtype=np.int64)


def five_readings(c):
    tp, fp, fn, tn = c.T
    gt_empty = (tp + fn) == 0
    r1 = np.mean([macro_per_image(*row, absent_value=1.0) for row in c])
    r2 = np.mean([macro_per_image(*row, absent_value=1.0)
                  for row, e in zip(c, gt_empty) if not e])
    r3 = np.mean([macro_per_image(*row, absent_value=0.0) for row in c])
    TP, FP, FN, TN = tp.sum(), fp.sum(), fn.sum(), tn.sum()
    r4 = 0.5 * (TP / (TP + FP + FN) + TN / (TN + FN + FP))
    r5 = TP / (TP + FP + FN)
    return {"R1": r1, "R2": r2, "R3": r3, "R4": r4, "R5": r5,
            "n": len(c), "n_empty": int(gt_empty.sum())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="records")
    ap.add_argument("--arms", nargs="+", default=["rescbam", "unet", "ctxnone", "ours"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--pattern", default="whu_{arm}_a05_s{seed}_eval_test_perimage.csv")
    ap.add_argument("--image-px", type=int, default=1024 * 1024)
    ap.add_argument("--out", default="tables/table8_macro_variants.md")
    a = ap.parse_args()

    per_arm = {}
    for arm in a.arms:
        full, nonempty = [], []
        for s in a.seeds:
            p = Path(a.runs) / a.pattern.format(arm=arm, seed=s)
            if not p.exists():
                raise SystemExit(f"[FAIL] not found: {p}")
            c = load_counts(p, a.image_px)
            full.append(five_readings(c))
            nonempty.append(five_readings(c[(c[:, 0] + c[:, 2]) > 0]))
        per_arm[arm] = {"full": full, "nonempty": nonempty}

    def ms(vals):
        v = np.asarray(vals, float)
        return v.mean(), v.std(ddof=1)

    lines = []
    n, ne = per_arm[a.arms[0]]["full"][0]["n"], per_arm[a.arms[0]]["full"][0]["n_empty"]
    lines.append(f"## Table 8 — The same predictions under five readings of \"IoU\"\n")
    lines.append(f"Test set: {n} images, {ne} ({100*ne/n:.1f}%) with no building in the "
                 f"ground truth. Mean ± SD over seeds {a.seeds}. Every column is the identical "
                 f"set of masks; only the scoring rule changes.\n")
    hdr = "| Definition of IoU | " + " | ".join(ARM_LABEL.get(x, x) for x in a.arms) + " |"
    lines.append(hdr)
    lines.append("|---|" + "---:|" * len(a.arms))
    for key, desc in READINGS:
        cells = []
        for arm in a.arms:
            m, sd = ms([r[key] for r in per_arm[arm]["full"]])
            cells.append(f"{m:.4f} ± {sd:.4f}")
        lines.append(f"| {desc} | " + " | ".join(cells) + " |")

    lines.append("\n**Spread of the seed means (max − min):**\n")
    lines.append("| Subset | Readings | " + " | ".join(ARM_LABEL.get(x, x) for x in a.arms) + " |")
    lines.append("|---|---|" + "---:|" * len(a.arms))
    for subset, label in (("full", f"all {n} images"),
                          ("nonempty", f"only the {n-ne} images that contain a building")):
        for keys, klabel in ((["R1", "R2", "R3", "R4"], "four macro readings R1–R4"),
                             (["R1", "R2", "R3", "R4", "R5"], "all five, incl. building-only R5")):
            cells = []
            for arm in a.arms:
                means = [ms([r[k] for r in per_arm[arm][subset]])[0] for k in keys]
                cells.append(f"{max(means) - min(means):.4f}")
            lines.append(f"| {label} | {klabel} | " + " | ".join(cells) + " |")
    lines.append("\nOn the building-only subset R1 = R2 = R3 by construction (no image lacks the "
                 "building class), so the residual spread is per-image-mean vs pooled aggregation, "
                 "not the empty-image convention.")

    lines.append("\n**Ranking under each reading (1 = best):**\n")
    for key, desc in READINGS:
        means = {arm: ms([r[key] for r in per_arm[arm]["full"]])[0] for arm in a.arms}
        order = sorted(a.arms, key=lambda x: -means[x])
        lines.append(f"- {key}: " + " > ".join(order))

    text = "\n".join(lines) + "\n"
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    out.with_suffix(".json").write_text(json.dumps(per_arm, indent=1), encoding="utf-8")
    try:
        print(text)
    except UnicodeEncodeError:  # consoles without UTF-8 (e.g. cp950 on Windows): the file is still written
        print(text.encode("ascii", "replace").decode("ascii"))
    print(f"wrote {out} and {out.with_suffix('.json')}")


if __name__ == "__main__":
    main()

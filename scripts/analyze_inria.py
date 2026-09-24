

"""Aggregate the Inria results and the pre-registered spread prediction."""

import argparse
import csv
import json
import os

import numpy as np

PX = 1024 * 1024

ARMS = [("rescbam", "Reference architecture (retrained)"),
        ("unet", "U-Net (ResNet-34)"),
        ("ctxnone", "Swin + Conv"),
        ("ours", "Swin + Mamba")]
MACRO = ["E", "B", "A", "C"]
PREDICTION = (0.036, 0.125)
EMPTY_FRAC_REGISTERED = 0.1184


def rows_of(runs, arm, seed, pattern):
    p = os.path.join(runs, pattern.format(arm=arm, seed=seed), "eval_test_perimage.csv")
    return list(csv.DictReader(open(p, encoding="utf-8"))) if os.path.exists(p) else None


def fnum(r, k):
    v = r.get(k, "")
    return 0.0 if v in ("", "nan", None) else float(v)


def metrics(rs):
    TP = sum(fnum(r, "tp") for r in rs)
    FP = sum(fnum(r, "fp") for r in rs)
    FN = sum(fnum(r, "fn") for r in rs)
    TN = len(rs) * PX - TP - FP - FN
    pr = TP / max(TP + FP, 1); rc = TP / max(TP + FN, 1)
    ngt = sum(fnum(r, "n_gt") for r in rs)
    found = sum(fnum(r, "found") for r in rs)
    bio = [float(r["boundary_iou"]) for r in rs
           if r.get("boundary_iou") not in ("", "nan", None)]
    cerr = [fnum(r, "n_pred_inst") - fnum(r, "n_gt") for r in rs if fnum(r, "n_gt") > 0]
    return dict(
        IoU=TP / max(TP + FP + FN, 1), F1=2 * pr * rc / max(pr + rc, 1e-9),
        precision=pr, recall=rc, accuracy=(TP + TN) / max(TP + TN + FP + FN, 1),
        inst_recall=found / max(ngt, 1),
        merges=sum(fnum(r, "merges") for r in rs) / max(found, 1),
        splits=sum(fnum(r, "splits") for r in rs) / max(found, 1),
        fp_per_img=sum(fnum(r, "fp_inst") for r in rs) / max(len(rs), 1),
        count_mae=float(np.mean(np.abs(cerr))) if cerr else float("nan"),
        count_bias=float(np.mean(cerr)) if cerr else float("nan"),
        boundary_iou=float(np.mean(bio)) if bio else float("nan"),
        n=len(rs))


def macro_readings(rs):
    out = {}
    counts = [(fnum(r, "tp"), fnum(r, "fp"), fnum(r, "fn")) for r in rs]
    for tag in MACRO:
        if tag == "C":
            TP = sum(c[0] for c in counts); FP = sum(c[1] for c in counts)
            FN = sum(c[2] for c in counts); TN = len(counts) * PX - TP - FP - FN
            out[tag] = 0.5 * (TP / (TP + FP + FN) + TN / (TN + FP + FN))
            continue
        m = []
        for tp, fp, fn in counts:
            if tag == "B" and (tp + fn) == 0:
                continue
            tn = PX - tp - fp - fn
            ib = (1.0 if tag == "E" else 0.0) if (tp + fp + fn) == 0 else tp / (tp + fp + fn)
            ig = tn / (tn + fp + fn) if (tn + fp + fn) else 1.0
            m.append(0.5 * (ib + ig))
        out[tag] = float(np.mean(m))
    return out


def agg(vals):
    a = np.asarray(vals, float)
    return float(a.mean()), (float(a.std(ddof=1)) if len(a) > 1 else 0.0)


def table(rows, header):
    out = ["| " + " | ".join(header) + " |",
           "|" + "---|" * len(header)]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs/inria")
    ap.add_argument("--pattern", default="inria_{arm}_a05_s{seed}")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--splits", default="splits/inria_split.json")
    ap.add_argument("--out", default="tables")


    ap.add_argument("--supp",
                    default="tables/S5_inria_results.md",
                    help="where to write the supplementary note; 'none' to skip. Use 'none' "
                         "for exploratory runs so a partial-seed analysis cannot replace the "
                         "submitted note.")
    a = ap.parse_args()

    nooverlap = set(json.load(open(a.splits))["test_nooverlap"])
    present = [(arm, lbl) for arm, lbl in ARMS
               if any(rows_of(a.runs, arm, s, a.pattern) for s in a.seeds)]
    if not present:
        raise SystemExit("[FAIL] no finished Inria runs under " + a.runs)
    print("arms found:", [x[0] for x in present])

    M, MN, R = {}, {}, {}
    for arm, _ in present:
        per_seed, per_seed_no, per_seed_rd = [], [], []
        for s in a.seeds:
            rs = rows_of(a.runs, arm, s, a.pattern)
            if rs is None:
                print("  missing: %s seed %d" % (arm, s)); continue
            per_seed.append(metrics(rs))
            per_seed_no.append(metrics([r for r in rs if r["stem"] in nooverlap]))
            per_seed_rd.append(macro_readings(rs))
        M[arm] = {k: agg([d[k] for d in per_seed]) for k in per_seed[0]}
        MN[arm] = {k: agg([d[k] for d in per_seed_no]) for k in per_seed_no[0]}
        R[arm] = {k: agg([d[k] for d in per_seed_rd]) for k in MACRO}

    empty = sum(1 for r in rows_of(a.runs, present[0][0], a.seeds[0], a.pattern)
                if fnum(r, "n_gt") == 0)
    n_all = M[present[0][0]]["n"][0]
    print("test tiles %d, empty %d (%.2f%%)" % (n_all, empty, 100 * empty / n_all))

    lines = ["# Inria results (generated by analyze_inria.py)", ""]
    lines.append("Mean ± SD over seeds %s. Definitions identical to the WHU tables." % a.seeds)
    lines.append("")
    lines.append("## Pixel level")
    lines += table([[lbl] + ["%.4f ± %.4f" % M[arm][k] for k in
                             ("IoU", "F1", "precision", "recall", "accuracy")]
                    for arm, lbl in present],
                   ["Method", "IoU", "F1", "Precision", "Recall", "Accuracy"])
    lines.append("")
    lines.append("## Instance and boundary level")
    lines += table([[lbl] + ["%.4f ± %.4f" % M[arm]["inst_recall"],
                             "%.4f ± %.4f" % M[arm]["merges"],
                             "%.4f ± %.4f" % M[arm]["splits"],
                             "%.2f ± %.2f" % M[arm]["fp_per_img"],
                             "%.2f ± %.2f" % M[arm]["count_mae"],
                             "%+.2f ± %.2f" % M[arm]["count_bias"],
                             "%.4f ± %.4f" % M[arm]["boundary_iou"]]
                    for arm, lbl in present],
                   ["Method", "Instance recall ↑", "Merges ↓", "Splits ↓",
                    "Spurious buildings per image ↓", "Count MAE ↓", "Count bias",
                    "Boundary IoU ↑"])
    lines.append("")
    lines.append("Merge rate is raw, with no floor subtracted: the Inria ground-truth floor is "
                 "0.0000 because Inria has no per-building polygons and the instance map is "
                 "derived from the binary label by connected components. An Inria instance is a "
                 "connected blob, a WHU instance is an annotated building; the two columns are "
                 "not comparable and must not be placed in one table.")
    lines.append("")
    lines.append("## No-overlap control (400 tiles, rows/cols 0-3)")
    lines += table([[lbl, "%.4f" % M[arm]["IoU"][0], "%.4f" % MN[arm]["IoU"][0],
                     "%+.4f" % (MN[arm]["IoU"][0] - M[arm]["IoU"][0])]
                    for arm, lbl in present],
                   ["Method", "IoU (all 625)", "IoU (no overlap, 400)", "Change"])
    lines.append("")
    lines.append("## The four macro readings")
    names = {"E": "per-image macro, absent class = 1", "B": "per-image macro, empty dropped",
             "A": "per-image macro, absent class = 0", "C": "pooled macro"}
    lines += table([[names[k]] + ["%.4f" % R[arm][k][0] for arm, _ in present]
                    for k in MACRO],
                   ["Definition of IoU"] + [lbl for _, lbl in present])
    lines.append("")
    spreads = {arm: max(R[arm][k][0] for k in MACRO) - min(R[arm][k][0] for k in MACRO)
               for arm, _ in present}
    lines.append("Spread per arm: " + ", ".join("%s %.4f" % (arm, v)
                                                for arm, v in spreads.items()))
    mean_spread = float(np.mean(list(spreads.values())))
    lo, hi = PREDICTION
    ok = lo < mean_spread < hi
    verdict = "PASS" if ok else "FALSIFIED"
    lines.append("")
    lines.append("## Check of the registered prediction")
    lines.append("")
    lines.append("Registered in the manuscript on 2026-09-05, before any Inria model existed: "
                 "the Inria test partition is %.2f%% empty, between HUMG (0.26%%) and "
                 "WHU (24.1%%), so if the spread is governed by the empty-image fraction "
                 "it must fall between %.3f and %.3f." % (100 * EMPTY_FRAC_REGISTERED, lo, hi))
    lines.append("")
    lines.append("Measured mean spread: **%.4f** → **%s**" % (mean_spread, verdict))
    if not ok:
        lines.append("")
        lines.append("The prediction is falsified. Report this as a finding: the mechanism "
                     "as stated does not account for the Inria measurement, and the claim "
                     "in the Discussion must be weakened accordingly.")

    os.makedirs(a.out, exist_ok=True)
    p = os.path.join(a.out, "inria_results.md")
    open(p, "w", encoding="utf-8").write("\n".join(lines) + "\n")


    supp = a.supp
    intro = [
        "# Full results on the Inria benchmark",
        "",
        "The main text reports the headline Inria numbers; this section gives the full "
        "tables. Definitions are identical to the WHU tables and are set out in "
        "Supplementary Note 1, with one exception that matters and is repeated below: "
        "instance-level numbers are not comparable between the two datasets, because Inria "
        "releases no per-building polygons.",
        "",
        "Generated by `code/scripts/analyze_inria.py` from the deposited per-image records.",
        "",
    ]
    if str(supp).lower() == "none":
        print("supplementary note not written (--supp none)")
    elif not os.path.isdir(os.path.dirname(supp)):
        print("[WARN] supplementary directory missing, note NOT written: "
              + os.path.dirname(supp))
    else:
        open(supp, "w", encoding="utf-8").write("\n".join(intro + lines[2:]) + "\n")
        print("wrote " + supp)
    json.dump({"pixel_instance": {k: {m: list(v) for m, v in M[k].items()} for k in M},
               "nooverlap": {k: {m: list(v) for m, v in MN[k].items()} for k in MN},
               "macro": {k: {m: list(v) for m, v in R[k].items()} for k in R},
               "spread": spreads, "mean_spread": mean_spread,
               "prediction": {"interval": list(PREDICTION), "verdict": verdict}},
              open(os.path.join(a.out, "inria_results.json"), "w"), indent=1)
    print("\n".join(lines[-8:]))
    print("\nwrote " + p)


if __name__ == "__main__":
    main()

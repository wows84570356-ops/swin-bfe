"""Paired image-level bootstrap over the per-image records (Table III).

The test images are resampled with replacement, the same image indices are
applied to both arms, and the pooled building-class IoU is recomputed inside
each resample. The interval is the 2.5th-97.5th percentile of the resampled
differences. Model weights are fixed throughout, so the interval reflects
test-set sampling only; retraining variation is handled by the seed rule.

    python scripts/paired_stats.py --base rescbam --arm ctxnone
    python scripts/paired_stats.py --base ctxnone --arm ours --dataset inria
"""

import argparse
import csv
from pathlib import Path

import numpy as np


def load(path):
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    stems = [r["stem"] for r in rows]
    c = np.array([[int(r["tp"]), int(r["fp"]), int(r["fn"])] for r in rows],
                 dtype=np.int64)
    return stems, c


def pooled_iou(c, idx):
    tp, fp, fn = c[idx].sum(0)
    return tp / max(tp + fp + fn, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="records")
    ap.add_argument("--dataset", default="whu", choices=["whu", "inria"])
    ap.add_argument("--base", required=True, help="run tag of the baseline, e.g. rescbam")
    ap.add_argument("--arm", required=True, help="run tag of the arm, e.g. ctxnone")
    ap.add_argument("--seed", type=int, default=0, help="training seed of both checkpoints")
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--rng", type=int, default=0, help="seed of the resampling")
    a = ap.parse_args()

    def path(tag):
        return Path(a.dir) / f"{a.dataset}_{tag}_a05_s{a.seed}_eval_test_perimage.csv"

    stems_a, ca = load(path(a.arm))
    stems_b, cb = load(path(a.base))
    if stems_a != stems_b:
        raise SystemExit("[FAIL] the two records list different images")
    n = len(stems_a)
    full = np.arange(n)
    va, vb = pooled_iou(ca, full), pooled_iou(cb, full)
    rng = np.random.default_rng(a.rng)
    diffs = np.empty(a.boot)
    for k in range(a.boot):
        idx = rng.integers(0, n, n)
        diffs[k] = pooled_iou(ca, idx) - pooled_iou(cb, idx)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    print(f"{a.dataset} seed {a.seed}: {a.arm} {va:.4f} vs {a.base} {vb:.4f} | "
          f"delta {va - vb:+.4f} | 95% interval [{lo:+.4f}, {hi:+.4f}] | "
          f"{a.boot} resamples of {n} images")


if __name__ == "__main__":
    main()

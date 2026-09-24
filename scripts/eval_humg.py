
"""Run the reference architecture's released weights on its own test set.

The question is whether its published IoU and F1 reproduce. Four choices
differ from eval_whu.py, each to remove a confound: no tiling or
feathering, because the images are natively 512x512 and match the
training size; fp32 rather than bf16 autocast, because the released
notebook is fp32; pixel-level metrics only, because the released masks
are binary and have no instance map.
"""

import argparse
import csv
import json
import math
import os
import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)


def build_official(ckpt_path, device):
    from swinbfe.rescbam_unet import ResCBAMUNetWrapped
    net = ResCBAMUNetWrapped().to(device)
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "model" in sd and isinstance(sd["model"], dict):
        sd = sd["model"]
    sd = {(k if k.startswith("net.") else "net." + k): v for k, v in sd.items()}
    net.load_state_dict(sd, strict=True)
    net.eval()
    return net


@torch.no_grad()
def predict_single(net, rgb, device):
    x = (rgb.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    t = torch.from_numpy(x.transpose(2, 0, 1)[None].copy()).to(device)
    out = net(t)
    return torch.sigmoid(out["logits"].float())[0, 0].cpu().numpy()


def macro_per_image(tp, fp, fn, tn, absent_value=1.0):
    def cls(t, f_p, f_n):
        d = t + f_p + f_n
        if d == 0:
            return absent_value, absent_value
        iou = t / d
        p = t / (t + f_p) if (t + f_p) else 0.0
        r = t / (t + f_n) if (t + f_n) else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        return iou, f1
    ib, fb = cls(tp, fp, fn)
    ig, fg = cls(tn, fn, fp)
    return (ib + ig) / 2.0, (fb + fg) / 2.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--root", default="data/humg/test_dataset")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"],
                    help="default cpu: 1545 images of 512 by 512 take about 37 minutes and never touch the GPU")
    ap.add_argument("--threads", type=int, default=4,
                    help="cap the CPU threads so this does not compete with a running training job")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="runs/ref_official/eval_humg.json")
    a = ap.parse_args()

    if a.device == "cpu":
        torch.set_num_threads(a.threads)

    img_dir = os.path.join(a.root, "images")
    lab_dir = os.path.join(a.root, "labels")
    names = sorted(os.listdir(img_dir))
    if a.limit:
        names = names[:a.limit]

    net = build_official(a.ckpt, a.device)
    print("loaded the official weights (strict=True); %d images; device=%s; fp32, one forward pass, no tiling"
          % (len(names), a.device), flush=True)

    TP = FP = FN = TN = 0
    macro_i, macro_f, macro_i0 = [], [], []
    rows = []
    n_empty = 0
    for k, n in enumerate(names, 1):
        bgr = cv2.imread(os.path.join(img_dir, n), cv2.IMREAD_COLOR)
        if bgr is None:
            raise SystemExit("[FAIL] cannot read image %s" % n)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        lab = cv2.imread(os.path.join(lab_dir, n), cv2.IMREAD_GRAYSCALE)
        if lab is None:
            raise SystemExit("[FAIL] cannot read label %s" % n)
        gt = lab == 255

        prob = predict_single(net, rgb, a.device)
        pred = prob > 0.5

        tp = int(np.logical_and(pred, gt).sum())
        fp = int(np.logical_and(pred, ~gt).sum())
        fn = int(np.logical_and(~pred, gt).sum())
        tn = int(gt.size - tp - fp - fn)
        TP += tp; FP += fp; FN += fn; TN += tn
        if not gt.any():
            n_empty += 1

        mi, mf = macro_per_image(tp, fp, fn, tn, absent_value=1.0)
        mi0, _ = macro_per_image(tp, fp, fn, tn, absent_value=0.0)
        macro_i.append(mi); macro_f.append(mf); macro_i0.append(mi0)
        rows.append((n, tp, fp, fn, tn, mi, mf))

        if k % 200 == 0:
            print("  %d/%d" % (k, len(names)), flush=True)

    iou_b = TP / max(TP + FP + FN, 1)
    prec = TP / max(TP + FP, 1)
    rec = TP / max(TP + FN, 1)
    f1_b = 2 * prec * rec / max(prec + rec, 1e-9)
    iou_g = TN / max(TN + FN + FP, 1)
    acc = (TP + TN) / max(TP + TN + FP + FN, 1)

    res = dict(
        ckpt=a.ckpt, n=len(names), n_empty=n_empty, device=a.device,
        tiling="none(single forward)", precision="fp32",
        building_IoU=iou_b, building_F1=f1_b, precision_=prec, recall=rec,
        accuracy=acc,
        macro_IoU_pooled=(iou_b + iou_g) / 2.0,
        macro_IoU_perimage_absent1=float(np.mean(macro_i)),
        macro_F1_perimage_absent1=float(np.mean(macro_f)),
        macro_IoU_perimage_absent0=float(np.mean(macro_i0)),
        ref_selfreport_IoU=0.900, ref_selfreport_F1=0.936,
    )

    print()
    print("=" * 72)
    print("reference work's self-reported values (HUMG): IoU 0.900 / F1 0.936")
    print("=" * 72)
    print("per-image macro, absent class scored 1.0 (their reading)  IoU %.4f   F1 %.4f"
          % (res["macro_IoU_perimage_absent1"], res["macro_F1_perimage_absent1"]))
    print("per-image macro, absent class scored 0 (sensitivity)      IoU %.4f"
          % res["macro_IoU_perimage_absent0"])
    print("pooled macro                                             IoU %.4f" % res["macro_IoU_pooled"])
    print("building class only (the reading used here)               IoU %.4f   F1 %.4f"
          % (iou_b, f1_b))
    print("Precision %.4f  Recall %.4f  Accuracy %.4f" % (prec, rec, acc))
    print("empty images %d / %d = %.2f%%" % (n_empty, len(names), 100 * n_empty / len(names)))
    d = res["macro_IoU_perimage_absent1"] - 0.900
    print()
    print("difference from the self-reported value: %+.4f IoU" % d)

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    with open(a.out.replace(".json", "_perimage.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name", "tp", "fp", "fn", "tn", "macro_iou_absent1", "macro_f1_absent1"])
        w.writerows(rows)
    print("wrote", a.out)


if __name__ == "__main__":
    main()


"""Evaluate on WHU at pixel, instance and boundary granularity.

Three points decide whether the numbers are meaningful:
  1. The encoder binds its input size, so 1024x1024 test images are
     scored from 512x512 tiles with 64 pixels of overlap, blended with a
     separable cosine window that is strictly positive everywhere. A
     pixel covered by a single tile therefore recovers that tile's
     probability exactly, and no seam is left at tile borders.
  2. Test tiles overlap by about 19.5 percent, so 2,220 images are not
     2,220 independent samples. --thin scores a spatially thinned subset
     as well; the paper reports both.
  3. Instance merges have a floor of about 2 percent because the ground
     truth itself contains abutting buildings that connected components
     join. The reported merge rate is net of that floor.
"""

import argparse, csv, json, os, sys
import numpy as np
import cv2
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from swinbfe.net import SwinBFE
from swinbfe import baselines
from swinbfe.dataset import resolve_path

# Checkpoints written while this code was being developed store the model
# under its former name. Both names select the same architecture, so old and
# new checkpoints load identically.
MODEL_ALIASES = ("swinbfe", "cracknet")
LEGACY_MODEL_NAME = "cracknet"


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)


def build_net(ck, img_size, device):
    a = ck.get("args", {})
    model = a.get("model", LEGACY_MODEL_NAME)
    if model in MODEL_ALIASES:
        net = SwinBFE(img_size=img_size,
                       context=a.get("context", "v2"),
                       pretrained=False)
    elif model in baselines.UNET_ALIASES:
        net = baselines.UNet(pretrained=False)
    elif model == "rescbam":
        from swinbfe.rescbam_unet import ResCBAMUNetWrapped
        net = ResCBAMUNetWrapped()
    else:
        raise SystemExit("unknown model: %s" % model)
    sd = ck["model"] if "model" in ck else ck["state_dict"]
    missing, unexpected = net.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print("  loaded weights: missing %d, unexpected %d" % (len(missing), len(unexpected)))
    return net.to(device).eval()


def _cos_window(t, ov):
    w = np.ones(t, np.float32)
    if ov > 0:
        r = 0.5 * (1 - np.cos(np.linspace(0, np.pi, ov + 2)[1:-1]))
        w[:ov] *= r
        w[-ov:] *= r[::-1]
    return w[:, None] * w[None, :]


@torch.no_grad()
def predict(net, rgb, device, tile=512, overlap=64, amp=True):
    H, W = rgb.shape[:2]
    x = (rgb.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    acc = np.zeros((H, W), np.float32)
    wsum = np.zeros((H, W), np.float32)
    win = _cos_window(tile, overlap)
    step = tile - overlap
    ys = list(range(0, max(H - tile, 0) + 1, step)) or [0]
    xs = list(range(0, max(W - tile, 0) + 1, step)) or [0]
    if ys[-1] + tile < H: ys.append(H - tile)
    if xs[-1] + tile < W: xs.append(W - tile)
    for y in ys:
        for xx in xs:
            patch = x[y:y + tile, xx:xx + tile]
            ph, pw = patch.shape[:2]
            if (ph, pw) != (tile, tile):
                patch = np.pad(patch, ((0, tile - ph), (0, tile - pw), (0, 0)), mode="reflect")
            t = torch.from_numpy(patch.transpose(2, 0, 1)[None].copy()).to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                out = net(t)
            p = torch.sigmoid(out["logits"].float())[0, 0].cpu().numpy()
            acc[y:y + ph, xx:xx + pw] += p[:ph, :pw] * win[:ph, :pw]
            wsum[y:y + ph, xx:xx + pw] += win[:ph, :pw]


    return acc / np.maximum(wsum, 1e-12)


@torch.no_grad()
def predict_resized(net, rgb, device, size=512, amp=True):
    H, W = rgb.shape[:2]
    small = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_LINEAR)
    x = (small.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    t = torch.from_numpy(x.transpose(2, 0, 1)[None].copy()).to(device)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
        out = net(t)
    p = torch.sigmoid(out["logits"].float())[0, 0].cpu().numpy()
    return cv2.resize(p, (W, H), interpolation=cv2.INTER_LINEAR)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--index", default="splits/whu_index.json")
    ap.add_argument("--splitjson", default="splits/whu_split.json")
    ap.add_argument("--data-root", default=None,
                    help="directory the index paths are relative to; "
                         "defaults to $SWINBFE_DATA_ROOT or the current directory")
    ap.add_argument("--thin", action="store_true",
                    help="spatial thinning: keep every second tile, to reduce the effect of the 19.5%% overlap inside the test split")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--tile", type=int, default=512)
    ap.add_argument("--rgb-override", default=None,
                    help="use the same-stem images under this directory instead. "
                         "Abort if any is missing, rather than falling back silently.")
    ap.add_argument("--arch", default="auto",
                    choices=["auto", "rescbam_official"],
                    help="auto: decide from the checkpoint contents; rescbam_official: the bare state_dict released by the reference work.")
    ap.add_argument("--resize-eval", action="store_true",
                    help="resize the whole image to --tile and run one forward pass, scaling the probability map back. "
                         "Required for models trained with --ref-proto: otherwise the test scale "
                         "differs from the training scale by a factor of two, and what is measured is scale mismatch rather than model quality.")
    a = ap.parse_args()

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from inst_metrics import instance_stats, boundary_iou, aggregate

    index = json.load(open(a.index))
    stems = json.load(open(a.splitjson))[a.split]
    if a.thin:
        stems = stems[::2]
    if a.limit:
        stems = stems[:a.limit]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    if a.arch == "rescbam_official" or ("args" not in ck and "model" not in ck):

        from swinbfe.rescbam_unet import ResCBAMUNetWrapped
        net = ResCBAMUNetWrapped().to(device)
        sd = {(k if k.startswith("net.") else "net." + k): v
              for k, v in ck.items()}
        net.load_state_dict(sd, strict=True)
        net.eval()
        print("loaded the reference work's official weights (bare state_dict, net. prefix added, strict=True)")
    else:
        net = build_net(ck, a.tile, device)
    print("evaluating %s: %d images (%s)" % (os.path.basename(os.path.dirname(a.ckpt)),
                                   len(stems), "thinned" if a.thin else "all"))

    TP = FP = FN = 0
    rows, bious, per_img = [], [], []
    for k, s in enumerate(stems, 1):
        rec = index[s]
        rgb_path = resolve_path(rec["rgb"], a.data_root)
        if a.rgb_override:
            base = os.path.basename(rgb_path)
            cand = os.path.join(a.rgb_override, base)
            if not os.path.exists(cand):
                import glob as _glob
                hits = _glob.glob(os.path.join(
                    a.rgb_override, os.path.splitext(base)[0] + ".*"))
                cand = hits[0] if hits else None
            if cand is None:
                raise SystemExit(
                    "[FAIL] --rgb-override has no image matching %s (%s). "
                    "Aborting rather than silently using the original and producing a false result." % (s, a.rgb_override))
            rgb_path = cand
        img_bgr = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise SystemExit("[FAIL] cannot read image: %s" % rgb_path)
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        gt = cv2.imread(resolve_path(rec["label"], a.data_root),
                        cv2.IMREAD_GRAYSCALE) > 127
        inst = cv2.imread(resolve_path(rec["inst"], a.data_root),
                          cv2.IMREAD_UNCHANGED)
        prob = (predict_resized(net, rgb, device, size=a.tile)
                if a.resize_eval else predict(net, rgb, device, tile=a.tile))
        pred = prob > 0.5
        tp = int(np.logical_and(pred, gt).sum())
        fp = int(np.logical_and(pred, ~gt).sum())
        fn = int(np.logical_and(~pred, gt).sum())
        TP += tp; FP += fp; FN += fn


        st, bi = None, float('nan')
        if inst is not None:
            st = instance_stats(pred, inst)
            rows.append(st)
            if inst.max() > 0:
                bi = boundary_iou(pred, gt)
                bious.append(bi)
        per_img.append((s, tp, fp, fn,
                        st['n_gt'] if st else 0, st['found'] if st else 0,
                        st['merges'] if st else 0, st['splits'] if st else 0,
                        st['fp'] if st else 0, st['n_pred_inst'] if st else 0, bi))
        if k % 200 == 0:
            print("  %d/%d" % (k, len(stems)), flush=True)

    iou = TP / max(TP + FP + FN, 1)
    prec = TP / max(TP + FP, 1)
    rec = TP / max(TP + FN, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    ag = aggregate(rows) if rows else {}

    print()
    print("=" * 70)
    print("Pixel level (pooled, the same metric set as the reference work)")
    print("  IoU %.4f   F1 %.4f   Precision %.4f   Recall %.4f" % (iou, f1, prec, rec))
    print()
    print("Instance level (not reported by the reference work)")
    for k_ in ("instance_recall", "merges_per_found", "splits_per_found",
               "fp_per_img", "count_mae", "count_bias"):
        if k_ in ag:
            print("  %-18s %.4f" % (k_, ag[k_]))
    print("  %-18s %.4f" % ("boundary_iou", float(np.mean(bious)) if bious else float("nan")))
    print()
    print("  note: the ground-truth merge floor is about 0.0198, because connected components join")
    print("        abutting buildings; subtract it or state it explicitly.")
    if not a.thin:
        print("  note: test tiles overlap by about 19.5%, so 2220 images are not 2220 independent samples;")
        print("        add --thin for the spatially thinned numbers.")

    out = a.out or (os.path.dirname(a.ckpt) + "/eval_%s%s%s.json"
                    % (a.split, "_thin" if a.thin else "",
                       "_resize" if a.resize_eval else ""))
    json.dump(dict(ckpt=a.ckpt, split=a.split, thin=a.thin, n=len(stems),
                   rgb_override=a.rgb_override, arch=a.arch,
                   IoU=iou, F1=f1, precision=prec, recall=rec,
                   boundary_iou=float(np.mean(bious)) if bious else None, **ag),
              open(out, "w"), indent=1)
    with open(out.replace(".json", "_perimage.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["stem", "tp", "fp", "fn", "n_gt", "found", "merges",
                    "splits", "fp_inst", "n_pred_inst", "boundary_iou"])
        w.writerows(per_img)
    print("\nwrote", out)


if __name__ == "__main__":
    main()

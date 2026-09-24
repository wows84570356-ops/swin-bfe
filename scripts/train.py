"""Train one arm for one seed.

Every arm in the paper uses identical settings apart from --context and
--model: AdamW at learning rate 1e-4 and weight decay 1e-2, batch size 4
with gradient accumulation 2, 512x512 random crops, 20,000 iterations,
bf16 autocast and ImageNet normalisation. Those are the defaults below,
so `python scripts/train.py --context none --seed 0` reproduces the
Swin-BFE run for seed 0 once --data-root points at the data.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import csv
import math
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from swinbfe.baselines import UNET_ALIASES, build_baseline
from swinbfe.dataset import TileDataset
from swinbfe.losses import compute_losses
from swinbfe.net import SwinBFE

# Checkpoints written while this code was being developed store the model
# under its former name. Both names select the same architecture, so old and
# new checkpoints load identically.
MODEL_ALIASES = ("swinbfe", "cracknet")


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def lr_at(it, base, iters, warmup=500):
    if it < warmup:
        return base * (it + 1) / warmup
    t = (it - warmup) / max(iters - warmup, 1)
    return base * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * t)))


@torch.no_grad()
def validate(net, loader, device, use_amp):
    net.eval()
    tp = fp = fn = 0
    for b in loader:
        x = b["image"].to(device, non_blocking=True)
        g = b["mask"].to(device) > 0.5
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            out = net(x)
        p = torch.sigmoid(out["logits"].float()) > 0.5
        tp += (p & g).sum().item()
        fp += (p & ~g).sum().item()
        fn += (~p & g).sum().item()
    net.train()
    return {"iou": tp / max(tp + fp + fn, 1),
            "f1": 2 * tp / max(2 * tp + fp + fn, 1)}


def build_model(a):
    if a.model in MODEL_ALIASES:
        return SwinBFE(encoder=a.encoder, img_size=a.crop, context=a.context,
                       pretrained=not a.no_pretrained)
    if a.model == "rescbam":
        from swinbfe.rescbam_unet import ResCBAMUNetWrapped
        return ResCBAMUNetWrapped()
    return build_baseline(a.model, pretrained=not a.no_pretrained)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default=None,
                    help="run directory under runs/; default <model or context>_s<seed>")
    ap.add_argument("--context", choices=["none", "v2", "vanilla"], default="none",
                    help="context block: none = Swin-BFE, v2 = Swin-BFE-Mamba, vanilla = scan control")
    ap.add_argument("--model", choices=list(MODEL_ALIASES) + list(UNET_ALIASES) + ["rescbam"],
                    default="swinbfe")
    ap.add_argument("--encoder", default="swin_small_patch4_window7_224.ms_in22k")
    ap.add_argument("--iters", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--crop", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--wd", type=float, default=1e-2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--ckpt-every", type=int, default=10000,
                    help="periodic checkpoint interval; 0 disables it")
    ap.add_argument("--index", default="splits/whu_index.json")
    ap.add_argument("--split", default="splits/whu_split.json")
    ap.add_argument("--data-root", default=None,
                    help="directory the index paths are relative to; "
                         "defaults to $SWINBFE_DATA_ROOT or the current directory")
    ap.add_argument("--rgb-override", default=None,
                    help="replace the RGB image with the same-stem file in this directory")
    ap.add_argument("--tversky-alpha", type=float, default=0.5)
    ap.add_argument("--tversky-beta", type=float, default=0.5)
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--ref-proto", action="store_true",
                    help="train under the reference architecture's own published recipe: "
                         "whole-image resize to --crop, colour jitter and 15-degree rotation, "
                         "cross-entropy only, no gradient accumulation")
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--resume", default=None)
    ap.add_argument("--smoke", action="store_true",
                    help="1,000 iterations with validation every 250, for checking the setup")
    a = ap.parse_args()
    if a.smoke:
        a.iters, a.eval_every = 1000, 250
    if a.ref_proto:
        a.accum = 1
    if a.name is None:
        tag = a.context if a.model in MODEL_ALIASES else a.model
        a.name = f"{'smoke_' if a.smoke else ''}{tag}_s{a.seed}"

    set_seed(a.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = (device == "cuda") and not a.no_amp
    out_dir = Path("runs") / a.name
    out_dir.mkdir(parents=True, exist_ok=True)

    common = dict(crop=a.crop, rgb_override=a.rgb_override,
                  ref_proto=a.ref_proto, data_root=a.data_root)
    tr = TileDataset(a.index, a.split, "train", seed=a.seed, **common)
    va = TileDataset(a.index, a.split, "val", **common)
    tl = DataLoader(tr, batch_size=a.batch, shuffle=True, drop_last=True,
                    num_workers=a.workers, pin_memory=True,
                    persistent_workers=a.workers > 0)
    vl = DataLoader(va, batch_size=max(a.batch, 2), num_workers=2)
    print(f"train {len(tr)} | val {len(va)} | device {device} | "
          f"model {a.model} | context {a.context} | crop {a.crop}"
          f"{' | REF-PROTO(resize+jitter+rot15, BCE only)' if a.ref_proto else ''}")

    net = build_model(a).to(device)
    nparam = sum(p.numel() for p in net.parameters())
    print(f"params {nparam/1e6:.2f}M")
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=a.wd)

    start_it, best = 0, -1.0
    if a.resume:
        ck = torch.load(a.resume, map_location="cpu", weights_only=False)
        net.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        start_it, best = ck["iter"], ck.get("best", -1.0)
        print(f"resumed @ {start_it}")

    log_path = out_dir / "log.csv"
    if not log_path.exists():
        with open(log_path, "w", newline="") as f:
            csv.writer(f).writerow(
                ["iter", "total", "bce", "tversky", "lr", "iou", "f1"])

    it = start_it
    nskip_l = nskip_g = 0
    t0, seen = time.time(), 0
    net.train()
    opt.zero_grad(set_to_none=True)
    run = {}
    while it < a.iters:
        for b in tl:
            if it >= a.iters:
                break
            lr = lr_at(it, a.lr, a.iters)
            for gp in opt.param_groups:
                gp["lr"] = lr
            x = b["image"].to(device, non_blocking=True)
            tgt = {"mask": b["mask"].to(device, non_blocking=True)}
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                out = {k: v.float() for k, v in net(x).items()}
                L = compute_losses(out, tgt, a.tversky_alpha, a.tversky_beta,
                                   tversky_weight=0.0 if a.ref_proto else 1.0)
            if not torch.isfinite(L["total"]):
                nskip_l += 1
                if nskip_l == 1 or nskip_l % 100 == 0:
                    print(f"[warn] non-finite loss (occurrence {nskip_l}, it={it})")
                opt.zero_grad(set_to_none=True)
                it += 1
                continue
            (L["total"] / a.accum).backward()
            if (it + 1) % a.accum == 0:
                gn = torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                if torch.isfinite(gn):
                    opt.step()
                else:
                    nskip_g += 1
                opt.zero_grad(set_to_none=True)
            for k, v in L.items():
                run[k] = run.get(k, 0.0) + float(v.detach())
            seen += 1
            it += 1

            if it % 50 == 0:
                ips = seen / (time.time() - t0)
                eta = (a.iters - it) / max(ips, 1e-9) / 3600
                msg = " ".join(f"{k}={run[k]/seen:.3f}"
                               for k in ("total", "bce", "tversky"))
                skips = (f" | skip(loss/grad) {nskip_l}/{nskip_g}"
                         if (nskip_l or nskip_g) else "")
                print(f"[{it}/{a.iters}] {msg} lr={lr:.2e} "
                      f"{ips:.2f}it/s ETA {eta:.1f}h{skips}")

            if it % a.eval_every == 0 or it == a.iters:
                m = validate(net, vl, device, use_amp)
                print(f"  == val @ {it}: iou {m['iou']:.4f} | f1 {m['f1']:.4f}")
                with open(log_path, "a", newline="") as f:
                    csv.writer(f).writerow(
                        [it] + [f"{run[k]/seen:.4f}"
                                for k in ("total", "bce", "tversky")] +
                        [f"{lr:.2e}", f"{m['iou']:.4f}", f"{m['f1']:.4f}"])
                ck = {"model": net.state_dict(), "opt": opt.state_dict(),
                      "iter": it, "best": best, "args": vars(a)}
                torch.save(ck, out_dir / "last.pt")
                if a.ckpt_every and it % a.ckpt_every == 0:
                    torch.save(ck, out_dir / f"it{it:06d}.pt")
                if m["iou"] > best:
                    best = m["iou"]
                    ck["best"] = best
                    torch.save(ck, out_dir / "best.pt")
                    print(f"  ** best iou {best:.4f} -> best.pt")
                run, seen, t0 = {}, 0, time.time()

    if nskip_l or nskip_g:
        print(f"[warn] skipped non-finite loss or gradient on {nskip_l}/{nskip_g} steps")
    if best < 0:
        print("[FAIL] no valid step at all: the first batch was already non-finite.")
    print(f"done. best iou {best:.4f} | checkpoints in {out_dir}/")


if __name__ == "__main__":
    main()

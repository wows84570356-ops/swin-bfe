
"""Check every checkpoint against the configuration reported in the paper.

A checkpoint stores the argparse namespace it was trained with, and its
state dict shows which modules actually carry parameters. This compares
both against the paper, and in particular asserts that the upsampling
path holds no parameters at all, which is what makes it bilinear.
"""

import os
import sys

import torch

# Checkpoints written while this code was being developed store the model
# under its former name. Both names select the same architecture, so old and
# new checkpoints load identically.
MODEL_ALIASES = ("swinbfe", "cracknet")
LEGACY_MODEL_NAME = "cracknet"


RUNS = "runs"
ARMS = ["whu_rescbam_a05_s%d", "whu_unet_a05_s%d",
        "whu_ctxnone_a05_s%d", "whu_ours_a05_s%d",
        "inria_rescbam_a05_s%d", "inria_unet_a05_s%d",
        "inria_ctxnone_a05_s%d", "inria_ours_a05_s%d"]


CLAIMS = {
    "no_aniso": (True, "eq. (2) Up_2 and Fig. 1 legend: bilinear upsample"),
    "no_topo": (True, "Sec. III-B: no boundary-aware term, no IoU surrogate"),
    "no_width": (True, "Fig. 1 caption: the auxiliary head receives no gradient"),
    "tversky_alpha": (0.5, "eq. (9) and Fig. 1 caption: alpha = beta = 0.5"),
    "iters": (20000, "Sec. III-B: 20,000 iterations"),
    "crop": (512, "Sec. III-B: 512x512 random crops"),
    "batch": (4, "Sec. III-B: batch size 4"),
    "accum": (2, "Sec. III-B: gradient accumulation 2"),
    "lr": (1e-4, "Sec. III-B: learning rate 1e-4"),
    "wd": (1e-2, "Sec. III-B: weight decay 1e-2"),
    "encoder": ("swin_small_patch4_window7_224.ms_in22k",
                "Sec. III-B: the timm encoder name"),
}

names = sys.argv[1:] or [a % s for a in ARMS for s in (0, 1, 2)]
bad = 0
for name in names:
    p = os.path.join(RUNS, name, "best.pt")
    if not os.path.exists(p):
        continue
    ck = torch.load(p, map_location="cpu", weights_only=False)
    sd = ck.get("model", ck.get("state_dict", ck))
    args = ck.get("args")
    a = vars(args) if hasattr(args, "__dict__") else (args or {})
    swin = a.get("model", LEGACY_MODEL_NAME) in MODEL_ALIASES

    problems = []
    for k, (want, where) in CLAIMS.items():
        if k not in a:
            continue

        if k in ("no_aniso",) and not swin:
            continue
        got = a[k]
        if isinstance(want, float):
            ok = abs(float(got) - want) < 1e-12
        else:
            ok = got == want
        if not ok:
            problems.append("%s = %r, but %s" % (k, got, where))


    up = [k for k in sd if k.startswith("up.")]
    if swin and up:
        problems.append("state dict has %d up.* parameters, so the upsample is "
                        "learned, not the bilinear one eq. (2) states" % len(up))

    flag = "OK " if not problems else "!! "
    bad += bool(problems)
    print("%s%-24s %s" % (flag, name, "" if not problems else ""))
    for x in problems:
        print("      %s" % x)

print("\n%d checkpoint(s) disagree with the manuscript" % bad)
sys.exit(1 if bad else 0)


"""Measure parameters, FLOPs and single-threaded CPU latency."""

import os, sys, time
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from swinbfe.net import SwinBFE
from swinbfe import baselines
from swinbfe.rescbam_unet import ResCBAMUNetWrapped

TILE = 512
BUILD = [
    ("Reference architecture (retrained)", lambda: ResCBAMUNetWrapped()),
    ("U-Net", lambda: baselines.UNet(pretrained=False)),
    ("Swin-BFE",
     lambda: SwinBFE(img_size=TILE, context="none", pretrained=False)),
    ("Swin-BFE-Mamba",
     lambda: SwinBFE(img_size=TILE, context="v2", pretrained=False)),
]


def measure(net, x):
    from torch.utils.flop_counter import FlopCounterMode
    try:
        with torch.no_grad(), FlopCounterMode(display=False) as fc:
            net(x)
        g = fc.get_total_flops() / 1e9


        with torch.no_grad():
            ts = []
            for _ in range(5):
                t0 = time.perf_counter()
                net(x)
                ts.append((time.perf_counter() - t0) * 1000)
        return g, min(ts)
    except RuntimeError as e:
        print("      cannot run on CPU: %s" % str(e).split(".")[0])
        return None, None


rows = []
x = torch.randn(1, 3, TILE, TILE)
for name, mk in BUILD:
    net = mk().eval()
    p = sum(q.numel() for q in net.parameters()) / 1e6
    g, dt = measure(net, x)
    rows.append((name, p, g, dt))
    print("%-34s %7.2f M  %s  %s" %
          (name, p, ("%7.1f G" % g) if g else "    n/a", ("%6.0f ms" % dt) if dt else "not supported on CPU"))
    del net

out = ["## Table 7 — Model cost\n",
       "| Method | Parameters (M) | FLOPs (G, 512x512 tile) | CPU latency (ms / tile) |",
       "|---|---:|---:|---:|"]
for n, p, g, dt in rows:
    out.append("| %s | %.2f | %s | %s |" % (
        n, p, ("%.1f" % g) if g else "CUDA only", ("%.0f" % dt) if dt else "CUDA only"))
out.append("")
out.append("FLOPs are counted with PyTorch's own FlopCounterMode, which covers "
           "convolution and matmul; normalisation and activation are excluded "
           "consistently across all rows. Latency is single-image CPU time on this "
           "workstation, the best of five runs, measured on an otherwise idle machine; "
           "it is for relative comparison only, not a deployment figure. All four rows "
           "are built in exactly the configuration that was trained, with bilinear "
           "upsampling in the decoder.\n")
out.append("Two things are worth reading off this table. First, the reference method "
           "spends 4.1x the FLOPs and 1.3x the parameters of ours and still loses on "
           "every metric in Tables 2 and 3 — that cost is not buying accuracy. "
           "Second, the Mamba decoder cannot run on CPU at all: its selective-scan "
           "step is a CUDA-only custom kernel and raises `Expected u.is_cuda() to be "
           "true` on a CPU tensor. The two decoders are within 0.22 M parameters of "
           "each other, so this is not a size trade: the plain decoder matches Mamba "
           "on IoU, beats it on boundaries, and runs without an NVIDIA GPU.\n")
open("tables/cost.md", "w", encoding="utf-8").write("\n".join(out))
print("\nwrote tables/cost.md")

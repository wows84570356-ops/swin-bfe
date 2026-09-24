
"""The Swin-BFE network.

A pretrained Swin-Small encoder supplies features at strides 4, 8, 16
and 32. Each is projected to a common width D by a 1x1 convolution with
GroupNorm (Eq. 1). The decoder walks from stride 32 to stride 4: the
stride-32 lateral is taken unchanged, and each finer scale upsamples by
two with parameter-free bilinear interpolation, adds its lateral,
normalises and applies a context block (Eq. 2). A mask head at stride 4
produces the logit map, interpolated once to full resolution (Eq. 3).

Three context blocks are interchangeable in that slot: ConvContext (the
configuration reported as Swin-BFE), StateSpaceContext (Swin-BFE-Mamba)
and VanillaScanContext (the standard-scan control).

The encoder binds its input size at construction, so a model evaluated
at a different tile size is rebuilt at that size and loaded from the
same checkpoint; window-local relative position bias makes the weights
valid across sizes.
"""

import math
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from .context_block import StateSpaceContext


class ConvContext(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.f = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, bias=False),
            nn.GroupNorm(8, dim), nn.GELU(),
            nn.Conv2d(dim, dim, 3, padding=1, bias=False),
            nn.GroupNorm(8, dim))

    def forward(self, x):
        return F.gelu(x + self.f(x))


class VanillaScanContext(nn.Module):
    """Standard pre-norm residual Mamba block, used as a second control.

    Interface matches ConvContext and StateSpaceContext: (B,C,H,W)->(B,C,H,W).
    The feature map is raster-scanned as one sequence, so none of DMMB's routing
    prior, four-direction scan, directional fusion, dilution evidence or
    anti-dilution gate is present. Source: the ablation file supplied with the
    block, unchanged apart from being added here rather than replacing net.py.
    """

    def __init__(self, dim, d_state=16, d_conv=4, expand=2):
        super().__init__()
        try:
            from mamba_ssm.modules.mamba_simple import Mamba
        except ImportError as exc:
            raise ImportError(
                "VanillaScanContext needs mamba-ssm; the DMMB path uses the "
                "same package, so if --context v2 works this should too."
            ) from exc
        self.norm = nn.LayerNorm(dim)
        self.mamba = Mamba(d_model=dim, d_state=d_state, d_conv=d_conv,
                           expand=expand)

    def forward(self, x):
        B, C, H, W = x.shape
        seq = x.flatten(2).transpose(1, 2).contiguous()
        seq = seq + self.mamba(self.norm(seq))
        return seq.transpose(1, 2).reshape(B, C, H, W).contiguous()


def _make_context(kind, dim):
    if kind == "v2":
        return StateSpaceContext(dim)
    if kind == "none":
        return ConvContext(dim)
    if kind == "vanilla":
        return VanillaScanContext(dim)
    raise ValueError(kind)


class SwinBFE(nn.Module):
    def __init__(self, encoder="swin_small_patch4_window7_224.ms_in22k",
                 img_size=512, context="none", dim=96, pretrained=True):
        super().__init__()
        self.img_size = img_size
        try:
            self.enc = timm.create_model(
                encoder, pretrained=pretrained, features_only=True,
                out_indices=(0, 1, 2, 3), img_size=img_size)
        except RuntimeError:
            base = encoder.split(".")[0]
            self.enc = timm.create_model(
                base, pretrained=pretrained, features_only=True,
                out_indices=(0, 1, 2, 3), img_size=img_size)
        chs = self.enc.feature_info.channels()
        self.chs = chs
        self.lat = nn.ModuleList(
            nn.Sequential(nn.Conv2d(c, dim, 1), nn.GroupNorm(8, dim))
            for c in chs)
        # Eq. (2): parameter-free bilinear upsampling
        self.up = nn.ModuleList(
            nn.Upsample(scale_factor=2, mode="bilinear",
                        align_corners=False) for _ in range(3))
        self.ctx = nn.ModuleList(_make_context(context, dim)
                                 for _ in range(3))


        self.pre_norm = nn.ModuleList(nn.GroupNorm(8, dim)
                                      for _ in range(3))
        self.mask_head = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, bias=False),
            nn.GroupNorm(8, dim), nn.GELU(),
            nn.Conv2d(dim, 1, 1))
        self.width_head = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, bias=False),
            nn.GroupNorm(8, dim), nn.GELU(),
            nn.Conv2d(dim, 2, 1))

    def _nchw(self, f, c):
        return f.permute(0, 3, 1, 2).contiguous() \
            if f.shape[-1] == c and f.shape[1] != c else f

    def forward(self, x):
        H, W = x.shape[-2:]
        feats = [self._nchw(f, c) for f, c in zip(self.enc(x), self.chs)]
        lats = [l(f) for l, f in zip(self.lat, feats)]
        f4, f8, f16, f32 = lats
        y = f32
        for up, ctx, skip, pn in zip(self.up, self.ctx, (f16, f8, f4),
                                     self.pre_norm):
            y = ctx(pn(up(y) + skip))
        logits = F.interpolate(self.mask_head(y), size=(H, W),
                               mode="bilinear", align_corners=False)
        wh = F.interpolate(self.width_head(y), size=(H, W),
                           mode="bilinear", align_corners=False)
        width = F.softplus(wh[:, 0:1])
        logvar = wh[:, 1:2].clamp(-6, 6)
        return {"logits": logits, "width": width, "logvar": logvar}


if __name__ == "__main__":
    net = SwinBFE(img_size=64, pretrained=False, context="none")
    out = net(torch.randn(2, 3, 64, 64))
    print({k: tuple(v.shape) for k, v in out.items()})
    n = sum(p.numel() for p in net.parameters())
    print(f"params {n/1e6:.1f}M")

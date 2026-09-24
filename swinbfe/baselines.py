"""The ResNet-34 U-Net baseline (Sec. II-C).

Checkpoints written during development store this model under the name
``unet_rgbd``; ``unet`` is the current name. Both select this class.
"""

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

UNET_ALIASES = ("unet", "unet_rgbd")


class Heads(nn.Module):
    """Mask head, plus the auxiliary two-channel head that is present in
    every released checkpoint but receives no gradient and does not enter
    the mask (Table I note)."""

    def __init__(self, dim):
        super().__init__()
        self.mask = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, bias=False),
            nn.GroupNorm(8, dim), nn.GELU(), nn.Conv2d(dim, 1, 1))
        self.width = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, bias=False),
            nn.GroupNorm(8, dim), nn.GELU(), nn.Conv2d(dim, 2, 1))

    def forward(self, f, hw):
        logits = F.interpolate(self.mask(f), size=hw, mode="bilinear",
                               align_corners=False)
        wh = F.interpolate(self.width(f), size=hw, mode="bilinear",
                           align_corners=False)
        return {"logits": logits,
                "width": F.softplus(wh[:, 0:1]),
                "logvar": wh[:, 1:2].clamp(-6, 6)}


class _InertProj(nn.Module):
    """A zero-initialised 1x1 projection that is never called.

    The released U-Net checkpoints contain one of these per encoder stage
    (``fuse.<i>.proj.*``), left over from an RGB-D variant that the paper
    does not use. They are kept so that those checkpoints load with
    ``strict=True`` and so that the parameter count matches Table I; they
    are not on the forward path.
    """

    def __init__(self, c):
        super().__init__()
        self.proj = nn.Conv2d(c, c, 1)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)


class _SharedEncoder(nn.Module):
    def __init__(self, name, pretrained):
        super().__init__()
        self.enc = timm.create_model(name, pretrained=pretrained,
                                     features_only=True,
                                     out_indices=(0, 1, 2, 3, 4))
        self.chs = self.enc.feature_info.channels()

    def forward(self, x):
        return self.enc(x)


class UNet(nn.Module):
    """ImageNet-pretrained ResNet-34 encoder with a four-stage decoder of
    3x3 convolution, GroupNorm and GELU blocks (channels 256/128/64/64)."""

    def __init__(self, pretrained=True, dec=(256, 128, 64, 64), head_dim=64):
        super().__init__()
        self.enc = _SharedEncoder("resnet34", pretrained)
        chs = self.enc.chs
        self.fuse = nn.ModuleList(_InertProj(c) for c in chs)
        ups, ins = [], chs[-1]
        for skip_c, out_c in zip(chs[-2::-1], dec):
            ups.append(nn.Sequential(
                nn.Conv2d(ins + skip_c, out_c, 3, padding=1, bias=False),
                nn.GroupNorm(8, out_c), nn.GELU(),
                nn.Conv2d(out_c, out_c, 3, padding=1, bias=False),
                nn.GroupNorm(8, out_c), nn.GELU()))
            ins = out_c
        self.ups = nn.ModuleList(ups)
        self.heads = Heads(head_dim)
        assert dec[-1] == head_dim

    def forward(self, x):
        hw = x.shape[-2:]
        fr = self.enc(x)
        y = fr[-1]
        for up, skip in zip(self.ups, fr[-2::-1]):
            y = F.interpolate(y, size=skip.shape[-2:], mode="bilinear",
                              align_corners=False)
            y = up(torch.cat([y, skip], 1))
        return self.heads(y, hw)


def build_baseline(name, pretrained=True):
    if name in UNET_ALIASES:
        return UNet(pretrained)
    raise ValueError(name)


if __name__ == "__main__":
    m = build_baseline("unet", pretrained=False)
    out = m(torch.randn(2, 3, 64, 64))
    assert out["logits"].shape == (2, 1, 64, 64)
    p = sum(q.numel() for q in m.parameters())
    print(f"unet: {p/1e6:.2f}M")

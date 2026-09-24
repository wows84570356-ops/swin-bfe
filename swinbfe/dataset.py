"""Dataset and tiling for the WHU and Inria building benchmarks.

Index files list every tile with paths relative to a data root; see
"Data layout" in the README. The root is given by ``data_root``, by the
environment variable ``SWINBFE_DATA_ROOT``, or defaults to the current
directory.
"""

import json
import os
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)


def data_root(root=None):
    """Resolve the data root: explicit argument, then $SWINBFE_DATA_ROOT, then '.'."""
    return root or os.environ.get("SWINBFE_DATA_ROOT") or "."


def resolve_path(p, root=None):
    """Join a relative index path onto the data root; absolute paths pass through."""
    return p if os.path.isabs(p) else os.path.join(data_root(root), p)


def _binarize(lbl):
    if lbl.ndim == 3:
        lbl = lbl[..., 0]
    return lbl > (127 if lbl.max() > 1 else 0)


def _ref_color_jitter(img, rng, brightness=0.2, contrast=0.2,
                      saturation=0.2, hue=0.1):
    x = img.astype(np.float32)
    if brightness > 0:
        x *= rng.uniform(1 - brightness, 1 + brightness)
    if contrast > 0:
        m = x.mean()
        x = (x - m) * rng.uniform(1 - contrast, 1 + contrast) + m
    if saturation > 0:
        g = (x * np.array([0.299, 0.587, 0.114], np.float32)).sum(-1,
                                                                  keepdims=True)
        x = g + (x - g) * rng.uniform(1 - saturation, 1 + saturation)
    x = np.clip(x, 0, 255).astype(np.uint8)
    if hue > 0:
        hsv = cv2.cvtColor(x, cv2.COLOR_RGB2HSV)
        shift = rng.uniform(-hue, hue) * 180.0
        hsv[..., 0] = (hsv[..., 0].astype(np.int16) +
                       int(round(shift))) % 180
        x = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return x


class TileDataset(Dataset):
    """Random 512x512 crops for training, centre crops for validation.

    Training crops are centred on a building pixel with probability
    ``foreground_p`` and uniformly otherwise, and with probability
    ``occlusion_p`` one to three small noise patches are pasted onto the
    building region. Both settings are the ones used for every run in
    the paper. ``ref_proto`` switches to the reference work's own recipe:
    whole-image resize to ``crop``, flips, colour jitter and rotation.
    """

    def __init__(self, index_json="splits/whu_index.json",
                 split_json="splits/whu_split.json", split="train",
                 crop=512, foreground_p=0.60, occlusion_p=0.3,
                 normalize=True, seed=0, rgb_override=None,
                 ref_proto=False, data_root=None):
        index = json.load(open(index_json, encoding="utf-8"))
        stems = json.load(open(split_json, encoding="utf-8"))[split]
        self.recs = [(s, index[s]) for s in stems]
        self.root = data_root
        self.crop = crop
        self.train = split == "train"
        self.occlusion_p = occlusion_p if self.train else 0.0
        self.p_foreground = foreground_p
        self.normalize = normalize
        self.rng = random.Random(seed)
        self.rgb_override = Path(rgb_override) if rgb_override else None
        self.ref_proto = ref_proto

    def __len__(self):
        return len(self.recs)

    def _load(self, rec):
        rgb_path = resolve_path(rec["rgb"], self.root)
        if self.rgb_override is not None:
            cand = self.rgb_override / Path(rec["rgb"]).name
            if cand.exists():
                rgb_path = str(cand)
        img = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(rgb_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        lbl = cv2.imread(resolve_path(rec["label"], self.root),
                         cv2.IMREAD_UNCHANGED)
        if lbl is None:
            raise FileNotFoundError(rec["label"])
        return img, _binarize(lbl)

    def _pick_center(self, mask):
        H, W = mask.shape
        jit = self.crop // 4
        if self.rng.random() < self.p_foreground and mask.any():
            ys, xs = np.nonzero(mask)
            i = self.rng.randrange(len(ys))
            return (ys[i] + self.rng.randint(-jit, jit),
                    xs[i] + self.rng.randint(-jit, jit))
        return self.rng.randrange(H), self.rng.randrange(W)

    def _window(self, shape, cy, cx):
        H, W = shape
        c = self.crop
        y0 = int(np.clip(cy - c // 2, 0, max(H - c, 0)))
        x0 = int(np.clip(cx - c // 2, 0, max(W - c, 0)))
        return y0, x0

    @staticmethod
    def _pad_to(a, c):
        H, W = a.shape[0], a.shape[1]
        pad = [(0, max(c - H, 0)), (0, max(c - W, 0))]
        if a.ndim == 3:
            pad.append((0, 0))
        return np.pad(a, pad, mode="reflect")

    def _occlude(self, img, mask):
        if not mask.any() or self.rng.random() > self.occlusion_p:
            return img
        img = img.copy()
        ys, xs = np.nonzero(mask)
        for _ in range(self.rng.randint(1, 3)):
            i = self.rng.randrange(len(ys))
            cy, cx = int(ys[i]), int(xs[i])
            h = self.rng.randint(8, 40)
            w = self.rng.randint(8, 40)
            y0, x0 = max(cy - h // 2, 0), max(cx - w // 2, 0)
            patch = img[y0:y0 + h, x0:x0 + w]
            base = patch.reshape(-1, 3).mean(0)
            noise = np.random.default_rng(self.rng.randrange(2**31)) \
                .normal(0, 12, patch.shape)
            img[y0:y0 + h, x0:x0 + w] = np.clip(base + noise, 0, 255)
        return img

    def _to_item(self, stem, img, mask):
        x = np.ascontiguousarray(img).astype(np.float32) / 255.0
        if self.normalize:
            x = (x - IMAGENET_MEAN) / IMAGENET_STD
        return {
            "image": torch.from_numpy(x.transpose(2, 0, 1).copy()),
            "mask": torch.from_numpy(
                np.ascontiguousarray(mask, np.float32)[None].copy()),
            "stem": stem,
        }

    def _ref_proto_item(self, stem, img, mask):
        c = self.crop
        img = cv2.resize(img, (c, c), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask.astype(np.float32), (c, c),
                          interpolation=cv2.INTER_NEAREST)
        if self.train:
            if self.rng.random() < 0.5:
                img, mask = np.fliplr(img), np.fliplr(mask)
            if self.rng.random() < 0.5:
                img, mask = np.flipud(img), np.flipud(mask)
            img = _ref_color_jitter(np.ascontiguousarray(img), self.rng)
            ang = self.rng.uniform(-15.0, 15.0)
            M = cv2.getRotationMatrix2D((c / 2.0, c / 2.0), ang, 1.0)
            img = cv2.warpAffine(np.ascontiguousarray(img), M, (c, c),
                                 flags=cv2.INTER_LINEAR)
            mask = cv2.warpAffine(np.ascontiguousarray(mask), M, (c, c),
                                  flags=cv2.INTER_NEAREST)
        return self._to_item(stem, img, mask)

    def __getitem__(self, i):
        stem, rec = self.recs[i]
        img, mask = self._load(rec)
        if self.ref_proto:
            return self._ref_proto_item(stem, img, mask)
        if self.train:
            cy, cx = self._pick_center(mask)
        else:
            cy, cx = mask.shape[0] // 2, mask.shape[1] // 2
        y0, x0 = self._window(mask.shape, cy, cx)
        c = self.crop
        sl = np.s_[y0:y0 + c, x0:x0 + c]
        img, mask = img[sl], mask[sl]
        img = self._occlude(img, mask)
        img = self._pad_to(img, c)
        mask = self._pad_to(mask.astype(np.float32), c)
        return self._to_item(stem, img, mask)


if __name__ == "__main__":
    import sys
    split = sys.argv[1] if len(sys.argv) > 1 else "train"
    ds = TileDataset(split=split)
    print(f"{split}: {len(ds)} images")
    hit = 0
    for i in range(12):
        b = ds[random.randrange(len(ds))]
        hit += bool(b["mask"].any())
        if i < 3:
            print({k: tuple(v.shape) for k, v in b.items()
                   if hasattr(v, "shape")},
                  f"fg {float(b['mask'].mean()):.2%}")
    print(f"crops containing a building, out of 12: {hit}/12")

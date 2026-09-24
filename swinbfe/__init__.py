"""Swin-BFE: building footprint extraction with an encoder-weighted design."""

from .baselines import UNet, build_baseline
from .dataset import TileDataset
from .losses import compute_losses
from .net import ConvContext, SwinBFE, VanillaScanContext
from .context_block import StateSpaceContext
from .rescbam_unet import ResCBAMUNetWrapped

__version__ = "1.0.0"
__all__ = [
    "SwinBFE",
    "ConvContext", "StateSpaceContext", "VanillaScanContext",
    "TileDataset", "compute_losses",
    "UNet", "build_baseline", "ResCBAMUNetWrapped",
]

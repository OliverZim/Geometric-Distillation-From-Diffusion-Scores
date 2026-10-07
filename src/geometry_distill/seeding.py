"""Global RNG seeding for reproducible runs."""

from __future__ import annotations

import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed the global Python, NumPy and PyTorch RNGs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

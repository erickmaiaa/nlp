"""Reproducibility helpers: fix every relevant random seed in one call."""

from __future__ import annotations

import os
import random

from ..config import get_settings
from ..logging_config import get_logger

logger = get_logger(__name__)


def seed_everything(seed: int | None = None) -> int:
    """Seed Python, ``PYTHONHASHSEED`` and (if present) NumPy.

    Args:
        seed: Seed value; defaults to ``Settings.random_seed``.

    Returns:
        The seed actually used.
    """
    seed = seed if seed is not None else get_settings().random_seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:  # pragma: no cover - numpy optional at import time
        pass
    logger.debug("Seeded all RNGs with %d", seed)
    return seed

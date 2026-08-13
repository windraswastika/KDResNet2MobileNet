import os
import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed model init, augmentation RNG, and all other stochastic sources.

    Note on the seed protocol (Reviewer E, point 2): the train/val/test
    partition is produced by a SEPARATE, fixed `split_seed` (see
    data_module.dataset.get_dataloaders), independent of this function.
    This function only controls what varies ACROSS the 5 evaluation seeds:
    model weight initialization (for newly-added classifier heads),
    data-loader shuffling order, and augmentation randomness. The test
    partition itself is identical across all 5 seeds.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

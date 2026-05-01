from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def collect_image_paths(root: str | Path) -> list[Path]:
    root_path = Path(root)
    if root_path.is_file():
        return [root_path]
    paths = [path for path in root_path.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS]
    return sorted(paths)

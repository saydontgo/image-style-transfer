from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

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


def make_image_transform(size: int | None, center_crop: bool = False) -> transforms.Compose:
    transform_steps: list[transforms.Transform] = []
    if size:
        transform_steps.append(transforms.Resize(size))
    if center_crop:
        transform_steps.append(transforms.CenterCrop(size))
    transform_steps.append(transforms.ToTensor())
    transform_steps.append(transforms.Lambda(lambda x: x.mul(255.0)))
    return transforms.Compose(transform_steps)


def load_image_as_tensor(
    image_path: str | Path,
    size: int | None = None,
    center_crop: bool = False,
) -> torch.Tensor:
    image = Image.open(image_path).convert("RGB")
    tensor = make_image_transform(size=size, center_crop=center_crop)(image)
    return tensor.unsqueeze(0)


def save_tensor_image(tensor: torch.Tensor, output_path: str | Path) -> None:
    output = tensor.detach().cpu().clamp(0.0, 255.0).squeeze(0)
    output = output.permute(1, 2, 0).numpy().astype(np.uint8)
    output_image = Image.fromarray(output)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_image.save(output_path)

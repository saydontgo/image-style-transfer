from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from tqdm import tqdm

from style_transfer.models import TransformerNet
from style_transfer.utils import collect_image_paths, load_image_as_tensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare a pretrained model with your own trained model.")
    parser.add_argument("--content-dir", type=str, required=True, help="Folder of content images used for comparison.")
    parser.add_argument("--baseline-model", type=str, required=True, help="Downloaded pretrained .pth/.ckpt model.")
    parser.add_argument("--custom-model", type=str, required=True, help="Your own trained .pth/.ckpt model.")
    parser.add_argument("--baseline-label", type=str, default="pretrained", help="Label shown on comparison output.")
    parser.add_argument("--custom-label", type=str, default="custom", help="Label shown on comparison output.")
    parser.add_argument("--output-dir", type=str, default="outputs/comparisons", help="Directory for comparison grids.")
    parser.add_argument("--image-size", type=int, default=768, help="Resize images for fair visual comparison.")
    parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    return parser.parse_args()


def load_compatible_model(model_path: str, device: torch.device) -> TransformerNet:
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
    model = TransformerNet().to(device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def tensor_to_image(tensor: torch.Tensor) -> Image.Image:
    array = tensor.detach().cpu().clamp(0.0, 255.0).squeeze(0).permute(1, 2, 0).numpy().astype(np.uint8)
    return Image.fromarray(array)


def add_label(image: Image.Image, label: str) -> Image.Image:
    canvas = Image.new("RGB", (image.width, image.height + 36), color=(255, 255, 255))
    canvas.paste(image, (0, 36))
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 10), label, fill=(0, 0, 0))
    return canvas


def make_triptych(original: Image.Image, baseline: Image.Image, custom: Image.Image, baseline_label: str, custom_label: str) -> Image.Image:
    labeled = [
        add_label(original, "content"),
        add_label(baseline, baseline_label),
        add_label(custom, custom_label),
    ]
    width = sum(image.width for image in labeled)
    height = max(image.height for image in labeled)
    merged = Image.new("RGB", (width, height), color=(255, 255, 255))
    offset = 0
    for image in labeled:
        merged.paste(image, (offset, 0))
        offset += image.width
    return merged


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    content_paths = collect_image_paths(args.content_dir)
    if not content_paths:
        raise FileNotFoundError(f"No content images found under: {args.content_dir}")

    baseline_model = load_compatible_model(args.baseline_model, device)
    custom_model = load_compatible_model(args.custom_model, device)

    summary = {
        "content_dir": args.content_dir,
        "baseline_model": args.baseline_model,
        "custom_model": args.custom_model,
        "num_images": len(content_paths),
        "image_size": args.image_size,
    }

    for image_path in tqdm(content_paths, desc="comparing", ncols=100):
        tensor = load_image_as_tensor(image_path, size=args.image_size).to(device)
        with torch.no_grad():
            baseline_out = baseline_model(tensor)
            custom_out = custom_model(tensor)

        original = tensor_to_image(tensor)
        baseline_img = tensor_to_image(baseline_out)
        custom_img = tensor_to_image(custom_out)
        grid = make_triptych(original, baseline_img, custom_img, args.baseline_label, args.custom_label)
        grid.save(output_dir / image_path.name)

    with open(output_dir / "summary.json", "w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)

    print(f"Comparison finished. Results saved to: {output_dir}")


if __name__ == "__main__":
    main()

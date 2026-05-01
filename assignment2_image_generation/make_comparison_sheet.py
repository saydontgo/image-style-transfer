from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw

from local_utils import collect_image_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create report-ready comparison sheets from original photos and generated style folders."
    )
    parser.add_argument("--input-dir", type=str, required=True, help="Original input image file or directory.")
    parser.add_argument(
        "--generated-root",
        type=str,
        required=True,
        help="Root directory whose subfolders are style names and contain generated images.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/assignment2_comparison",
        help="Directory used to store comparison sheets.",
    )
    parser.add_argument(
        "--label-height",
        type=int,
        default=36,
        help="Extra height added above each panel for the text label.",
    )
    return parser.parse_args()


def add_label(image: Image.Image, label: str, label_height: int) -> Image.Image:
    canvas = Image.new("RGB", (image.width, image.height + label_height), color=(255, 255, 255))
    canvas.paste(image, (0, label_height))
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 10), label, fill=(0, 0, 0))
    return canvas


def resize_to_match_height(image: Image.Image, height: int) -> Image.Image:
    if image.height == height:
        return image
    scale = height / image.height
    width = max(1, round(image.width * scale))
    return image.resize((width, height), Image.Resampling.LANCZOS)


def compose_row(images: list[Image.Image]) -> Image.Image:
    width = sum(image.width for image in images)
    height = max(image.height for image in images)
    canvas = Image.new("RGB", (width, height), color=(255, 255, 255))
    offset = 0
    for image in images:
        canvas.paste(image, (offset, 0))
        offset += image.width
    return canvas


def main() -> None:
    args = parse_args()
    input_paths = collect_image_paths(args.input_dir)
    if not input_paths:
        raise FileNotFoundError(f"No images found under: {args.input_dir}")

    generated_root = Path(args.generated_root)
    if not generated_root.exists():
        raise FileNotFoundError(f"Generated root does not exist: {generated_root}")

    style_dirs = sorted(path for path in generated_root.iterdir() if path.is_dir())
    if not style_dirs:
        raise FileNotFoundError(f"No style subdirectories found under: {generated_root}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for input_path in input_paths:
        original = Image.open(input_path).convert("RGB")
        panels = [add_label(original, "input", args.label_height)]

        for style_dir in style_dirs:
            generated_path = style_dir / input_path.name
            if not generated_path.exists():
                print(f"skip missing output: {generated_path}")
                continue
            generated = Image.open(generated_path).convert("RGB")
            generated = resize_to_match_height(generated, original.height)
            panels.append(add_label(generated, style_dir.name, args.label_height))

        result = compose_row(panels)
        output_path = output_dir / input_path.name
        result.save(output_path)
        print(f"saved: {output_path}")


if __name__ == "__main__":
    main()

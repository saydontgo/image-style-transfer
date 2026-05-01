from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from style_transfer.utils import collect_image_paths, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate multiple artistic styles from input photos with a diffusion image-to-image pipeline."
    )
    parser.add_argument("--input-dir", type=str, required=True, help="Input image file or directory.")
    parser.add_argument(
        "--styles-file",
        type=str,
        required=True,
        help="JSON file that defines style names, prompts, and optional inference settings.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/assignment2_diffusion",
        help="Directory used to store generated images.",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="runwayml/stable-diffusion-v1-5",
        help="Hugging Face model id used for image-to-image generation.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="cuda or cpu. Falls back to cpu when cuda is unavailable.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=768,
        help="Resize the long side of each input image before generation.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Base random seed.")
    return parser.parse_args()


def load_styles(styles_file: str | Path) -> dict[str, dict[str, Any]]:
    with open(styles_file, "r", encoding="utf-8") as file:
        styles = json.load(file)
    if not isinstance(styles, dict) or not styles:
        raise ValueError("Styles file must contain a non-empty JSON object.")
    for style_name, config in styles.items():
        if not isinstance(config, dict) or "prompt" not in config:
            raise ValueError(f"Style '{style_name}' must be an object with at least a 'prompt' field.")
    return styles


def resolve_device(device_name: str) -> torch.device:
    if device_name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def resize_long_side(image: Image.Image, long_side: int) -> Image.Image:
    if long_side <= 0:
        return image
    width, height = image.size
    current_long_side = max(width, height)
    if current_long_side == long_side:
        return image
    scale = long_side / current_long_side
    new_size = (max(1, round(width * scale)), max(1, round(height * scale)))
    return image.resize(new_size, Image.Resampling.LANCZOS)


def load_pipeline(model_id: str, device: torch.device):
    try:
        from diffusers import AutoPipelineForImage2Image
    except ImportError as error:
        raise ImportError(
            "diffusers is not installed. Run: pip install diffusers transformers accelerate safetensors"
        ) from error

    dtype = torch.float16 if device.type == "cuda" else torch.float32
    pipe = AutoPipelineForImage2Image.from_pretrained(model_id, torch_dtype=dtype)
    pipe = pipe.to(device)
    return pipe


def main() -> None:
    args = parse_args()
    input_paths = collect_image_paths(args.input_dir)
    if not input_paths:
        raise FileNotFoundError(f"No images found under: {args.input_dir}")

    styles = load_styles(args.styles_file)
    device = resolve_device(args.device)
    set_seed(args.seed)
    pipeline = load_pipeline(args.model_id, device)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "model_id": args.model_id,
        "input_dir": args.input_dir,
        "styles_file": args.styles_file,
        "image_size": args.image_size,
        "seed": args.seed,
        "num_inputs": len(input_paths),
        "styles": styles,
    }
    with open(output_dir / "run_config.json", "w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)

    for image_index, image_path in enumerate(input_paths):
        image = Image.open(image_path).convert("RGB")
        image = resize_long_side(image, args.image_size)

        for style_offset, (style_name, style_config) in enumerate(styles.items()):
            seed = args.seed + image_index * 1000 + style_offset
            generator = torch.Generator(device="cpu").manual_seed(seed)
            result = pipeline(
                prompt=style_config["prompt"],
                negative_prompt=style_config.get("negative_prompt"),
                image=image,
                strength=style_config.get("strength", 0.7),
                guidance_scale=style_config.get("guidance_scale", 7.5),
                num_inference_steps=style_config.get("num_inference_steps", 30),
                generator=generator,
            ).images[0]

            style_dir = output_dir / style_name
            style_dir.mkdir(parents=True, exist_ok=True)
            result.save(style_dir / image_path.name)
            print(f"[{style_name}] saved: {style_dir / image_path.name}")


if __name__ == "__main__":
    main()

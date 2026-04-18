from __future__ import annotations

import argparse
from pathlib import Path

import torch
from tqdm import tqdm

from style_transfer.models import TransformerNet
from style_transfer.models.checkpoint_compat import extract_compatible_state_dict
from style_transfer.utils import collect_image_paths, load_image_as_tensor, save_tensor_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stylize images with a trained fast NST model.")
    parser.add_argument("--model", type=str, required=True, help="Path to a compatible .pth or .ckpt model.")
    parser.add_argument("--input", type=str, required=True, help="Image file or folder to stylize.")
    parser.add_argument("--output-dir", type=str, default="outputs/inference", help="Directory for stylized outputs.")
    parser.add_argument("--image-size", type=int, default=0, help="Resize input images to this value. 0 keeps original size.")
    parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    return parser.parse_args()


def load_model(model_path: str, device: torch.device) -> TransformerNet:
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = extract_compatible_state_dict(checkpoint)
    model = TransformerNet().to(device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    output_dir = Path(args.output_dir)
    image_paths = collect_image_paths(args.input)
    if not image_paths:
        raise FileNotFoundError(f"No input images found under: {args.input}")

    model = load_model(args.model, device)
    for image_path in tqdm(image_paths, desc="stylizing", ncols=100):
        tensor = load_image_as_tensor(image_path, size=args.image_size or None).to(device)
        with torch.no_grad():
            stylized = model(tensor)
        save_tensor_image(stylized, output_dir / image_path.name)

    print(f"Stylized {len(image_paths)} image(s). Output dir: {output_dir}")


if __name__ == "__main__":
    main()

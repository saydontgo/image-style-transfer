from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from style_transfer.models import TransformerNet, VGG16Features, gram_matrix
from style_transfer.utils import collect_image_paths, load_image_as_tensor, save_tensor_image, set_seed

IMAGE_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGE_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
CONTENT_LAYER_WEIGHTS = {
    "relu2_2": 0.5,
    "relu3_3": 1.0,
}
STYLE_LAYER_WEIGHTS = {
    "relu1_2": 1.0,
    "relu2_2": 0.75,
    "relu3_3": 0.5,
    "relu4_3": 0.25,
}


class FlatImageDataset(Dataset):
    def __init__(self, image_paths: list[Path], image_size: int):
        self.image_paths = image_paths
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        image = Image.open(self.image_paths[index]).convert("RGB")
        image = image.resize((self.image_size, self.image_size), Image.Resampling.LANCZOS)
        tensor = torch.from_numpy(np.asarray(image)).permute(2, 0, 1).float()
        return tensor


def normalize_batch(batch: torch.Tensor) -> torch.Tensor:
    mean = IMAGE_MEAN.to(batch.device, batch.dtype)
    std = IMAGE_STD.to(batch.device, batch.dtype)
    return (batch / 255.0 - mean) / std


def total_variation_loss(batch: torch.Tensor) -> torch.Tensor:
    diff_x = batch[:, :, :, 1:] - batch[:, :, :, :-1]
    diff_y = batch[:, :, 1:, :] - batch[:, :, :-1, :]
    return diff_x.abs().mean() + diff_y.abs().mean()


def luminance_edges(batch: torch.Tensor) -> torch.Tensor:
    batch = batch / 255.0
    luminance = 0.299 * batch[:, 0:1] + 0.587 * batch[:, 1:2] + 0.114 * batch[:, 2:3]
    diff_x = nn.functional.pad(luminance[:, :, :, 1:] - luminance[:, :, :, :-1], (0, 1, 0, 0))
    diff_y = nn.functional.pad(luminance[:, :, 1:, :] - luminance[:, :, :-1, :], (0, 0, 0, 1))
    return torch.sqrt(diff_x.square() + diff_y.square() + 1e-6)


def edge_preservation_loss(stylized_batch: torch.Tensor, content_batch: torch.Tensor) -> torch.Tensor:
    return nn.functional.l1_loss(luminance_edges(stylized_batch), luminance_edges(content_batch))


def ensure_finite(name: str, tensor: torch.Tensor) -> None:
    if not torch.isfinite(tensor).all():
        raise RuntimeError(f"Non-finite values detected in {name}. Try disabling --mixed-precision or lowering the learning rate/style weight.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a fast neural style transfer model.")
    parser.add_argument("--dataset", type=str, required=True, help="Folder containing MS-COCO style content images.")
    parser.add_argument("--style-image", type=str, required=True, help="Single style image path.")
    parser.add_argument("--output-dir", type=str, default="checkpoints", help="Directory for checkpoints and final weights.")
    parser.add_argument("--preview-dir", type=str, default="", help="Optional folder of content images used for preview exports.")
    parser.add_argument("--run-name", type=str, default="", help="Experiment name. Defaults to style name + timestamp.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")

    # Hyperparameter: increase image size for finer textures, decrease it to cut VRAM and train faster.
    parser.add_argument("--image-size", type=int, default=256, help="Training content image size.")
    # Hyperparameter: larger style size keeps more style details, but costs extra preprocessing memory.
    parser.add_argument("--style-size", type=int, default=512, help="Resize long side of the style image to this value.")
    # Hyperparameter: 8 is a good starting point for 16 GB VRAM at 256x256. Lower it first if OOM happens.
    parser.add_argument("--batch-size", type=int, default=8, help="Training batch size.")
    # Hyperparameter: start from 2e-4. Lower it if loss oscillates, raise slightly if convergence is too slow.
    parser.add_argument("--learning-rate", type=float, default=2e-4, help="Adam learning rate.")
    # Hyperparameter: 2 to 4 epochs on a 20k-40k subset is usually enough for coursework-level results.
    parser.add_argument("--epochs", type=int, default=2, help="Number of full passes over the dataset subset.")
    # Hyperparameter: use 20k-40k for one GPU when you want faster turnaround; larger values improve robustness.
    parser.add_argument("--subset-size", type=int, default=20000, help="Use only the first N images. 0 means use all.")
    # Hyperparameter: increase for stronger artistic texture/color transfer; decrease to preserve photo structure.
    parser.add_argument("--style-weight", type=float, default=1e5, help="Weight applied to style loss.")
    # Hyperparameter: increase for better content fidelity; decrease if stylization feels too weak.
    parser.add_argument("--content-weight", type=float, default=1.0, help="Weight applied to content loss.")
    parser.add_argument("--edge-weight", type=float, default=12.0, help="Weight applied to edge preservation loss for text and fine structure.")
    # Hyperparameter: set to 0 if results are already clean, or raise slightly (1e-7 ~ 1e-5) to reduce noisy artifacts.
    parser.add_argument("--tv-weight", type=float, default=1e-6, help="Total variation regularization weight.")
    parser.add_argument("--num-workers", type=int, default=4, help="Dataloader workers.")
    parser.add_argument("--log-interval", type=int, default=1000, help="Steps between console logs.")
    parser.add_argument("--checkpoint-interval", type=int, default=1000, help="Steps between checkpoint saves.")
    parser.add_argument("--mixed-precision", action="store_true", help="Enable AMP training on CUDA.")
    return parser.parse_args()


def resolve_run_name(args: argparse.Namespace) -> str:
    if args.run_name:
        return args.run_name
    style_name = Path(args.style_image).stem.replace(" ", "_")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{style_name}_{timestamp}"


def export_preview_images(
    model: TransformerNet,
    image_paths: list[Path],
    output_dir: Path,
    image_size: int,
    device: torch.device,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    model_was_training = model.training
    model.eval()
    with torch.no_grad():
        for image_path in image_paths[:4]:
            tensor = load_image_as_tensor(image_path, size=image_size).to(device)
            stylized = model(tensor)
            save_tensor_image(stylized, output_dir / image_path.name)
    if model_was_training:
        model.train()


def save_training_artifacts(
    model: TransformerNet,
    optimizer: torch.optim.Optimizer,
    output_dir: Path,
    run_name: str,
    step: int,
    epoch: int,
    args: argparse.Namespace,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    state = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "epoch": epoch,
        "args": vars(args),
    }
    torch.save(state, output_dir / f"{run_name}_step_{step}.ckpt")
    torch.save(model.state_dict(), output_dir / f"{run_name}_step_{step}.pth")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_name = resolve_run_name(args)
    output_dir = Path(args.output_dir)
    preview_root = Path("outputs") / run_name

    image_paths = collect_image_paths(args.dataset)
    if not image_paths:
        raise FileNotFoundError(f"No images found under dataset path: {args.dataset}")
    if args.subset_size > 0:
        image_paths = image_paths[: args.subset_size]

    preview_paths = collect_image_paths(args.preview_dir) if args.preview_dir else image_paths[:4]
    dataset = FlatImageDataset(image_paths=image_paths, image_size=args.image_size)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )

    model = TransformerNet().to(device)
    loss_network = VGG16Features().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    mse_loss = nn.MSELoss()
    scaler = GradScaler(enabled=args.mixed_precision and device.type == "cuda")

    style_batch = load_image_as_tensor(args.style_image, size=args.style_size).to(device)
    with torch.no_grad():
        style_features = loss_network(normalize_batch(style_batch.float()))
        style_grams = {name: gram_matrix(value).detach() for name, value in style_features.items()}

    metadata = {
        "run_name": run_name,
        "style_image": args.style_image,
        "dataset": args.dataset,
        "num_training_images": len(image_paths),
        "device": str(device),
        "created_at": datetime.now().isoformat(),
        "hyperparameters": vars(args),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / f"{run_name}_config.json", "w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)

    model.train()
    global_step = 0
    for epoch in range(1, args.epochs + 1):
        progress = tqdm(dataloader, desc=f"epoch {epoch}/{args.epochs}", ncols=100)
        for batch in progress:
            global_step += 1
            content_batch = batch.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                content_features = loss_network(normalize_batch(content_batch.float()))

            with autocast(enabled=args.mixed_precision and device.type == "cuda"):
                stylized_batch = model(content_batch)

            # Keep perceptual losses in float32. AMP on VGG features / Gram matrix
            # can be numerically unstable and collapse outputs to black images.
            stylized_batch_for_loss = stylized_batch.float()
            stylized_features = loss_network(normalize_batch(stylized_batch_for_loss))

            content_loss = torch.zeros((), device=device, dtype=stylized_batch_for_loss.dtype)
            for name, weight in CONTENT_LAYER_WEIGHTS.items():
                content_loss = content_loss + mse_loss(stylized_features[name], content_features[name]) * weight
            content_loss = content_loss * args.content_weight

            style_loss = torch.zeros((), device=device, dtype=stylized_batch_for_loss.dtype)
            style_weight_total = sum(STYLE_LAYER_WEIGHTS.values())
            for name, features in stylized_features.items():
                target = style_grams[name].expand(content_batch.size(0), -1, -1)
                style_loss = style_loss + mse_loss(gram_matrix(features), target) * STYLE_LAYER_WEIGHTS[name]
            style_loss = style_loss / style_weight_total
            style_loss = style_loss * args.style_weight

            edge_loss = edge_preservation_loss(stylized_batch_for_loss, content_batch.float()) * args.edge_weight
            tv_loss = total_variation_loss(stylized_batch_for_loss) * args.tv_weight
            total_loss = content_loss + style_loss + edge_loss + tv_loss
            ensure_finite("stylized_batch", stylized_batch_for_loss)
            ensure_finite("total_loss", total_loss)

            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            progress.set_postfix(
                total=f"{total_loss.item():.2f}",
                style=f"{style_loss.item():.2f}",
                content=f"{content_loss.item():.2f}",
                edge=f"{edge_loss.item():.2f}",
            )

            if global_step % args.log_interval == 0:
                print(
                    f"[step {global_step}] total={total_loss.item():.4f} "
                    f"style={style_loss.item():.4f} content={content_loss.item():.4f} "
                    f"edge={edge_loss.item():.4f} tv={tv_loss.item():.4f}"
                )

            if global_step % args.checkpoint_interval == 0:
                save_training_artifacts(model, optimizer, output_dir, run_name, global_step, epoch, args)
                export_preview_images(
                    model=model,
                    image_paths=preview_paths,
                    output_dir=preview_root / f"step_{global_step}",
                    image_size=args.image_size,
                    device=device,
                )

    final_checkpoint = output_dir / f"{run_name}_final.pth"
    torch.save(model.state_dict(), final_checkpoint)
    export_preview_images(
        model=model,
        image_paths=preview_paths,
        output_dir=preview_root / "final",
        image_size=args.image_size,
        device=device,
    )
    print(f"Training finished. Final model saved to: {final_checkpoint}")


if __name__ == "__main__":
    main()

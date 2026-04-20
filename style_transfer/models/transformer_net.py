from __future__ import annotations

import torch
from torch import nn


LEGACY_OPTIONAL_STATE_PREFIXES = (
    "skip2_proj.",
    "skip1_proj.",
    "detail_head.",
)


class ConvLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int):
        super().__init__()
        padding = kernel_size // 2
        self.reflection_pad = nn.ReflectionPad2d(padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv2d(self.reflection_pad(x))


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            ConvLayer(channels, channels, kernel_size=3, stride=1),
            nn.InstanceNorm2d(channels, affine=True),
            nn.ReLU(inplace=True),
            ConvLayer(channels, channels, kernel_size=3, stride=1),
            nn.InstanceNorm2d(channels, affine=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class UpsampleConvLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        upsample: int | None = None,
    ):
        super().__init__()
        self.upsample = upsample
        self.layer = ConvLayer(in_channels, out_channels, kernel_size, stride)

    def forward(self, x: torch.Tensor, output_size: tuple[int, int] | None = None) -> torch.Tensor:
        if self.upsample:
            if output_size is not None:
                x = nn.functional.interpolate(x, mode="nearest", size=output_size)
            else:
                x = nn.functional.interpolate(x, mode="nearest", scale_factor=self.upsample)
        return self.layer(x)


class TransformerNet(nn.Module):
    """
    Johnson et al. fast neural style transfer network.

    This architecture is intentionally kept close to the common PyTorch example so
    external `.pth` checkpoints with the same layout are easy to reuse locally.
    """

    def __init__(self):
        super().__init__()
        self.downsampling = nn.Sequential(
            ConvLayer(3, 32, kernel_size=9, stride=1),
            nn.InstanceNorm2d(32, affine=True),
            nn.ReLU(inplace=True),
            ConvLayer(32, 64, kernel_size=3, stride=2),
            nn.InstanceNorm2d(64, affine=True),
            nn.ReLU(inplace=True),
            ConvLayer(64, 128, kernel_size=3, stride=2),
            nn.InstanceNorm2d(128, affine=True),
            nn.ReLU(inplace=True),
        )
        self.residuals = nn.Sequential(
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
        )
        self.upsampling = nn.Sequential(
            UpsampleConvLayer(128, 64, kernel_size=3, stride=1, upsample=2),
            nn.InstanceNorm2d(64, affine=True),
            nn.ReLU(inplace=True),
            UpsampleConvLayer(64, 32, kernel_size=3, stride=1, upsample=2),
            nn.InstanceNorm2d(32, affine=True),
            nn.ReLU(inplace=True),
            ConvLayer(32, 3, kernel_size=9, stride=1),
        )
        self.skip2_proj = nn.Conv2d(64, 64, kernel_size=1, stride=1)
        self.skip1_proj = nn.Conv2d(32, 32, kernel_size=1, stride=1)
        self.detail_head = nn.Sequential(
            ConvLayer(32 + 32 + 3, 32, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            ConvLayer(32, 16, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            ConvLayer(16, 3, kernel_size=3, stride=1),
        )
        self._initialize_detail_path()

    def _initialize_detail_path(self) -> None:
        # Start as the original Johnson network when loading old checkpoints.
        nn.init.zeros_(self.skip2_proj.weight)
        nn.init.zeros_(self.skip2_proj.bias)
        nn.init.zeros_(self.skip1_proj.weight)
        nn.init.zeros_(self.skip1_proj.bias)
        final_conv = self.detail_head[-1].conv2d
        nn.init.zeros_(final_conv.weight)
        nn.init.zeros_(final_conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.downsampling[0](x)
        h = self.downsampling[1](h)
        h = self.downsampling[2](h)
        skip1 = h

        h = self.downsampling[3](h)
        h = self.downsampling[4](h)
        h = self.downsampling[5](h)
        skip2 = h

        h = self.downsampling[6](h)
        h = self.downsampling[7](h)
        h = self.downsampling[8](h)

        h = self.residuals(h)

        h = self.upsampling[0](h, output_size=skip2.shape[-2:])
        h = self.upsampling[1](h)
        h = h + self.skip2_proj(skip2)
        h = self.upsampling[2](h)

        h = self.upsampling[3](h, output_size=skip1.shape[-2:])
        h = self.upsampling[4](h)
        h = h + self.skip1_proj(skip1)
        h = self.upsampling[5](h)

        base_output = self.upsampling[6](h)
        detail_input = torch.cat([h, skip1, x], dim=1)
        detail_residual = self.detail_head(detail_input)
        x = base_output + detail_residual
        # Keep training-time gradients intact. Output range is clipped only when
        # saving images for preview/inference.
        return x


def load_transformer_state_dict(model: TransformerNet, state_dict: dict[str, torch.Tensor]) -> None:
    incompatible = model.load_state_dict(state_dict, strict=False)
    disallowed_missing = [
        key
        for key in incompatible.missing_keys
        if not key.startswith(LEGACY_OPTIONAL_STATE_PREFIXES)
    ]
    if disallowed_missing or incompatible.unexpected_keys:
        problems: list[str] = []
        if disallowed_missing:
            problems.append(f"missing keys: {', '.join(disallowed_missing[:10])}")
        if incompatible.unexpected_keys:
            problems.append(f"unexpected keys: {', '.join(incompatible.unexpected_keys[:10])}")
        raise RuntimeError("Checkpoint is incompatible with the current TransformerNet layout: " + "; ".join(problems))

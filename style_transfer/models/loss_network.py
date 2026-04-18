from __future__ import annotations

import torch
from torch import nn
from torchvision import models


def gram_matrix(features: torch.Tensor) -> torch.Tensor:
    batch_size, channels, height, width = features.shape
    flattened = features.view(batch_size, channels, height * width)
    gram = flattened.bmm(flattened.transpose(1, 2))
    return gram / (channels * height * width)


class VGG16Features(nn.Module):
    """
    Feature extractor used for perceptual style/content losses.

    The selected layers are common for fast NST and balance training speed with
    stylization quality on a single 16 GB GPU.
    """

    def __init__(self):
        super().__init__()
        weights = models.VGG16_Weights.IMAGENET1K_V1
        features = models.vgg16(weights=weights).features.eval()
        self.slice1 = nn.Sequential(*features[:4])    # relu1_2
        self.slice2 = nn.Sequential(*features[4:9])   # relu2_2
        self.slice3 = nn.Sequential(*features[9:16])  # relu3_3
        self.slice4 = nn.Sequential(*features[16:23])  # relu4_3
        for parameter in self.parameters():
            parameter.requires_grad = False

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.slice1(x)
        relu1_2 = h
        h = self.slice2(h)
        relu2_2 = h
        h = self.slice3(h)
        relu3_3 = h
        h = self.slice4(h)
        relu4_3 = h
        return {
            "relu1_2": relu1_2,
            "relu2_2": relu2_2,
            "relu3_3": relu3_3,
            "relu4_3": relu4_3,
        }

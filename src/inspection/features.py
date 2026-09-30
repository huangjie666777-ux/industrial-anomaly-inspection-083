"""Frozen ResNet18 local feature extractor (layer2 + layer3)."""

from __future__ import annotations

from functools import lru_cache

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from . import config


class FeatureExtractor(nn.Module):
    """Extract layer2/layer3 features and concatenate aligned local descriptors.

    layer2 output is 28x28x128 and layer3 output is 14x14x256. layer2 is
    bilinearly aligned to 14x14, giving 196 local descriptors of dimension
    128 + 256 = 384 per image.
    """

    def __init__(self, weights_path) -> None:
        super().__init__()
        state_dict = torch.load(weights_path, map_location="cpu")
        backbone = torchvision.models.resnet18(weights=None)
        backbone.load_state_dict(state_dict)
        backbone.eval()
        for parameter in backbone.parameters():
            parameter.requires_grad_(False)
        self.stem = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
        )
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.eval()

    @torch.no_grad()
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        x = self.stem(image)
        x = self.layer1(x)
        layer2 = self.layer2(x)  # N x 128 x 28 x 28
        layer3 = self.layer3(layer2)  # N x 256 x 14 x 14
        aligned = F.interpolate(
            layer2,
            size=(config.GRID_SIZE, config.GRID_SIZE),
            mode="bilinear",
            align_corners=False,
        )
        descriptors = torch.cat([aligned, layer3], dim=1)
        # N x 196 x 384
        return descriptors.flatten(2).transpose(1, 2)


@lru_cache(maxsize=1)
def get_extractor() -> FeatureExtractor:
    return FeatureExtractor(config.MODEL_PATH)

"""ResNet18 for single-channel images."""

from typing import cast

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18


def build_model(num_classes: int, *, pretrained: bool = True) -> nn.Module:
    """ResNet18 with a one-channel stem and a new classification head.

    With ``pretrained``, the ImageNet weights of the first convolution are summed over the three
    colour channels, which keeps its response to a grey image the same as to the RGB one.
    """
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
    rgb = model.conv1.weight.detach()
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    if pretrained:
        with torch.no_grad():
            model.conv1.weight.copy_(rgb.sum(dim=1, keepdim=True))
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return cast(nn.Module, model)

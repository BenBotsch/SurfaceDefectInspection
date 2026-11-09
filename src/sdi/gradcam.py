"""Grad-CAM: which image regions drive the score of a class.

Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based
Localization", ICCV 2017. The activations of the last residual stage are weighted by the mean
gradient of the class score with respect to them, summed over channels and clipped at zero.
"""

import numpy as np
import numpy.typing as npt
import torch
from torch import nn
from torch.nn import functional


def grad_cam(
    model: nn.Module,
    tensor: torch.Tensor,
    target: int | None = None,
    layer: nn.Module | None = None,
) -> npt.NDArray[np.float32]:
    """The class activation map of one image, scaled to the input size and to [0, 1].

    Args:
        model: A classifier returning class scores; it is put in eval mode.
        tensor: One preprocessed image, shape 1 x C x H x W.
        target: Class to explain; the predicted class if unset.
        layer: Layer whose activations are weighted; ``model.layer4`` of a ResNet if unset.

    Returns:
        An H x W map, 0 where the layer gives no evidence for the class.
    """
    if tensor.ndim != 4 or tensor.shape[0] != 1:
        raise ValueError(f"expected one image of shape 1 x C x H x W, got {tuple(tensor.shape)}")
    layer = layer if layer is not None else model.get_submodule("layer4")
    captured: dict[str, torch.Tensor] = {}

    def keep(_: nn.Module, __: object, output: torch.Tensor) -> None:
        captured["activations"] = output

    handle = layer.register_forward_hook(keep)
    try:
        model.eval()
        scores = model(tensor)
    finally:
        handle.remove()
    index = int(scores.argmax(dim=1)) if target is None else target
    activations = captured["activations"]
    # The gradient of the score with respect to the activations only; the weights get none.
    (gradients,) = torch.autograd.grad(scores[0, index], activations)
    weights = gradients.mean(dim=(2, 3), keepdim=True)
    cam = functional.relu((weights * activations).sum(dim=1, keepdim=True))
    cam = functional.interpolate(cam, size=tensor.shape[2:], mode="bilinear", align_corners=False)
    out = cam[0, 0].detach().cpu().numpy().astype(np.float32)
    peak = float(out.max())
    return out / peak if peak > 0 else out

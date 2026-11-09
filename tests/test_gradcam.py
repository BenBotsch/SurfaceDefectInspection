import pytest
import torch

from sdi.dataset import CLASSES
from sdi.gradcam import grad_cam
from sdi.model import build_model
from sdi.preprocess import SIZE


def test_map_has_the_input_size_and_lies_in_unit_range() -> None:
    torch.manual_seed(0)
    model = build_model(len(CLASSES), pretrained=False)

    cam = grad_cam(model, torch.randn(1, 1, SIZE, SIZE), target=2)

    assert cam.shape == (SIZE, SIZE)
    assert cam.min() >= 0
    assert cam.max() == pytest.approx(1)


def test_hooks_are_removed_and_the_model_still_predicts() -> None:
    model = build_model(len(CLASSES), pretrained=False)
    image = torch.randn(1, 1, SIZE, SIZE)

    grad_cam(model, image)

    assert not model.get_submodule("layer4")._forward_hooks
    assert all(p.grad is None for p in model.parameters())
    with torch.no_grad():
        assert model(image).shape == (1, len(CLASSES))


def test_only_single_images_are_accepted() -> None:
    model = build_model(len(CLASSES), pretrained=False)

    with pytest.raises(ValueError, match="one image"):
        grad_cam(model, torch.randn(2, 1, SIZE, SIZE))

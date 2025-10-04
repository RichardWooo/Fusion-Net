from __future__ import annotations

from typing import Callable, Tuple

import torch
from torch import Tensor, nn


def grad_cam(
    model: nn.Module,
    image: Tensor,
    target_class: int,
    target_layer: nn.Module,
    preprocess: Callable[[Tensor], Tensor] | None = None,
) -> Tuple[Tensor, Tensor]:
    """Compute Grad-CAM heatmap for a classification model."""

    model.eval()
    gradients: list[Tensor] = []
    activations: list[Tensor] = []

    def backward_hook(_: nn.Module, grad_input: Tuple[Tensor, ...], grad_output: Tuple[Tensor, ...]):
        gradients.append(grad_output[0])

    def forward_hook(_: nn.Module, input: Tuple[Tensor, ...], output: Tensor):
        activations.append(output)

    handle_forward = target_layer.register_forward_hook(forward_hook)
    handle_backward = target_layer.register_full_backward_hook(backward_hook)

    if preprocess:
        input_tensor = preprocess(image)
    else:
        input_tensor = image

    input_tensor = input_tensor.unsqueeze(0)
    input_tensor.requires_grad_(True)

    scores = model(input_tensor)
    loss = scores[0, target_class]
    model.zero_grad()
    loss.backward()

    grads = gradients.pop()
    acts = activations.pop()

    handle_forward.remove()
    handle_backward.remove()

    weights = grads.mean(dim=(2, 3), keepdim=True)
    cam = torch.relu((weights * acts).sum(dim=1, keepdim=True))
    cam = torch.nn.functional.interpolate(cam, size=image.shape[1:], mode="bilinear", align_corners=False)

    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)
    heatmap = cam.squeeze().detach()

    overlay = 0.5 * image + 0.5 * heatmap.unsqueeze(0)
    overlay = torch.clamp(overlay, 0, 1)

    return heatmap, overlay

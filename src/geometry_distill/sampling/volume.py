"""Jacobian volume elements and projection residuals."""

from __future__ import annotations

import torch
import torch.nn as nn

_SINGULAR_VALUE_FLOOR = 1e-8


def projection_residual_squared(
    projector: nn.Module,
    x: torch.Tensor,
) -> torch.Tensor:
    r"""Squared projection residual :math:`\|x - \mathrm{proj}(x)\|^2` per sample."""
    x_hat = projector(x)
    return ((x - x_hat) ** 2).sum(dim=-1)


def _is_atlas(projector: nn.Module) -> bool:
    return getattr(projector, "_is_atlas", False)


def encoder_jacobian(
    projector: nn.Module,
    x: torch.Tensor,
    z: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute the encoder Jacobian, shape ``(batch, k, x_dim)``.

    ``z`` may be a precomputed ``encode(x)`` to avoid a duplicate forward.

    Raises:
        TypeError: If *projector* is a multi-chart atlas.
    """
    if _is_atlas(projector):
        raise TypeError(
            "encoder_jacobian requires a single-chart projector "
            "(_is_atlas=False), but an atlas (_is_atlas=True) was "
            "provided."
        )

    if not x.requires_grad:
        x = x.detach().requires_grad_(True)

    if z is None:
        z = projector.encode(x)
    k = z.shape[1]
    grad_outputs = torch.eye(k, device=z.device, dtype=z.dtype)
    grad_outputs = grad_outputs[:, None, :].expand(k, x.shape[0], k)
    grads = torch.autograd.grad(
        z,
        x,
        grad_outputs=grad_outputs,
        is_grads_batched=True,
        create_graph=True,
        retain_graph=True,
    )[0]
    return grads.permute(1, 0, 2)


def log_volume_from_jacobian(J: torch.Tensor) -> torch.Tensor:
    r"""Log-volume element :math:`\log\sqrt{\det(J^T J)}` per sample, shape (batch,)."""
    gram = J @ J.transpose(-2, -1)
    eigenvalues = torch.linalg.eigvalsh(gram)
    eigenvalues = torch.clamp(eigenvalues, min=_SINGULAR_VALUE_FLOOR**2)
    return 0.5 * torch.log(eigenvalues).sum(dim=-1)


def log_volume_element(
    projector: nn.Module,
    x: torch.Tensor,
) -> torch.Tensor:
    r"""Log-volume element :math:`\log\sqrt{\det(J^T J)}` per sample, shape (batch,)."""
    return log_volume_from_jacobian(encoder_jacobian(projector, x))

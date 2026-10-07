"""Staged loss functions for projection distillation."""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
from pydantic import Field
from torch.func import jvp

from ..config import ComponentConfig


class ProjectionLossConfig(ComponentConfig):
    """Configuration for projection loss weights."""

    lambda_proj: float = Field(
        default=1.0,
        ge=0.0,
        description="Weight of the projection matching loss.",
    )
    lambda_fix: float = Field(
        default=1.0,
        ge=0.0,
        description="Weight of the on-manifold fixed-point loss.",
    )
    lambda_idem: float = Field(
        default=1.0,
        ge=0.0,
        description="Weight of the idempotence loss.",
    )
    lambda_fiber: float = Field(
        default=0.0,
        ge=0.0,
        description="Weight of the fiber-invariance loss (0 disables it).",
    )
    lambda_overlap: float = Field(
        default=0.0,
        ge=0.0,
        description="Weight of the chart-overlap loss (0 disables it).",
    )
    lambda_orth: float = Field(
        default=0.0,
        ge=0.0,
        description="Weight of the residual-orthogonality loss (0 disables it).",
    )
    lambda_iso: float = Field(
        default=0.0,
        ge=0.0,
        description="Weight of the latent-isometry loss (0 disables it).",
    )
    iso_dirs: int = Field(
        default=4,
        ge=1,
        description="Random directions per sample for the isometry loss.",
    )


def _zero_on_device(x: torch.Tensor, projector: nn.Module) -> torch.Tensor:
    param = next(projector.parameters(), None)
    device = param.device if param is not None else x.device
    return torch.zeros((), device=device, dtype=x.dtype)


def _encode_latent(projector: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Return latent codes from a projector's ``encode``."""
    if not hasattr(projector, "encode"):
        raise AttributeError(
            "projector must have an encode(x) method for fiber/overlap losses"
        )
    result = projector.encode(x)
    if isinstance(result, tuple):
        return result[1]
    return result  # type: ignore[return-value]


def _is_atlas(projector: nn.Module) -> bool:
    return getattr(projector, "_is_atlas", False)


def loss_proj(pred: torch.Tensor, pseudo_target: torch.Tensor) -> torch.Tensor:
    """Match the projector output to the pseudo-projection target."""
    return torch.mean((pred - pseudo_target) ** 2)


def loss_fix(pred: torch.Tensor, x_on: torch.Tensor) -> torch.Tensor:
    """Keep on-manifold points fixed by the projector."""
    return torch.mean((pred - x_on) ** 2)


def loss_idem(pred: torch.Tensor, twice: torch.Tensor) -> torch.Tensor:
    """Idempotence loss ``||P(P(x)) - P(x)||²``."""
    return torch.mean((twice - pred.detach()) ** 2)


def loss_fiber(z_off: torch.Tensor, z_proj: torch.Tensor) -> torch.Tensor:
    """Keep latent codes unchanged by the projection (fiber invariance)."""
    return torch.mean((z_off - z_proj) ** 2)


def loss_orth(
    pred: torch.Tensor,
    residual: torch.Tensor,
    z: torch.Tensor,
    projector: nn.Module,
) -> torch.Tensor:
    """Penalize residual components along the estimated tangent space."""
    if _is_atlas(projector):
        return _zero_on_device(pred, projector)

    with torch.enable_grad():
        if not z.requires_grad:
            z = z.detach().requires_grad_(True)
            pred = projector.decode(z)
        jt_r = torch.autograd.grad(
            pred,
            z,
            grad_outputs=residual,
            create_graph=True,
        )[0]
    return torch.mean(jt_r**2)


def loss_iso(
    x: torch.Tensor,
    projector: nn.Module,
    n_dirs: int = 4,
) -> torch.Tensor:
    """Local isometry loss of the latent map on the learned tangent space."""
    latent_fn = (
        projector.soft_latent if hasattr(projector, "soft_latent") else projector.encode
    )
    b, d = x.shape
    xr = x.detach().unsqueeze(1).expand(b, n_dirs, d).reshape(-1, d)
    v = torch.randn_like(xr)
    _, t = jvp(projector, (xr,), (v,))
    t = t.detach()
    _, u = jvp(latent_fn, (xr,), (t,))
    t_sq = (t**2).sum(dim=-1)
    u_sq = (u**2).sum(dim=-1)
    log_ratio = torch.log(u_sq.clamp_min(1e-12)) - torch.log(t_sq.clamp_min(1e-12))
    return torch.mean(log_ratio**2)


def loss_overlap(
    x: torch.Tensor,
    projector: nn.Module,
) -> torch.Tensor:
    """Penalize disagreement between overlapping chart decoders (atlas only)."""
    if not _is_atlas(projector):
        return _zero_on_device(x, projector)

    weights, all_z = projector.encode(x)  # type: ignore[misc]
    _batch, n_charts, _k = all_z.shape

    if n_charts < 2:
        return _zero_on_device(x, projector)

    per_chart: list[torch.Tensor] = []
    for j in range(n_charts):
        x_j = projector.decode(all_z[:, j, :], chart_idx=j)  # type: ignore[misc]
        per_chart.append(x_j)
    per_chart = torch.stack(per_chart, dim=1)

    diff = per_chart.unsqueeze(2) - per_chart.unsqueeze(1)
    sq_dist = (diff**2).sum(dim=-1)
    w_prod = weights.unsqueeze(2) * weights.unsqueeze(1)

    diag_mask = torch.eye(n_charts, dtype=torch.bool, device=x.device)
    w_prod = w_prod.masked_fill(diag_mask, 0.0)
    sq_dist = sq_dist.masked_fill(diag_mask, 0.0)

    return (w_prod * sq_dist).sum(dim=(1, 2)).mean()


def compute_losses(
    x_off: torch.Tensor,
    pseudo_target: torch.Tensor,
    x_on: Optional[torch.Tensor],
    projector: nn.Module,
    config: ProjectionLossConfig,
) -> Dict[str, torch.Tensor]:
    """Compute all active losses and their weighted sum.

    Returns:
        Per-loss values plus ``"total"``, the weighted sum for backprop.
    """
    results: Dict[str, torch.Tensor] = {}

    if _is_atlas(projector):
        pred_off = projector(x_off)
        z_off = _encode_latent(projector, x_off)
    else:
        z_off = projector.encode(x_off)
        pred_off = projector.decode(z_off)

    proj = loss_proj(pred_off, pseudo_target)
    results["proj"] = proj
    total = config.lambda_proj * proj

    if x_on is not None and config.lambda_fix > 0:
        pred_on = pred_off if x_on is x_off else projector(x_on)
        fix = loss_fix(pred_on, x_on)
        results["fix"] = fix
        total = total + config.lambda_fix * fix

    if config.lambda_idem > 0 or config.lambda_fiber > 0:
        pred_detached = pred_off.detach()
        if _is_atlas(projector):
            z_proj = _encode_latent(projector, pred_detached)
            twice = projector(pred_detached)
        else:
            z_proj = projector.encode(pred_detached)
            twice = projector.decode(z_proj) if config.lambda_idem > 0 else None

        if config.lambda_idem > 0:
            idem = loss_idem(pred_off, twice)
            results["idem"] = idem
            total = total + config.lambda_idem * idem

        if config.lambda_fiber > 0:
            fiber = loss_fiber(z_off, z_proj)
            results["fiber"] = fiber
            total = total + config.lambda_fiber * fiber

    if config.lambda_orth > 0:
        orth = loss_orth(pred_off, x_off - pred_off, z_off, projector)
        results["orth"] = orth
        total = total + config.lambda_orth * orth

    if config.lambda_overlap > 0:
        overlap = loss_overlap(x_off, projector)
        results["overlap"] = overlap
        total = total + config.lambda_overlap * overlap

    if config.lambda_iso > 0:
        iso = loss_iso(x_on if x_on is not None else x_off, projector, config.iso_dirs)
        results["iso"] = iso
        total = total + config.lambda_iso * iso

    results["total"] = total
    return results

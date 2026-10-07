#!/usr/bin/env python
"""Pooled population transport on a real scRNA-seq working space."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Literal, Self

import numpy as np
import ot
import torch
import torch.nn as nn
from pydantic import Field, model_validator
from pydantic_settings import SettingsConfigDict
from scipy.spatial.distance import cdist

from experiments.real_data.common.space import load_geometry
from geometry_distill.config import ExperimentConfig
from geometry_distill.device import resolve_device
from geometry_distill.geodesics.metric import WarpedPullbackMetricConfig
from geometry_distill.geodesics.refine import PathRefinementConfig, refine_paths
from geometry_distill.projection import projector_from_checkpoint
from geometry_distill.run import RunConfig, RunWriter
from geometry_distill.seeding import set_seed

logger = logging.getLogger(__name__)


class TransportExperimentConfig(ExperimentConfig):
    """Configuration for the real-data population-transport script."""

    model_config = SettingsConfigDict(
        cli_kebab_case=True,
        cli_parse_none_str="none",
    )

    scenario: Literal["eb", "cite50", "cite100", "multi50", "multi100"] = Field(
        default="eb",
        description="Real dataset to transport.",
    )
    data_root: Path = Field(
        default=Path("experiments/real_data/eb/data"),
        description="Data root containing <scenario>/geometry/.",
    )
    seed: int = Field(
        default=42,
        description="Master seed (coupling, subsampling, evaluation draws).",
    )
    held_out: str = Field(
        default="T4_5C",
        description="Held-out timepoint (window) to predict.",
    )
    projector_checkpoint: Path = Field(
        description="Path to the projector checkpoint (stage 2 output).",
    )

    n_pairs: int = Field(
        default=1500,
        ge=1,
        description="Source/target cells used for the OT coupling and CFM baselines.",
    )
    beta: float = Field(
        default=10.0,
        description="Residual weight of the warped metric used by chord refinement.",
    )
    fm_hidden_dim: int = Field(
        default=64,
        ge=1,
        description="Width of the flow-matching velocity network.",
    )
    fm_epochs: int = Field(
        default=150,
        ge=0,
        description="Training epochs per CFM baseline.",
    )
    time_samples: int = Field(
        default=20,
        ge=1,
        description="Time samples per source cell in the CFM loss.",
    )
    batch_size: int = Field(
        default=128,
        ge=1,
        description="CFM training batch size.",
    )
    lr: float = Field(
        default=1e-3,
        description="Learning rate for the CFM velocity networks.",
    )
    weight_decay: float = Field(
        default=1e-4,
        description="Weight decay for the CFM velocity networks.",
    )
    chord_refine_max_ratio: float = Field(
        default=5.0,
        gt=0.0,
        description="Revert a refined path to its chord beyond this length ratio.",
    )
    pair_space: Literal["latent", "ambient"] = Field(
        default="latent",
        description="Space of the OT pairing and chord refinement.",
    )
    refine_steps: int = Field(
        default=300,
        ge=0,
        description="Optimisation steps per refined chord.",
    )
    refine_lr: float = Field(
        default=3e-3,
        description="Learning rate of the chord refinement.",
    )
    source_window: str = Field(
        default="",
        description="Source window; default the one before the held-out window.",
    )
    target_window: str = Field(
        default="",
        description="Target window; default the one after the held-out window.",
    )

    gaga_test_frac: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Fraction of each population used as the GAGA test split.",
    )
    gaga_n_tsteps: int = Field(
        default=100,
        ge=2,
        description="Time steps of the GAGA trajectory cloud.",
    )
    gaga_n_match_rep: int = Field(
        default=5,
        ge=1,
        description="Repeats of the sample-size-matched W1.",
    )

    run: RunConfig = Field(
        description="Output directory and overwrite behavior.",
    )

    @model_validator(mode="after")
    def _check_windows(self) -> Self:
        if self.source_window and self.source_window == self.held_out:
            raise ValueError("source_window must differ from the held-out window")
        if self.target_window and self.target_window == self.held_out:
            raise ValueError("target_window must differ from the held-out window")
        return self


class VelocityMLP(nn.Module):
    def __init__(self, x_dim: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(x_dim + 1, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, x_dim),
        )

    def forward(self, x, t):
        return self.net(torch.cat([x, t.unsqueeze(-1)], dim=-1))


def window_order(labels) -> list[str]:
    """Return the collection windows ordered by mean pseudotime."""
    return list(labels.groupby("Timepoint")["Pseudotime"].mean().sort_values().index)


def w1(X, Y) -> float:
    """Exact W1 (Euclidean cost) between two point clouds."""
    a = np.ones(len(X)) / len(X)
    b = np.ones(len(Y)) / len(Y)
    C = cdist(
        np.asarray(X, dtype=np.float64),
        np.asarray(Y, dtype=np.float64),
        metric="euclidean",
    )
    return float(ot.emd2(a, b, C))


def train_velocity_cfm(
    x0: torch.Tensor,
    x1: torch.Tensor,
    x_dim: int,
    config: TransportExperimentConfig,
    run: RunWriter,
    tag: str,
    device,
) -> VelocityMLP:
    """Fit a velocity field on straight paths ``x_t = (1-t) x0 + t x1``."""
    n = len(x0)
    vel = VelocityMLP(x_dim, config.fm_hidden_dim).to(device)
    opt = torch.optim.AdamW(
        vel.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    for ep in range(config.fm_epochs):
        perm = torch.randperm(n, device=device)
        tot = 0.0
        for j in range(0, n, config.batch_size):
            idx = perm[j : j + config.batch_size]
            b = len(idx)
            m = config.time_samples
            pi = idx.repeat_interleave(m)
            tt = torch.rand(b * m, device=device)
            x_t = (1 - tt).unsqueeze(-1) * x0[pi] + tt.unsqueeze(-1) * x1[pi]
            u = x1[pi] - x0[pi]
            loss = ((vel(x_t, tt) - u) ** 2).sum(-1).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss.detach()) * b
        run.log_metrics(ep, {f"fm_loss_{tag}": tot / n})
    vel.eval()
    return vel


def projection_variant(projector_checkpoint: Path) -> str | None:
    """Return the variant recorded next to a projector checkpoint, if any."""
    config_path = Path(projector_checkpoint).parent / "config.json"
    if not config_path.is_file():
        return None
    try:
        return json.loads(config_path.read_text()).get("variant")
    except json.JSONDecodeError:
        return None


def main() -> None:
    config = TransportExperimentConfig()
    set_seed(config.seed)

    try:
        geom = load_geometry(config.scenario, config.data_root)
        run = RunWriter(config.run)
    except (FileExistsError, FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc)) from None

    with run:
        run.write_config(config)
        _run(config, run, geom)


def _run(
    config: TransportExperimentConfig,
    run: RunWriter,
    geom: dict,
) -> None:
    logger.info("Output directory: %s", run.out_dir)

    device = resolve_device()
    logger.info("Using device: %s", device)

    X = geom["X_pca"].astype(np.float32)
    labels = geom["labels"]
    tp = labels["Timepoint"].values
    windows = window_order(labels)
    if config.held_out not in windows:
        raise ValueError(f"held_out {config.held_out!r} not in windows {windows}")
    i = windows.index(config.held_out)
    prev_w = config.source_window or windows[i - 1]
    next_w = config.target_window or windows[i + 1]
    if not windows.index(prev_w) < i < windows.index(next_w):
        raise ValueError(
            f"windows must bracket the held-out window: "
            f"{prev_w} < {config.held_out} < {next_w}"
        )

    ckpt = torch.load(
        config.projector_checkpoint, map_location=device, weights_only=False
    )
    projector = projector_from_checkpoint(ckpt, device=device)
    variant = projection_variant(config.projector_checkpoint)
    day_blind = variant == f"holdout_{config.held_out}"
    if variant is not None and not day_blind:
        logger.warning(
            "projection run variant is %r, not holdout_%s -> NOT day-blind",
            variant,
            config.held_out,
        )

    src_all, tgt_all, real = X[tp == prev_w], X[tp == next_w], X[tp == config.held_out]
    rng = np.random.default_rng(config.seed)
    src = src_all[
        rng.choice(len(src_all), min(config.n_pairs, len(src_all)), replace=False)
    ]
    tgt = tgt_all[
        rng.choice(len(tgt_all), min(config.n_pairs, len(tgt_all)), replace=False)
    ]
    logger.info(
        "held out %s: source %s (%s), target %s (%s), real %s; pairs %sx%s",
        config.held_out,
        prev_w,
        len(src_all),
        next_w,
        len(tgt_all),
        len(real),
        len(src),
        len(tgt),
    )

    with torch.no_grad():
        enc = (
            projector.soft_latent
            if hasattr(projector, "soft_latent")
            else projector.encode
        )
        zs = enc(torch.from_numpy(src).to(device)).cpu().numpy()
        zt = enc(torch.from_numpy(tgt).to(device)).cpu().numpy()
    if config.pair_space == "ambient":
        zs, zt = src, tgt
    logger.info("OT pairing space: %s", config.pair_space)
    Mc = ((zs[:, None] - zt[None]) ** 2).sum(-1)
    a_u, b_u = np.ones(len(zs)) / len(zs), np.ones(len(zt)) / len(zt)
    Pmat = ot.emd(a_u, b_u, Mc)
    match = Pmat.argmax(1)
    x0 = torch.from_numpy(src).to(device)
    x1 = torch.from_numpy(tgt[match]).to(device)

    # Graph-free chord refinement: an interior waypoint can travel arbitrarily far
    # off-manifold while the energy measured through the projector stays cheap, so
    # paths that grow beyond chord_refine_max_ratio revert to the chord.
    K_CHORD = 31
    tt = torch.linspace(0, 1, K_CHORD, device=device).view(1, -1, 1)
    chords = (1 - tt) * x0.unsqueeze(1) + tt * x1.unsqueeze(1)
    cr, _ = refine_paths(
        chords,
        projector,
        PathRefinementConfig(
            metric=WarpedPullbackMetricConfig(beta=config.beta, space="ambient"),
            steps=config.refine_steps,
            lr=config.refine_lr,
        ),
    )
    with torch.no_grad():
        cr = torch.cat(
            [
                cr[:, :1],
                projector(cr[:, 1:-1].reshape(-1, X.shape[1])).reshape(
                    cr.shape[0], -1, X.shape[1]
                ),
                cr[:, -1:],
            ],
            dim=1,
        )

        def _path_length(paths_):
            return (paths_[:, 1:] - paths_[:, :-1]).norm(dim=-1).sum(dim=1)

        len_cr, len_ch = _path_length(cr), _path_length(chords)
        bad = ~torch.isfinite(len_cr) | (
            len_cr > config.chord_refine_max_ratio * len_ch.clamp_min(1e-8)
        )
        if bad.any():
            cr[bad] = chords[bad]
    path_arms = {"chord_refined_pairs": cr}

    xs = torch.from_numpy(src).to(device)
    xt = torch.from_numpy(tgt).to(device)
    Pamb = ot.emd(
        np.ones(len(src)) / len(src),
        np.ones(len(tgt)) / len(tgt),
        ((src[:, None] - tgt[None]) ** 2).sum(-1),
    )
    vels = {
        "ot_cfm": train_velocity_cfm(
            xs,
            xt[torch.from_numpy(Pamb.argmax(1)).to(device)],
            X.shape[1],
            config,
            run,
            "ot_cfm",
            device,
        ),
        "i_cfm": train_velocity_cfm(
            xs,
            xt[torch.randperm(len(xt), device=device)],
            X.shape[1],
            config,
            run,
            "i_cfm",
            device,
        ),
    }

    g_rng = np.random.default_rng(config.seed + 7)
    n_ts = config.gaga_n_tsteps
    s_idx = g_rng.choice(
        len(src_all), max(2, int(config.gaga_test_frac * len(src_all))), replace=False
    )
    r_idx = g_rng.choice(
        len(real), max(2, int(config.gaga_test_frac * len(real))), replace=False
    )
    x_start = torch.from_numpy(src_all[s_idx]).to(device)
    real_test = real[r_idx]

    def traj_cloud(v, x):
        out = [x]
        h = 1.0 / (n_ts - 1)
        with torch.no_grad():
            for s_ in range(n_ts - 1):
                ts_ = torch.full((x.shape[0],), s_ * h, device=x.device)
                k1 = v(x, ts_)
                k2 = v(x + 0.5 * h * k1, ts_ + 0.5 * h)
                k3 = v(x + 0.5 * h * k2, ts_ + 0.5 * h)
                k4 = v(x + h * k3, ts_ + h)
                x = x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
                out.append(x)
        return torch.cat(out).cpu().numpy()

    n_match = int(len(r_idx))
    m_rng = np.random.default_rng(config.seed + 11)

    def w1_matched(X_, Y_):
        def sub(A):
            return (
                A
                if len(A) <= n_match
                else A[m_rng.choice(len(A), n_match, replace=False)]
            )

        vals = [w1(sub(X_), sub(Y_)) for _ in range(config.gaga_n_match_rep)]
        return {
            "W1_traj_matched": float(np.mean(vals)),
            "W1_traj_matched_std": float(np.std(vals)),
        }

    def _proj_cloud(c):
        with torch.no_grad():
            return (
                projector(torch.from_numpy(np.asarray(c, dtype=np.float32)).to(device))
                .cpu()
                .numpy()
            )

    gaga = {
        "n_start": int(len(s_idx)),
        "n_real": int(len(r_idx)),
        "n_tsteps": n_ts,
        "source": prev_w,
        "target": next_w,
        "held_out": config.held_out,
    }
    for k, v in vels.items():
        cloud = traj_cloud(v, x_start)
        gaga[k] = w1_matched(cloud, real_test)
        gaga[f"{k}_projected"] = w1_matched(_proj_cloud(cloud), real_test)

    p_idx = g_rng.choice(len(x0), min(len(x0), len(s_idx)), replace=False)
    lin = (
        torch.cat(
            [(1 - tau) * x0[p_idx] + tau * x1[p_idx] for tau in np.linspace(0, 1, n_ts)]
        )
        .cpu()
        .numpy()
    )
    gaga["linear"] = w1_matched(lin, real_test)
    gaga["linear_projected"] = w1_matched(_proj_cloud(lin), real_test)

    def _resample(paths_, m):
        K = paths_.shape[1]
        idx = torch.linspace(0, K - 1, m, device=paths_.device)
        lo = idx.floor().long().clamp(0, K - 1)
        hi = idx.ceil().long().clamp(0, K - 1)
        w_ = (idx - lo.to(idx.dtype)).view(1, -1, 1)
        return (1 - w_) * paths_[:, lo] + w_ * paths_[:, hi]

    for name, paths_ in path_arms.items():
        with torch.no_grad():
            sel = paths_[p_idx]
            cloud = _resample(sel, n_ts).reshape(-1, X.shape[1]).cpu().numpy()
        gaga[name] = w1_matched(cloud, real_test)
        gaga[f"{name}_projected"] = w1_matched(_proj_cloud(cloud), real_test)

    gaga["n_match"] = n_match
    gaga["n_match_rep"] = int(config.gaga_n_match_rep)

    evaluation = {
        "held_out": config.held_out,
        "source": prev_w,
        "target": next_w,
        "n_real": int(len(real)),
        "day_blind": bool(day_blind),
        "gaga_protocol": gaga,
    }
    run.save_json("evaluation.json", evaluation)
    logger.info(
        "GAGA protocol | matched at n=%s x%s: %s",
        gaga["n_match"],
        gaga["n_match_rep"],
        {
            k: round(v["W1_traj_matched"], 3)
            for k, v in gaga.items()
            if isinstance(v, dict) and "W1_traj_matched" in v
        },
    )


if __name__ == "__main__":
    main()

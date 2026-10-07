# Geometric Distillation from Diffusion Scores



**Geometric Distillation** turns a pretrained diffusion model into an explicit, continuous representation of your data's underlying geometry.

Recovering low-dimensional manifold structure from high-dimensional data typically requires building intermediate geometric surrogates (e.g., $k$-NN graphs, affinity kernels, or precomputed geodesic matrices). **Geometry Distillation asks whether these intermediate constructions are necessary at all.**

A diffusion model trained on data inherently captures manifold structure. Its small-noise score vectors point directly toward the data support, giving instant pseudo-projections. We distill these score-derived projections into a lightweight encoder–decoder or multi-chart neural atlas. This exposes **projection, intrinsic coordinates, local metrics, and volume elements** as explicit, reusable objects.

### Key Capabilities

* 🎯 **Out-of-sample Projection:** Map noisy or off-manifold points onto the learned support.
* 🧬 **Coordinate Discovery:** Extract intrinsic latent coordinates directly without $t$-SNE, UMAP, or PHATE.
* 📏 **Graph-Free Geodesics:** Compute exact geodesic paths by refining straight chords via the pullback metric.
* 🎲 **Volume-Guided Sampling:** Sample uniformly along the manifold using volume-guided Langevin dynamics.
* 🔮 **Population Transport:** Predict unseen intermediate populations (e.g., developmental biology time-courses) via geodesic midpoints.

---

## ⚡ Quickstart

### Installation

We use [`uv`](https://docs.astral.sh/uv/) for deterministic environment management, so after cloning the repository, all you need to do is run `uv sync`.

*Note: Device selection automatically defaults to `CUDA` $\rightarrow$ `Apple MPS` $\rightarrow$ `CPU`.*

### Minimal Verification Run

Run a quick test on synthetic manifolds (takes ~2 minutes):

```bash
uv run python -m experiments.toy_manifolds.scripts.reproduce \
    --config experiments/toy_manifolds/configs/reproduce_test.json

```

---

## 💡 How It Works

### 1. Pseudo-Projections from Score Models

Let $s_\theta(x, \sigma)$ be a pretrained, frozen diffusion score network. The small-noise score identity provides a local pseudo-target for the nearest-point projection $\pi_{\mathcal M}(x)$:

$$\hat\pi_{\theta,\sigma}(x) \;=\; x + \sigma^2\, s_\theta(x, \sigma) \;\approx\; \pi_{\mathcal M}(x) + \mathcal O(\sigma^2)$$

### 2. Student Distillation

We perturb off-manifold points ($x_\text{off} = x + \tau\xi$) and train a student projector $P_{\phi,\psi} = D_\psi \circ E_\phi$ using generic geometric regularization (no ground-truth manifold required):

$$\mathcal L = \mathcal L_\text{proj} + \lambda_1 \mathcal L_\text{fix} + \lambda_2 \mathcal L_\text{idem} + \lambda_3 \mathcal L_\text{orth} + \lambda_4 \mathcal L_\text{iso}$$

| Loss Term | Geometric Property Preserved |
| --- | --- |
| $\mathcal L_\text{fix}$ | **Fixpoint:** Points already on the manifold stay put. |
| $\mathcal L_\text{idem}$ | **Idempotency:** Projecting twice yields the same point ($P(P(x)) = P(x)$). |
| $\mathcal L_\text{orth}$ | **Orthogonality:** Residual vectors $(x - P(x))$ remain normal to the tangent space. |
| $\mathcal L_\text{iso}$ | **Isometry:** Encoder acts as a local isometry on the learned tangent space. |

Further specialized loss terms are part of this library, e.g., for training multi-chart atlases.

### 3. Derived Geometric Objects

Once trained, explicit differential geometry quantities are available analytically:

```
Projection          ──►  P(x) = D(E(x))
Intrinsic Latent    ──►  z = E(x)
Distance to Support ──►  d(x) = ½ ||x - P(x)||²
Pullback Metric     ──►  g(x) = J_E(x)ᵀ J_E(x)
Volume Element      ──►  dV = sqrt(det g(x))

```

---

## 💻 Usage Example

Distill geometry on custom data in under 30 lines of code:

```python
import torch
from torch.utils.data import DataLoader, TensorDataset
from geometry_distill import (
    ScoreNetwork, ScoreNetworkConfig, ScoreTrainer, ScoreTrainerConfig,
    LogUniformSigmaSchedule, LogUniformSigmaScheduleConfig,
    SingleChartProjector, SingleChartConfig,
    ProjectionTrainer, ProjectionTrainerConfig, ProjectionLossConfig,
    FlowPseudoProjector, FlowConfig, resolve_device
)

device = resolve_device()
X = torch.as_tensor(your_data, dtype=torch.float32) # Shape: (N, D)
loader = DataLoader(TensorDataset(X), batch_size=256, shuffle=True)

# 1. Train or load the pretrained score model (Teacher)
score = ScoreNetwork(ScoreNetworkConfig(x_dim=X.shape[1])).to(device)
ScoreTrainer(
    ScoreTrainerConfig(epochs=1000, patience=300),
    score,
    LogUniformSigmaSchedule(LogUniformSigmaScheduleConfig(sigma_min=0.01, sigma_max=0.5)),
).fit(loader)

# 2. Distill into an explicit neural projector (Student)
projector = SingleChartProjector(
    SingleChartConfig(x_dim=X.shape[1], k=2, hidden_dim=64) # k = intrinsic dimension
).to(device)

trainer = ProjectionTrainer(
    ProjectionTrainerConfig(epochs=1000),
    projector, score,
    FlowPseudoProjector(FlowConfig(sigmas=(0.05, 0.035, 0.02)), score),
    ProjectionLossConfig(lambda_fix=1.0, lambda_idem=1.0, lambda_orth=1.0),
)
trainer.fit(loader)

# 3. Query geometric properties
x_proj = projector(X.to(device))     # Nearest point projection
z      = projector.encode(X.to(device)) # Low-dimensional intrinsic coordinates

```

---

## 📊 Benchmark Results

### 1. Synthetic Single-Cell Trajectories (Splatter)

Geodesic paths refined via our pullback metric significantly outperform traditional graph-based shortest paths ($k$-NN) across different topological structures:

| Scenario | Path Length MSE ($\times 10^{-3}$) ↓ |  | Distance to Manifold ↓ |  |
| --- | --- | --- | --- | --- |
|  | **Ours** | $k\text{-NN-}15$ | **Ours** | $k\text{-NN-}15$ |
| **Linear** | **1.37 ± 0.21** | 13.58 ± 0.14 | **0.0209** | 0.0343 |
| **Bifurcation** | **0.39 ± 0.01** | 12.21 ± 0.09 | **0.0179** | 0.0293 |
| **Tree** | **1.69 ± 0.15** | 14.29 ± 0.08 | **0.0193** | 0.0295 |

### 2. Held-Out Population Prediction ($W_1$ Distance)

Predicting unobserved intermediate timepoints on single-cell developmental timecourses (NeurIPS 2022 Open Problems Benchmark & Embryoid Body data). Values represent 1-Wasserstein distance ($W_1 \downarrow$):

| Dataset | **Ours (Geodesic Midpoint)** | Linear Interp. | OT-CFM | I-CFM |
| --- | --- | --- | --- | --- |
| **Embryoid Body (EB)** | **10.76 ± 0.07** | **10.76 ± 0.05** | 11.90 ± 0.01 | 12.05 ± 0.01 |
| **CITE-seq (D=50)** | **18.14 ± 0.19** | **18.23 ± 0.19** | 21.36 ± 0.18 | 22.15 ± 0.25 |
| **Multiome (D=50)** | **19.24 ± 0.05** | 21.12 ± 0.07 | 22.87 ± 0.06 | 23.68 ± 0.05 |

---

## 🔬 Reproducing Paper Results

All experimental suites live under `experiments/`. Runs produce a `complete.json` marker and can be safely resumed with `--skip-existing`. Add `--dry-run` to view execution schedules.

| Campaign | Description | Entry Point |
| --- | --- | --- |
| **Toy Manifolds** | Analytic surfaces (hemisphere, saddle, paraboloid) | `experiments.toy_manifolds.scripts.reproduce` |
| **Synthetic Data** | Splatter topologies (linear, bifurcation, tree) | `experiments.synthetic_data.scripts.reproduce` |
| **Real Single-Cell** | EB time-course, CITE-seq, and Multiome datasets | `experiments.real_data.scripts.reproduce` |

### Step-by-Step Instructions

```bash
# Inspect planned runs
uv run python -m experiments.toy_manifolds.scripts.reproduce --dry-run

# Run full suite (3 manifolds x 5 seeds x 4 stages = 60 jobs)
uv run python -m experiments.toy_manifolds.scripts.reproduce

```

```bash
# Generate synthetic counts (Requires R with splatter)
Rscript experiments.synthetic_data.synthetic.data_generation.simulate_splatter.R

# Construct geometries & splits
uv run python -m experiments.synthetic_data.scripts.prepare_geometry

# Execute campaign
uv run python -m experiments.synthetic_data.scripts.reproduce

```

```bash
# Process Embryoid Body data
uv run python experiments/real_data/eb/prepare_eb_pca.py
uv run python -m experiments.real_data.scripts.prepare_eb_geometry

# Run benchmark evaluations
uv run python -m experiments.real_data.scripts.reproduce \
    --config experiments/real_data/configs/reproduce.json

```

### Generating Figures

Paper figures can be regenerated using the notebooks provided in `notebooks/`:

---

## 📁 Repository Structure

```text
geometry-distill/
├── src/geometry_distill/        # Core library package
│   ├── score/                   # Diffusion score architectures & DSM trainers
│   ├── projection/              # Single-chart & soft atlas projectors
│   ├── sampling/                # Volume elements & volume-guided Langevin sampling
│   └── geodesics/               # Pullback metrics & path optimization solvers
├── experiments/                 # Paper reproduction pipelines
│   ├── toy_manifolds/           # Analytic manifold benchmarks
│   ├── synthetic_data/          # Splatter simulation workflows
│   └── real_data/               # Single-cell time-course benchmarks
├── notebooks/                   # Jupyter notebooks for generating paper figures
├── pyproject.toml               # Package configuration and dependencies
└── README.md

```

---
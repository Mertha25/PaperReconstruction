# Lightweight Transformer Surrogate for Sparse-Sensor Thermal Field Reconstruction in Multi-Pass Heat Exchangers

Official code for the paper presented at the **17th International Conference on Emerging Ubiquitous Systems and Pervasive Networks (EUSPN 2026)**, October 28–30, 2026, Almaty, Kazakhstan — published in *Procedia Computer Science*.

**Authors:** Mamerthe W. Mubake¹, Selain K. Kasereka¹, Kang Luo², Kyandoghere Kyamakya¹,*
¹ Institute for Smart Systems Technologies, University of Klagenfurt, Austria
² School of Energy Science and Engineering, Harbin Institute of Technology, China

---

## Project overview

This repository introduces a lightweight **Tiny Vision Transformer (Tiny ViT)** (~0.75M parameters) that reconstructs the full 2D temperature field of a multi-pass heat exchanger in real time from a small number of point sensors (sparse sensing). The model is compared against four baselines: RBF interpolation, MLP, U-Net, and a lightweight Fourier Neural Operator (FNO-lite).

The **HEX dataset** is generated synthetically by solving the steady-state convection-diffusion equation on a rectangular 3-pass domain (alternating flow direction), with Latin Hypercube Sampling over 4 physical parameters (inlet temperature, cold-wall temperature, flow velocity, thermal conductivity).

## Repository structure

```
PaperReconstruction/
├── HEX_dataset/                 # HEX dataset (64x128 grid)
│   ├── train/                   # 4000 samples (sample_XXXX.mat)
│   ├── val/                     # 500 samples
│   ├── test/                    # 550 samples (500 LHS sample_XXXX.mat + 50 hard_XXXX.mat)
│   └── pod_artifacts/           # POD modes, mean, std, sensor positions
├── checkpoints/
│   ├── mlp_best.pth              # Trained MLP weights (best model)
│   ├── unet_best.pth             # Trained U-Net weights
│   ├── vit_best.pth              # Trained Tiny ViT weights
│   └── fno_best.pth              # Trained FNO-lite weights
├── code/
│   ├── generate_hex.py           # Dataset generation (PDE solver + LHS)
│   ├── training.ipynb            # Notebook 2 — model definitions and training (4 models)
│   └── evaluation.ipynb          # Notebook 3 — metrics, robustness study, paper figures
├── logs/                         # Per-epoch training curves (loss, MAE, RMSE) as CSV
├── results/
│   ├── results_table.csv / .tex  # Main results table (single source of truth for the paper)
│   ├── reconstruction_comparison.png
│   ├── error_maps.png
│   └── sensor_study.png
└── training_curves.png
```

## The HEX dataset

Each `.mat` sample corresponds to solving the steady-state convection-diffusion equation:

```
rho * cp * (ux * dT/dx + uy * dT/dy) = k * (d²T/dx² + d²T/dy²)
```

on a **0.3 m × 0.1 m** domain discretized on a **64×128 grid**, split into **3 horizontal passes** with alternating flow direction (top → left-to-right, middle → right-to-left, bottom → left-to-right).

**Variable parameters (Latin Hypercube Sampling):**

| Parameter | Symbol | Range |
|---|---|---|
| Hot fluid inlet temperature | `T_in` | 340 – 400 K |
| Cold wall temperature | `T_cold` | 280 – 300 K |
| Flow velocity | `u0` | 0.01 – 0.1 m/s |
| Thermal conductivity | `k` | 1 – 100 W/(m·K) |

**Input encoding (Strategy A)** — each sample is encoded as a `(3, 64, 128)` tensor:
- Channel 0: temperature values at sensor positions (0 elsewhere)
- Channel 1: binary mask of sensor positions
- Channel 2: `T_in` broadcast over the entire grid (operating condition)

**Contents of each `.mat` file:**
- `T_field` (64×128) — full temperature field [K] (ground truth)
- `X_input` (3×64×128) — encoded input (Strategy A)
- `u_obs` (M,) — sensor readings [K]
- `u_pos` (M,2) — sensor positions
- `params` (4,) — `[T_in, T_cold, u0, k]`

**Split:** all **5000** LHS parameter vectors are drawn in a single pass and partitioned into **4000 train / 500 val / 500 test**; no configuration appears in more than one split. In addition, **50 hard-case fields** (10 extreme configurations × 5 random sensor placements) are generated separately (seed 999) and stored in `test/` for the robustness study.

## Models

| Model | Type | Parameters | Role |
|---|---|---|---|
| RBF Interpolation | Classical interpolation (thin-plate-spline, regularized, physically clamped) | — | Non-learned baseline |
| MLP | Fully connected | ~27.9 M | Spatially "blind" baseline |
| U-Net | CNN encoder-decoder + skip connections | ~1.93 M | Strong DL baseline |
| **Tiny ViT (proposed)** | CNN + Transformer | **~0.75 M** | Main contribution |
| FNO-lite | Lightweight Fourier Neural Operator | ~1.18 M | Physics/spectral baseline |

The proposed **Tiny ViT** combines a CNN encoder (3 stages) that reduces the input to a 128→64-channel bottleneck, flattened into 128 tokens of dimension 64, processed by **3 Transformer layers (4 attention heads)**, then reconstructed by a CNN decoder with skip connections. Self-attention lets every spatial token directly attend to every other token — a key advantage for capturing the thermal coupling between the exchanger's 3 passes.

> **Note on the RBF baseline.** An unsmoothed cubic RBF is numerically unstable on steep high-Péclet fields and coincident sensor pixels (it can extrapolate to ~10⁴ K). The baseline therefore uses a thin-plate-spline kernel with a linear polynomial trend and a small regularization term, with the reconstruction clamped to the physical range `[T_MIN, T_MAX]`. This keeps RBF a learning-free lower bound without numerical blow-up.

**Training:** loss = `0.8 × MSE + 0.2 × MAE` on fields normalized to [0,1] (bounds `T_MIN=280K`, `T_MAX=400K`), AdamW optimizer + CosineAnnealingLR, up to 200 epochs, early stopping (patience=40), batch size 32. The dataset is cached in RAM once at load time so epochs are GPU-bound (≈45 min total for the four models on a T4).

## Main results

Reconstruction performance on the HEX test set (M=16 sensors, 500 LHS fields). Best accuracy per column in **bold**; `Params` bolded for the model with the fewest learned parameters. Inference times are indicative (GPU-dependent) and no speed advantage is claimed for any single learned model.

| Model | MAE (K) | RMSE (K) | Max Err. (K) | R² | Params (M) | Time (ms) |
|---|---|---|---|---|---|---|
| RBF Interpolation | 7.40 | 15.64 | 119.9 | 0.676 | — | 3.3 |
| MLP | 7.06 | 10.14 | 72.5 | 0.864 | 27.93 | 0.5 |
| U-Net | 0.60 | 1.16 | 32.6 | 0.9982 | 1.93 | 1.8 |
| **Tiny ViT (proposed)** | **0.41** | **0.72** | **25.7** | **0.9993** | **0.75** | 3.2 |
| FNO-lite | 3.24 | 7.24 | 113.2 | 0.9306 | 1.18 | 2.0 |

➡️ The Tiny ViT achieves the best reconstruction accuracy (MAE, RMSE, max error, R², and — in the paper — SSIM, HotMAE, HDR) with the **smallest parameter budget** of all learned models (61% fewer than the U-Net), while all learned models run in a few milliseconds (real-time capable).

**Robustness (MAE, K — full results in the paper's Table 2):**
- **Sensor dropout:** the Tiny ViT at M=8 (0.67 K) still outperforms the U-Net at M=12 (0.75 K).
- **Sensor noise:** the Tiny ViT keeps the lowest MAE at 1/3/5% noise (0.66 / 1.66 / 2.93 K).
- **Hard cases (50 extreme configs, Pe ≈ 0.99–986):** Tiny ViT 1.14 K, vs U-Net 1.55 K and FNO-lite 5.56 K.

Figures available in `results/`: visual reconstruction comparison (`reconstruction_comparison.png`), absolute error maps (`error_maps.png`), sensor-count sensitivity study (`sensor_study.png`), training curves (`training_curves.png`).

## Installation

```bash
git clone https://github.com/Mertha25/PaperReconstruction.git
cd PaperReconstruction
pip install numpy scipy matplotlib scikit-image torch pandas
```

## Usage

### 1. Regenerate the HEX dataset (optional — already provided in `HEX_dataset/`)

```bash
python code/generate_hex.py
```

### 2. Train the models

Open and run `code/training.ipynb` (Google Colab recommended, T4 GPU). All 4 models (MLP, U-Net, Tiny ViT, FNO-lite) are trained sequentially (~45 min total on a T4). Checkpoints are saved to `checkpoints/` and per-epoch logs to `logs/`.

### 3. Evaluate and reproduce the paper's figures

Open and run `code/evaluation.ipynb`: computes metrics (MAE, RMSE, MaxErr, R², SSIM, hotspot metrics), runs the robustness study (noise, sensor dropout, hard cases), and generates the LaTeX tables and figures used in the paper. Re-running it overwrites `results/results_table.csv` (the single source of truth for the paper tables).

The four pretrained checkpoints in `checkpoints/` allow the results to be reproduced directly without retraining.

## Citation

```bibtex
@inproceedings{mubake2026lightweight,
  title     = {Lightweight Transformer Surrogate for Sparse-Sensor Thermal
               Field Reconstruction in Multi-Pass Heat Exchangers},
  author    = {Mubake, Mamerthe W. and Kasereka, Selain K. and Luo, Kang
               and Kyamakya, Kyandoghere},
  booktitle = {Procedia Computer Science},
  note      = {The 17th International Conference on Emerging Ubiquitous
               Systems and Pervasive Networks (EUSPN 2026)},
  address   = {Almaty, Kazakhstan},
  year      = {2026}
}
```

## License

To be specified by the authors (no license is currently defined in the repository).

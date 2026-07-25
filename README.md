# A Lightweight Transformer Surrogate for Real-Time Thermal Field Reconstruction in a Simplified Multi-Pass Heat Exchanger from Sparse Sensors

Official code for the paper presented at the **17th International Conference on Emerging Ubiquitous Systems and Pervasive Networks (EUSPN 2026)**, October 28–30, 2026, Almaty, Kazakhstan — published in *Procedia Computer Science*.

**Authors:** Mamerthe Wabiwa Mubake¹, Selain K. Kasereka¹, Kang Luo², Kyandoghere Kyamakya¹,*
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
│   ├── test/                    # 500 samples (500 LHS + targeted "hard" cases)
│   └── pod_artifacts/           # POD modes, mean, std, sensor positions
├── checkpoints/
│   ├── vit_best.pth              # Trained Tiny ViT weights (best model)
│   └── fno_best.pth              # Trained FNO-lite weights
├── code/
│   ├── generate_hex.py           # Dataset generation (PDE solver + LHS)
│   ├── training.ipynb            # Notebook 2 — model definitions and training (4 models)
│   └── evaluation.ipynb          # Notebook 3 — metrics, robustness study, paper figures
├── logs/                         # Per-epoch training curves (loss, MAE, RMSE) as CSV
├── results/
│   ├── results_table.csv / .tex  # Main results table (as reported in the paper)
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

Split: **5000 samples → 4000 train / 500 val / 500 test** (including a subset of targeted "hard" cases used for the robustness study).

## Models

| Model | Type | Parameters | Role |
|---|---|---|---|
| RBF Interpolation | Classical interpolation | — | Non-learned baseline |
| MLP | Fully connected | ~26.9 M | Spatially "blind" baseline |
| U-Net | CNN encoder-decoder + skip connections | ~1.93 M | Strong DL baseline |
| **Tiny ViT (proposed)** | CNN + Transformer | **~0.75 M** | Main contribution |
| FNO-lite | Lightweight Fourier Neural Operator | ~1.18 M | Physics/spectral baseline |

The proposed **Tiny ViT** combines a CNN encoder (3 stages) that reduces the input to a 128→64-channel bottleneck, flattened into 128 tokens of dimension 64, processed by **3 Transformer layers (4 attention heads)**, then reconstructed by a CNN decoder with skip connections. Self-attention lets every spatial token directly attend to every other token — a key advantage for capturing the thermal coupling between the exchanger's 3 passes.

**Training:** loss = `0.8 × MSE + 0.2 × MAE` on fields normalized to [0,1] (bounds `T_MIN=280K`, `T_MAX=400K`), AdamW optimizer + CosineAnnealingLR, up to 200 epochs, early stopping (patience=40), batch size 32.

## Main results

Reconstruction performance on the HEX test set (M=16 sensors, 500 samples):

| Model | MAE (K) | RMSE (K) | Max Err. (K) | R² | Time (ms) |
|---|---|---|---|---|---|
| RBF Interpolation | 39777.61 | 39925.58 | 55476.45 | −2110163.25 | **2.14** |
| MLP | 7.12 | 10.07 | 70.40 | 0.8658 | 10.24 |
| U-Net | 0.615 | 1.183 | 31.47 | 0.9981 | 64.58 |
| **Tiny ViT (proposed)** | **0.368** | **0.628** | **21.34** | **0.9995** | 37.33 |
| FNO-lite | 3.98 | 7.90 | 113.07 | 0.9174 | 18.89 |

➡️ The Tiny ViT achieves the best reconstruction accuracy (MAE, RMSE, max error, R²) while remaining substantially lighter than U-Net and the MLP, with an inference time compatible with real-time use.

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

Open and run `code/evaluation.ipynb`: computes metrics (MAE, RMSE, MaxErr, R², SSIM, hotspot detection rate), runs the robustness study (noise, sensor dropout, hard cases), and generates the LaTeX tables and figures used in the paper.

The pretrained checkpoints (`checkpoints/vit_best.pth`, `checkpoints/fno_best.pth`) allow the results to be reproduced directly without retraining.

## Citation

```bibtex
@inproceedings{mubake2026lightweight,
  title     = {A Lightweight Transformer Surrogate for Real-Time Thermal Field
               Reconstruction in a Simplified Multi-Pass Heat Exchanger from
               Sparse Sensors},
  author    = {Mubake, Mamerthe Wabiwa and Kasereka, Selain K. and Luo, Kang
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

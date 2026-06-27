"""
SynthoGen AI — TabDDPM Training Script (Production v2)
=======================================================
A self-contained Tabular Denoising Diffusion Probabilistic Model (TabDDPM)
implemented from scratch in PyTorch. No external diffusion libraries needed.

v2 Improvements over v1:
    - ResidualBlock with LayerNorm (replaces plain Linear+SiLU)
    - FiLM conditioning for timestep (more expressive than addition)
    - AdamW optimizer with weight decay
    - Reproducibility via --seed
    - Cosine beta schedule option (often outperforms linear)
    - Batched generation for memory efficiency

Architecture:
    1. Preprocessing  — Label-encode categoricals, MinMax-scale numericals to [0, 1].
    2. Forward Process — Gradually corrupt clean data with Gaussian noise over T steps.
    3. Denoiser (Residual MLP + FiLM) — Predicts the noise added at each step.
    4. Reverse Process  — Start from pure noise, iteratively denoise to generate samples.
    5. Postprocessing  — Inverse-scale numericals, inverse-decode categoricals.

Usage:
    python models/train_tabddpm.py \\
        --dataset synthea \\
        --input data/processed/synthea_flattened.csv \\
        --output data/synthetic/tabddpm_synthea.csv \\
        --epochs 200 \\
        --num_rows 5000 \\
        --device cuda \\
        --seed 42
"""

import argparse
import json
import logging
import math
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train_tabddpm")


# ============================================================================
# 1. PREPROCESSING: Encode & Normalize tabular data for diffusion
# ============================================================================

class TabularPreprocessor:
    """
    Handles the conversion between raw DataFrames and normalized tensors.

    - Categorical columns -> LabelEncoded integers -> scaled to [0, 1]
    - Numerical columns  -> MinMax scaled to [0, 1]

    All parameters are saved so we can perfectly inverse-transform
    the generated samples back to the original data space.
    """

    def __init__(self):
        self.num_cols = []
        self.cat_cols = []
        self.col_order = []
        self.num_min = {}
        self.num_max = {}
        self.cat_classes = {}
        self.cat_n_classes = {}

    def fit_transform(self, df):
        """Fit on a DataFrame and return a normalized numpy array."""
        df = df.copy()

        self.cat_cols = df.select_dtypes(
            include=["object", "category", "bool"]
        ).columns.tolist()
        self.num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        self.col_order = self.num_cols + self.cat_cols

        transformed_parts = []

        # --- Numerical columns ---
        for col in self.num_cols:
            vals = df[col].values.astype(np.float64)
            # Replace any remaining NaNs with column median
            if np.any(np.isnan(vals)):
                median_val = float(np.nanmedian(vals))
                vals = np.where(np.isnan(vals), median_val, vals)
            col_min = float(np.nanmin(vals))
            col_max = float(np.nanmax(vals))
            self.num_min[col] = col_min
            self.num_max[col] = col_max
            denom = col_max - col_min
            if denom < 1e-12:
                denom = 1.0
            scaled = (vals - col_min) / denom
            transformed_parts.append(scaled.reshape(-1, 1))

        # --- Categorical columns ---
        for col in self.cat_cols:
            classes = np.sort(df[col].dropna().unique())
            self.cat_classes[col] = classes
            self.cat_n_classes[col] = len(classes)
            class_to_idx = {c: i for i, c in enumerate(classes)}
            encoded = df[col].map(class_to_idx).fillna(0).values.astype(np.float64)
            n = max(len(classes) - 1, 1)
            scaled = encoded / n
            transformed_parts.append(scaled.reshape(-1, 1))

        return np.hstack(transformed_parts).astype(np.float32)

    def inverse_transform(self, data):
        """Convert a normalized numpy array back to a DataFrame."""
        df_out = pd.DataFrame()
        col_idx = 0

        for col in self.num_cols:
            vals = data[:, col_idx]
            col_min = self.num_min[col]
            col_max = self.num_max[col]
            denom = col_max - col_min
            if denom < 1e-12:
                denom = 1.0
            df_out[col] = np.clip(vals, 0.0, 1.0) * denom + col_min
            col_idx += 1

        for col in self.cat_cols:
            vals = data[:, col_idx]
            classes = self.cat_classes[col]
            n = max(len(classes) - 1, 1)
            indices = np.clip(np.round(vals * n), 0, len(classes) - 1).astype(int)
            df_out[col] = classes[indices]
            col_idx += 1

        return df_out[self.col_order]

    def save(self, path):
        """Save preprocessor config to JSON."""
        config = {
            "num_cols": self.num_cols, "cat_cols": self.cat_cols,
            "col_order": self.col_order, "num_min": self.num_min,
            "num_max": self.num_max,
            "cat_classes": {k: v.tolist() for k, v in self.cat_classes.items()},
            "cat_n_classes": self.cat_n_classes,
        }
        with open(path, "w") as f:
            json.dump(config, f, indent=2)

    def load(self, path):
        """Load preprocessor config from JSON."""
        with open(path, "r") as f:
            config = json.load(f)
        self.num_cols = config["num_cols"]
        self.cat_cols = config["cat_cols"]
        self.col_order = config["col_order"]
        self.num_min = config["num_min"]
        self.num_max = config["num_max"]
        self.cat_classes = {k: np.array(v) for k, v in config["cat_classes"].items()}
        self.cat_n_classes = config["cat_n_classes"]


# ============================================================================
# 2. BUILDING BLOCKS
# ============================================================================

class SinusoidalPosEmb(nn.Module):
    """Sinusoidal positional embedding for diffusion timesteps."""

    def __init__(self, dim):
        super(SinusoidalPosEmb, self).__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000.0) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device).float() * -emb)
        emb = t.float().unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        return emb


class ResidualBlock(nn.Module):
    """
    Pre-norm residual block with LayerNorm.
    Uses a two-layer MLP with a skip connection for stable deep networks.
    """

    def __init__(self, d_in, d_out, dropout=0.0):
        super(ResidualBlock, self).__init__()
        self.norm = nn.LayerNorm(d_in)
        self.linear1 = nn.Linear(d_in, d_out)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_out, d_out)
        if d_in != d_out:
            self.shortcut = nn.Linear(d_in, d_out, bias=False)
        else:
            self.shortcut = None

    def forward(self, x):
        residual = x if self.shortcut is None else self.shortcut(x)
        h = self.norm(x)
        h = self.act(self.linear1(h))
        h = self.dropout(h)
        h = self.linear2(h)
        return residual + h


# ============================================================================
# 3. RESIDUAL MLP DENOISER WITH FiLM CONDITIONING
# ============================================================================

class ResidualMLPDenoiser(nn.Module):
    """
    Production-quality denoiser for tabular diffusion.

    Key design choices:
      - ResidualBlock with LayerNorm for stable deep training
      - FiLM conditioning: timestep controls per-feature scale+shift
        (more expressive than simple additive conditioning)
      - Separate input/output projections
    """

    def __init__(self, d_in, d_hidden=256, n_layers=4, dropout=0.0):
        super(ResidualMLPDenoiser, self).__init__()

        # Timestep embedding → FiLM parameters (scale + shift per layer)
        self.time_emb = nn.Sequential(
            SinusoidalPosEmb(d_hidden),
            nn.Linear(d_hidden, d_hidden),
            nn.SiLU(),
        )
        # One FiLM projection per residual layer (outputs scale + shift)
        self.film_projections = nn.ModuleList([
            nn.Linear(d_hidden, d_hidden * 2) for _ in range(n_layers)
        ])

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.SiLU(),
        )

        # Residual backbone
        self.res_blocks = nn.ModuleList([
            ResidualBlock(d_hidden, d_hidden, dropout=dropout)
            for _ in range(n_layers)
        ])

        # Output projection
        self.output_norm = nn.LayerNorm(d_hidden)
        self.output_proj = nn.Linear(d_hidden, d_in)

    def forward(self, x, t):
        # Timestep embedding
        t_emb = self.time_emb(t)                        # (B, d_hidden)

        # Project input
        h = self.input_proj(x)                           # (B, d_hidden)

        # Residual blocks with FiLM conditioning
        for res_block, film_proj in zip(self.res_blocks, self.film_projections):
            # FiLM: Feature-wise Linear Modulation
            film_params = film_proj(t_emb)               # (B, d_hidden * 2)
            scale, shift = film_params.chunk(2, dim=-1)  # each (B, d_hidden)
            h = h * (1.0 + scale) + shift                # modulate
            h = res_block(h)                             # residual transform

        # Output
        h = self.output_norm(h)
        return self.output_proj(h)                       # (B, d_in)


# ============================================================================
# 4. GAUSSIAN DIFFUSION PROCESS (supports linear & cosine schedules)
# ============================================================================

class GaussianDiffusion:
    """
    Gaussian diffusion with linear or cosine noise schedule.

    Forward:  q(x_t | x_0) = N(x_t; sqrt(alpha_bar_t) * x_0, (1 - alpha_bar_t) * I)
    Reverse:  p(x_{t-1} | x_t) = N(x_{t-1}; mu_theta(x_t, t), sigma^2_t * I)
    """

    def __init__(self, T=1000, schedule="linear", device="cpu"):
        self.T = T
        self.device = device

        if schedule == "cosine":
            # Cosine schedule (Nichol & Dhariwal, 2021)
            steps = torch.arange(T + 1, dtype=torch.float64, device=device)
            f_t = torch.cos(((steps / T) + 0.008) / 1.008 * (math.pi / 2)) ** 2
            alpha_bars = f_t / f_t[0]
            betas = 1.0 - (alpha_bars[1:] / alpha_bars[:-1])
            betas = torch.clamp(betas, min=1e-6, max=0.999).float()
        else:
            # Linear schedule
            betas = torch.linspace(1e-4, 0.02, T, dtype=torch.float32, device=device)

        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)

        self.betas = betas
        self.alphas = alphas
        self.alpha_bars = alpha_bars
        self.sqrt_alpha_bars = torch.sqrt(alpha_bars)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1.0 - alpha_bars)

    def q_sample(self, x_0, t, noise=None):
        """Forward process: add noise to x_0 at timestep t."""
        if noise is None:
            noise = torch.randn_like(x_0)
        sqrt_ab = self.sqrt_alpha_bars[t].unsqueeze(1)
        sqrt_1_ab = self.sqrt_one_minus_alpha_bars[t].unsqueeze(1)
        return sqrt_ab * x_0 + sqrt_1_ab * noise, noise

    @torch.no_grad()
    def p_sample(self, model, x_t, t_val, batch_size):
        """Single reverse step: denoise x_t -> x_{t-1}."""
        t_tensor = torch.full(
            (batch_size,), t_val, device=self.device, dtype=torch.long
        )
        betas_t = self.betas[t_val]
        sqrt_one_minus_ab = self.sqrt_one_minus_alpha_bars[t_val]
        sqrt_recip_alpha = 1.0 / torch.sqrt(self.alphas[t_val])

        pred_noise = model(x_t, t_tensor)
        mean = sqrt_recip_alpha * (
            x_t - (betas_t / sqrt_one_minus_ab) * pred_noise
        )

        if t_val > 0:
            sigma = torch.sqrt(betas_t)
            return mean + sigma * torch.randn_like(x_t)
        return mean

    @torch.no_grad()
    def sample(self, model, n_samples, d_in, batch_gen=2048):
        """
        Full reverse process with batched generation for memory efficiency.
        Generates in chunks of `batch_gen` to avoid GPU OOM on large requests.
        """
        model.eval()
        all_samples = []
        remaining = n_samples

        while remaining > 0:
            chunk = min(remaining, batch_gen)
            x = torch.randn(chunk, d_in, device=self.device)

            for t in reversed(range(self.T)):
                x = self.p_sample(model, x, t, chunk)

                if t % 200 == 0:
                    logger.info(
                        "  Sampling chunk %d/%d | step %d/%d",
                        n_samples - remaining + chunk, n_samples,
                        self.T - t, self.T,
                    )

            all_samples.append(x)
            remaining -= chunk

        return torch.cat(all_samples, dim=0)


# ============================================================================
# 5. TRAINING LOOP
# ============================================================================

def train_diffusion(data_tensor, model, diffusion, epochs, batch_size, lr, device):
    """Train the denoiser to predict noise. Returns loss history."""

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    dataset = TensorDataset(data_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    loss_history = []
    best_loss = float("inf")
    best_state = None
    model.train()

    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        n_batches = 0

        for (batch,) in loader:
            batch = batch.to(device)
            B = batch.size(0)

            t = torch.randint(0, diffusion.T, (B,), device=device)
            x_noisy, noise = diffusion.q_sample(batch, t)
            pred_noise = model(x_noisy, t)
            loss = F.mse_loss(pred_noise, noise)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = total_loss / max(n_batches, 1)
        loss_history.append(avg_loss)

        # Track best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 10 == 0 or epoch == 1:
            current_lr = scheduler.get_last_lr()[0]
            logger.info(
                "Epoch %4d/%d | Loss: %.6f | Best: %.6f | LR: %.2e",
                epoch, epochs, avg_loss, best_loss, current_lr,
            )

    # Restore best model weights
    if best_state is not None:
        model.load_state_dict(best_state)
        logger.info("Restored best model weights (loss=%.6f).", best_loss)

    return loss_history


# ============================================================================
# 6. REPRODUCIBILITY
# ============================================================================

def set_seed(seed):
    """Set all random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info("Random seed set to %d.", seed)


# ============================================================================
# 7. CLI & MAIN
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="Train TabDDPM on a clinical CSV")
    p.add_argument("--dataset", type=str, required=True,
                   help="Dataset name tag (synthea / diabetes / framingham)")
    p.add_argument("--input", type=str, required=True,
                   help="Path to processed CSV file")
    p.add_argument("--output", type=str, required=True,
                   help="Path to save generated synthetic CSV")
    p.add_argument("--epochs", type=int, default=200,
                   help="Training epochs (default: 200)")
    p.add_argument("--batch_size", type=int, default=256,
                   help="Mini-batch size (default: 256)")
    p.add_argument("--lr", type=float, default=1e-3,
                   help="Learning rate (default: 1e-3)")
    p.add_argument("--timesteps", type=int, default=1000,
                   help="Number of diffusion timesteps T (default: 1000)")
    p.add_argument("--schedule", type=str, default="cosine",
                   choices=["linear", "cosine"],
                   help="Noise schedule type (default: cosine)")
    p.add_argument("--hidden_dim", type=int, default=256,
                   help="Denoiser hidden layer width (default: 256)")
    p.add_argument("--n_layers", type=int, default=4,
                   help="Number of residual layers (default: 4)")
    p.add_argument("--dropout", type=float, default=0.0,
                   help="Dropout rate in denoiser (default: 0.0)")
    p.add_argument("--num_rows", type=int, default=None,
                   help="Synthetic rows to generate (default: same as input)")
    p.add_argument("--device", type=str, default=None,
                   help="Device: 'cuda' or 'cpu' (default: auto-detect)")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for reproducibility (default: 42)")
    p.add_argument("--save_model", type=str, default=None,
                   help="Model save path (default: saved_models/tabddpm_<dataset>.pt)")
    return p.parse_args()


def main():
    args = parse_args()

    # ------------------------------------------------------------------
    # 1. Validate & Setup
    # ------------------------------------------------------------------
    if not os.path.isfile(args.input):
        logger.error("Input file not found: %s", args.input)
        logger.error("Make sure data_prep.py has been run first.")
        sys.exit(1)

    set_seed(args.seed)

    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    logger.info("Using device: %s", device)
    if device == "cuda":
        logger.info("GPU: %s", torch.cuda.get_device_name(0))

    logger.info("=" * 65)
    logger.info("  TabDDPM v2 (Production) — Dataset: %s", args.dataset.upper())
    logger.info("=" * 65)

    # ------------------------------------------------------------------
    # 2. Load data
    # ------------------------------------------------------------------
    logger.info("Loading data from: %s", args.input)
    real_data = pd.read_csv(args.input)
    logger.info("Loaded %d rows x %d columns", *real_data.shape)

    num_rows = args.num_rows if args.num_rows else len(real_data)

    # ------------------------------------------------------------------
    # 3. Preprocess
    # ------------------------------------------------------------------
    logger.info("Preprocessing: encoding categoricals, scaling numericals ...")
    preprocessor = TabularPreprocessor()
    data_np = preprocessor.fit_transform(real_data)
    d_in = data_np.shape[1]
    logger.info("Preprocessed shape: %s (d_in = %d)", data_np.shape, d_in)
    logger.info("  Numerical cols  (%d): %s", len(preprocessor.num_cols), preprocessor.num_cols)
    logger.info("  Categorical cols (%d): %s", len(preprocessor.cat_cols), preprocessor.cat_cols)

    data_tensor = torch.from_numpy(data_np).to(device)

    # ------------------------------------------------------------------
    # 4. Initialize fresh model + diffusion
    # ------------------------------------------------------------------
    logger.info(
        "Initializing ResidualMLPDenoiser (hidden=%d, layers=%d, dropout=%.2f)",
        args.hidden_dim, args.n_layers, args.dropout,
    )
    model = ResidualMLPDenoiser(
        d_in=d_in,
        d_hidden=args.hidden_dim,
        n_layers=args.n_layers,
        dropout=args.dropout,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    logger.info("Model parameters: %s", "{:,}".format(total_params))

    diffusion = GaussianDiffusion(
        T=args.timesteps, schedule=args.schedule, device=device,
    )
    logger.info(
        "Diffusion: T=%d, schedule=%s", args.timesteps, args.schedule,
    )

    # ------------------------------------------------------------------
    # 5. Train
    # ------------------------------------------------------------------
    logger.info("Training started ...")
    t0 = time.time()
    loss_history = train_diffusion(
        data_tensor=data_tensor, model=model, diffusion=diffusion,
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        device=device,
    )
    elapsed = time.time() - t0
    logger.info("Training complete in %.1f seconds (%.1f min).", elapsed, elapsed / 60)

    # ------------------------------------------------------------------
    # 6. Generate
    # ------------------------------------------------------------------
    logger.info("Generating %d synthetic rows ...", num_rows)
    t0_gen = time.time()
    synthetic_tensor = diffusion.sample(model, num_rows, d_in)
    gen_elapsed = time.time() - t0_gen
    logger.info("Generation complete in %.1f seconds.", gen_elapsed)

    synthetic_np = np.clip(synthetic_tensor.cpu().numpy(), 0.0, 1.0)
    synthetic_df = preprocessor.inverse_transform(synthetic_np)

    # ------------------------------------------------------------------
    # 7. Save outputs
    # ------------------------------------------------------------------
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    synthetic_df.to_csv(args.output, index=False)
    logger.info("Synthetic data saved -> %s", args.output)

    model_path = args.save_model or os.path.join(
        "saved_models", "tabddpm_{}.pt".format(args.dataset)
    )
    os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "d_in": d_in, "hidden_dim": args.hidden_dim,
        "n_layers": args.n_layers, "dropout": args.dropout,
        "timesteps": args.timesteps, "schedule": args.schedule,
        "loss_history": loss_history, "dataset": args.dataset,
        "seed": args.seed,
    }, model_path)
    logger.info("Model checkpoint saved -> %s", model_path)

    prep_path = model_path.replace(".pt", "_preprocessor.json")
    preprocessor.save(prep_path)
    logger.info("Preprocessor config saved -> %s", prep_path)

    # ------------------------------------------------------------------
    # 8. Sanity check
    # ------------------------------------------------------------------
    logger.info("")
    logger.info("--- SANITY CHECK ---")
    for col in preprocessor.num_cols[:5]:
        r_mean, r_std = real_data[col].mean(), real_data[col].std()
        s_mean, s_std = synthetic_df[col].mean(), synthetic_df[col].std()
        logger.info(
            "  %-25s  real: mu=%.2f std=%.2f  |  synth: mu=%.2f std=%.2f",
            col, r_mean, r_std, s_mean, s_std,
        )

    logger.info("=" * 65)
    logger.info("  TabDDPM v2 [%s] — DONE", args.dataset.upper())
    logger.info("=" * 65)


if __name__ == "__main__":
    main()

"""
SynthoGen AI — TabSyn Training Script (Production)
====================================================
State-of-the-art Latent Diffusion Model for Tabular Data Synthesis.

Why TabSyn > TabDDPM:
    TabDDPM runs diffusion directly on raw (normalized) data space.
    TabSyn first trains a VAE to compress data into a smooth, low-dimensional
    latent space, then runs diffusion THERE. The smooth latent space makes
    the diffusion model's job dramatically easier, producing higher-quality
    synthetic data with better column correlations.

Architecture (Two-Phase Training):
    Phase 1 — VAE:
        Encoder: data -> ResidualMLP -> (mu, logvar) -> reparameterize -> z
        Decoder: z -> ResidualMLP -> reconstructed data
        Loss:    MSE_reconstruction + beta * KL_divergence
        (beta uses linear warmup to avoid posterior collapse)

    Phase 2 — Latent Diffusion:
        Freeze the VAE encoder.
        Encode ALL training data into latent codes z.
        Train a DDPM denoiser (ResidualMLP + FiLM) on these latent codes.

    Generation:
        1. Sample latent codes from reverse diffusion
        2. Decode with frozen VAE decoder
        3. Inverse-transform to original data space

Usage:
    python models/train_tabsyn.py \\
        --dataset synthea \\
        --input data/processed/synthea_flattened.csv \\
        --output data/synthetic/tabsyn_synthea.csv \\
        --vae_epochs 100 \\
        --diff_epochs 200 \\
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
logger = logging.getLogger("train_tabsyn")


# ============================================================================
# 1. PREPROCESSING (identical to TabDDPM — self-contained for portability)
# ============================================================================

class TabularPreprocessor:
    """
    Handles conversion between raw DataFrames and [0,1]-normalized tensors.
    Categorical -> LabelEncode -> scale to [0,1].
    Numerical   -> MinMax scale to [0,1].
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
        df = df.copy()
        self.cat_cols = df.select_dtypes(
            include=["object", "category", "bool"]
        ).columns.tolist()
        self.num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        self.col_order = self.num_cols + self.cat_cols
        transformed_parts = []

        for col in self.num_cols:
            vals = df[col].values.astype(np.float64)
            if np.any(np.isnan(vals)):
                vals = np.where(np.isnan(vals), float(np.nanmedian(vals)), vals)
            col_min, col_max = float(np.nanmin(vals)), float(np.nanmax(vals))
            self.num_min[col] = col_min
            self.num_max[col] = col_max
            denom = col_max - col_min
            if denom < 1e-12:
                denom = 1.0
            transformed_parts.append(((vals - col_min) / denom).reshape(-1, 1))

        for col in self.cat_cols:
            classes = np.sort(df[col].dropna().unique())
            self.cat_classes[col] = classes
            self.cat_n_classes[col] = len(classes)
            class_to_idx = {c: i for i, c in enumerate(classes)}
            encoded = df[col].map(class_to_idx).fillna(0).values.astype(np.float64)
            n = max(len(classes) - 1, 1)
            transformed_parts.append((encoded / n).reshape(-1, 1))

        return np.hstack(transformed_parts).astype(np.float32)

    def inverse_transform(self, data):
        df_out = pd.DataFrame()
        col_idx = 0
        for col in self.num_cols:
            vals = data[:, col_idx]
            denom = self.num_max[col] - self.num_min[col]
            if denom < 1e-12:
                denom = 1.0
            df_out[col] = np.clip(vals, 0.0, 1.0) * denom + self.num_min[col]
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

class ResidualBlock(nn.Module):
    """Pre-norm residual block with LayerNorm + SiLU + skip connection."""

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
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


# ============================================================================
# 3. TABULAR VAE (Phase 1)
# ============================================================================

class TabularVAE(nn.Module):
    """
    Variational Autoencoder for tabular data.

    The encoder maps normalized tabular data into a smooth Gaussian latent
    space via the reparameterization trick. The decoder reconstructs the
    original data from latent codes. The KL divergence term ensures the
    latent space is smooth and continuous — ideal for diffusion.

    Architecture:
        Encoder: Input -> ResBlocks -> (mu, logvar)
        Decoder: z -> ResBlocks -> Sigmoid -> Output in [0, 1]
    """

    def __init__(self, d_in, d_latent, d_hidden=256, n_layers=3, dropout=0.0):
        super(TabularVAE, self).__init__()
        self.d_in = d_in
        self.d_latent = d_latent

        # --- Encoder ---
        enc_layers = [nn.Linear(d_in, d_hidden), nn.SiLU()]
        for _ in range(n_layers):
            enc_layers.append(ResidualBlock(d_hidden, d_hidden, dropout))
        self.encoder_backbone = nn.Sequential(*enc_layers)
        self.enc_mu = nn.Linear(d_hidden, d_latent)
        self.enc_logvar = nn.Linear(d_hidden, d_latent)

        # --- Decoder ---
        dec_layers = [nn.Linear(d_latent, d_hidden), nn.SiLU()]
        for _ in range(n_layers):
            dec_layers.append(ResidualBlock(d_hidden, d_hidden, dropout))
        dec_layers.append(nn.LayerNorm(d_hidden))
        dec_layers.append(nn.Linear(d_hidden, d_in))
        dec_layers.append(nn.Sigmoid())  # Output in [0, 1]
        self.decoder = nn.Sequential(*dec_layers)

    def encode(self, x):
        """Encode input to latent distribution parameters."""
        h = self.encoder_backbone(x)
        return self.enc_mu(h), self.enc_logvar(h)

    def reparameterize(self, mu, logvar):
        """Sample z from q(z|x) using the reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        """Decode latent code to reconstructed data in [0, 1]."""
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar


def vae_loss_fn(x_recon, x, mu, logvar, beta):
    """
    VAE loss = Reconstruction (MSE) + beta * KL Divergence.
    Beta warmup prevents posterior collapse in early epochs.
    """
    recon_loss = F.mse_loss(x_recon, x, reduction="mean")
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kl_loss, recon_loss.item(), kl_loss.item()


# ============================================================================
# 4. LATENT DENOISER (Phase 2 — Diffusion in Latent Space)
# ============================================================================

class LatentDenoiser(nn.Module):
    """
    Denoiser that operates in the VAE's latent space.
    Uses FiLM conditioning (Feature-wise Linear Modulation) for timestep.
    """

    def __init__(self, d_latent, d_hidden=256, n_layers=4, dropout=0.0):
        super(LatentDenoiser, self).__init__()

        self.time_emb = nn.Sequential(
            SinusoidalPosEmb(d_hidden),
            nn.Linear(d_hidden, d_hidden),
            nn.SiLU(),
        )

        self.film_projections = nn.ModuleList([
            nn.Linear(d_hidden, d_hidden * 2) for _ in range(n_layers)
        ])

        self.input_proj = nn.Sequential(
            nn.Linear(d_latent, d_hidden),
            nn.SiLU(),
        )

        self.res_blocks = nn.ModuleList([
            ResidualBlock(d_hidden, d_hidden, dropout) for _ in range(n_layers)
        ])

        self.output_norm = nn.LayerNorm(d_hidden)
        self.output_proj = nn.Linear(d_hidden, d_latent)

    def forward(self, z_noisy, t):
        t_emb = self.time_emb(t)
        h = self.input_proj(z_noisy)

        for res_block, film_proj in zip(self.res_blocks, self.film_projections):
            film_params = film_proj(t_emb)
            scale, shift = film_params.chunk(2, dim=-1)
            h = h * (1.0 + scale) + shift
            h = res_block(h)

        h = self.output_norm(h)
        return self.output_proj(h)


# ============================================================================
# 5. GAUSSIAN DIFFUSION (cosine schedule by default)
# ============================================================================

class GaussianDiffusion:
    """Gaussian diffusion with linear or cosine noise schedule."""

    def __init__(self, T=1000, schedule="cosine", device="cpu"):
        self.T = T
        self.device = device

        if schedule == "cosine":
            steps = torch.arange(T + 1, dtype=torch.float64, device=device)
            f_t = torch.cos(((steps / T) + 0.008) / 1.008 * (math.pi / 2)) ** 2
            alpha_bars = f_t / f_t[0]
            betas = 1.0 - (alpha_bars[1:] / alpha_bars[:-1])
            betas = torch.clamp(betas, min=1e-6, max=0.999).float()
        else:
            betas = torch.linspace(1e-4, 0.02, T, dtype=torch.float32, device=device)

        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        self.betas = betas
        self.alphas = alphas
        self.alpha_bars = alpha_bars
        self.sqrt_alpha_bars = torch.sqrt(alpha_bars)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1.0 - alpha_bars)

    def q_sample(self, x_0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_0)
        sqrt_ab = self.sqrt_alpha_bars[t].unsqueeze(1)
        sqrt_1_ab = self.sqrt_one_minus_alpha_bars[t].unsqueeze(1)
        return sqrt_ab * x_0 + sqrt_1_ab * noise, noise

    @torch.no_grad()
    def p_sample(self, model, x_t, t_val, batch_size):
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
            return mean + torch.sqrt(betas_t) * torch.randn_like(x_t)
        return mean

    @torch.no_grad()
    def sample(self, model, n_samples, d_latent, batch_gen=2048):
        model.eval()
        all_samples = []
        remaining = n_samples

        while remaining > 0:
            chunk = min(remaining, batch_gen)
            x = torch.randn(chunk, d_latent, device=self.device)

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
# 6. TRAINING — PHASE 1: VAE
# ============================================================================

def train_vae(vae, data_tensor, epochs, batch_size, lr, beta_max, device):
    """
    Train the VAE with beta-warmup (KL weight linearly increases).
    Returns (loss_history, recon_history, kl_history).
    """
    optimizer = torch.optim.AdamW(vae.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loader = DataLoader(
        TensorDataset(data_tensor), batch_size=batch_size, shuffle=True, drop_last=False
    )

    warmup_epochs = max(int(epochs * 0.3), 1)  # 30% warmup
    loss_history = []
    best_loss = float("inf")
    best_state = None
    vae.train()

    for epoch in range(1, epochs + 1):
        # Beta warmup: linearly increase from 0 to beta_max
        if epoch <= warmup_epochs:
            beta = beta_max * (epoch / warmup_epochs)
        else:
            beta = beta_max

        total_loss, total_recon, total_kl = 0.0, 0.0, 0.0
        n_batches = 0

        for (batch,) in loader:
            batch = batch.to(device)
            x_recon, mu, logvar = vae(batch)
            loss, recon_val, kl_val = vae_loss_fn(x_recon, batch, mu, logvar, beta)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(vae.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            total_recon += recon_val
            total_kl += kl_val
            n_batches += 1

        scheduler.step()
        avg_loss = total_loss / max(n_batches, 1)
        avg_recon = total_recon / max(n_batches, 1)
        avg_kl = total_kl / max(n_batches, 1)
        loss_history.append(avg_loss)

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_state = {k: v.cpu().clone() for k, v in vae.state_dict().items()}

        if epoch % 10 == 0 or epoch == 1:
            logger.info(
                "[VAE] Epoch %4d/%d | Loss: %.6f | Recon: %.6f | KL: %.6f | beta: %.4f",
                epoch, epochs, avg_loss, avg_recon, avg_kl, beta,
            )

    if best_state is not None:
        vae.load_state_dict(best_state)
        logger.info("[VAE] Restored best model (loss=%.6f).", best_loss)

    return loss_history


# ============================================================================
# 7. TRAINING — PHASE 2: LATENT DIFFUSION
# ============================================================================

def train_latent_diffusion(denoiser, latent_codes, diffusion, epochs, batch_size, lr, device):
    """Train the denoiser on VAE-encoded latent codes."""
    optimizer = torch.optim.AdamW(denoiser.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loader = DataLoader(
        TensorDataset(latent_codes), batch_size=batch_size, shuffle=True, drop_last=False
    )

    loss_history = []
    best_loss = float("inf")
    best_state = None
    denoiser.train()

    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        n_batches = 0

        for (batch,) in loader:
            batch = batch.to(device)
            B = batch.size(0)
            t = torch.randint(0, diffusion.T, (B,), device=device)
            z_noisy, noise = diffusion.q_sample(batch, t)
            pred_noise = denoiser(z_noisy, t)
            loss = F.mse_loss(pred_noise, noise)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(denoiser.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = total_loss / max(n_batches, 1)
        loss_history.append(avg_loss)

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_state = {k: v.cpu().clone() for k, v in denoiser.state_dict().items()}

        if epoch % 10 == 0 or epoch == 1:
            current_lr = scheduler.get_last_lr()[0]
            logger.info(
                "[Diffusion] Epoch %4d/%d | Loss: %.6f | Best: %.6f | LR: %.2e",
                epoch, epochs, avg_loss, best_loss, current_lr,
            )

    if best_state is not None:
        denoiser.load_state_dict(best_state)
        logger.info("[Diffusion] Restored best model (loss=%.6f).", best_loss)

    return loss_history


# ============================================================================
# 8. GENERATION PIPELINE
# ============================================================================

@torch.no_grad()
def generate_synthetic(vae, denoiser, diffusion, preprocessor, n_samples, device):
    """
    Full TabSyn generation pipeline:
        1. Sample latent codes from reverse diffusion
        2. Decode with frozen VAE decoder
        3. Inverse-transform to original data space
    """
    vae.eval()
    denoiser.eval()

    # Step 1: Sample latent codes
    logger.info("Step 1/3: Sampling %d latent codes via reverse diffusion ...", n_samples)
    latent_samples = diffusion.sample(denoiser, n_samples, vae.d_latent)

    # Step 2: Decode with VAE
    logger.info("Step 2/3: Decoding latent codes with VAE decoder ...")
    decoded = vae.decode(latent_samples)
    synthetic_np = np.clip(decoded.cpu().numpy(), 0.0, 1.0)

    # Step 3: Inverse transform
    logger.info("Step 3/3: Inverse-transforming to original data space ...")
    return preprocessor.inverse_transform(synthetic_np)


# ============================================================================
# 9. REPRODUCIBILITY
# ============================================================================

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info("Random seed set to %d.", seed)


# ============================================================================
# 10. CLI & MAIN
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Train TabSyn (Latent Diffusion) on a clinical CSV"
    )
    # --- Data ---
    p.add_argument("--dataset", type=str, required=True,
                   help="Dataset name tag (synthea / diabetes / framingham)")
    p.add_argument("--input", type=str, required=True,
                   help="Path to processed CSV file")
    p.add_argument("--output", type=str, required=True,
                   help="Path to save generated synthetic CSV")

    # --- VAE (Phase 1) ---
    p.add_argument("--vae_epochs", type=int, default=100,
                   help="VAE training epochs (default: 100)")
    p.add_argument("--vae_lr", type=float, default=1e-3,
                   help="VAE learning rate (default: 1e-3)")
    p.add_argument("--d_latent", type=int, default=None,
                   help="Latent dimension (default: same as d_in)")
    p.add_argument("--vae_hidden", type=int, default=256,
                   help="VAE hidden layer width (default: 256)")
    p.add_argument("--vae_layers", type=int, default=3,
                   help="VAE residual blocks (default: 3)")
    p.add_argument("--beta_max", type=float, default=0.01,
                   help="Max KL weight for VAE (default: 0.01)")

    # --- Diffusion (Phase 2) ---
    p.add_argument("--diff_epochs", type=int, default=200,
                   help="Diffusion training epochs (default: 200)")
    p.add_argument("--diff_lr", type=float, default=1e-3,
                   help="Diffusion learning rate (default: 1e-3)")
    p.add_argument("--timesteps", type=int, default=1000,
                   help="Number of diffusion timesteps (default: 1000)")
    p.add_argument("--schedule", type=str, default="cosine",
                   choices=["linear", "cosine"],
                   help="Noise schedule type (default: cosine)")
    p.add_argument("--diff_hidden", type=int, default=256,
                   help="Denoiser hidden width (default: 256)")
    p.add_argument("--diff_layers", type=int, default=4,
                   help="Denoiser residual layers (default: 4)")

    # --- General ---
    p.add_argument("--batch_size", type=int, default=256,
                   help="Mini-batch size (default: 256)")
    p.add_argument("--dropout", type=float, default=0.0,
                   help="Dropout rate (default: 0.0)")
    p.add_argument("--num_rows", type=int, default=None,
                   help="Synthetic rows to generate (default: same as input)")
    p.add_argument("--device", type=str, default=None,
                   help="Device: 'cuda' or 'cpu' (default: auto-detect)")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed (default: 42)")
    p.add_argument("--save_model", type=str, default=None,
                   help="Model save path (default: saved_models/tabsyn_<dataset>.pt)")
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
    logger.info("  TabSyn (Latent Diffusion) — Dataset: %s", args.dataset.upper())
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
    logger.info("Preprocessing ...")
    preprocessor = TabularPreprocessor()
    data_np = preprocessor.fit_transform(real_data)
    d_in = data_np.shape[1]
    d_latent = args.d_latent if args.d_latent else d_in
    logger.info("d_in = %d, d_latent = %d", d_in, d_latent)
    logger.info("  Numerical cols  (%d): %s", len(preprocessor.num_cols), preprocessor.num_cols)
    logger.info("  Categorical cols (%d): %s", len(preprocessor.cat_cols), preprocessor.cat_cols)

    data_tensor = torch.from_numpy(data_np).to(device)

    # ==================================================================
    # PHASE 1: TRAIN VAE
    # ==================================================================
    logger.info("")
    logger.info("=" * 65)
    logger.info("  PHASE 1: Training VAE (learning latent space)")
    logger.info("=" * 65)

    vae = TabularVAE(
        d_in=d_in, d_latent=d_latent, d_hidden=args.vae_hidden,
        n_layers=args.vae_layers, dropout=args.dropout,
    ).to(device)

    vae_params = sum(p.numel() for p in vae.parameters())
    logger.info("VAE parameters: %s", "{:,}".format(vae_params))

    t0 = time.time()
    vae_loss_history = train_vae(
        vae=vae, data_tensor=data_tensor, epochs=args.vae_epochs,
        batch_size=args.batch_size, lr=args.vae_lr,
        beta_max=args.beta_max, device=device,
    )
    vae_elapsed = time.time() - t0
    logger.info("VAE training complete in %.1f seconds.", vae_elapsed)

    # Validate VAE reconstruction quality
    vae.eval()
    with torch.no_grad():
        x_recon, _, _ = vae(data_tensor)
        recon_mse = F.mse_loss(x_recon, data_tensor).item()
    logger.info("VAE reconstruction MSE: %.6f", recon_mse)

    # ==================================================================
    # PHASE 2: ENCODE DATA & TRAIN LATENT DIFFUSION
    # ==================================================================
    logger.info("")
    logger.info("=" * 65)
    logger.info("  PHASE 2: Training Diffusion in Latent Space")
    logger.info("=" * 65)

    # Encode all training data with frozen VAE
    vae.eval()
    with torch.no_grad():
        mu, logvar = vae.encode(data_tensor)
        # Use the mean (mu) as the latent code — more stable than sampling
        latent_codes = mu.detach()
    logger.info("Encoded %d samples to latent space (shape: %s)", len(latent_codes), latent_codes.shape)

    # Initialize denoiser
    denoiser = LatentDenoiser(
        d_latent=d_latent, d_hidden=args.diff_hidden,
        n_layers=args.diff_layers, dropout=args.dropout,
    ).to(device)

    diff_params = sum(p.numel() for p in denoiser.parameters())
    logger.info("Denoiser parameters: %s", "{:,}".format(diff_params))

    diffusion = GaussianDiffusion(
        T=args.timesteps, schedule=args.schedule, device=device,
    )
    logger.info("Diffusion: T=%d, schedule=%s", args.timesteps, args.schedule)

    t0 = time.time()
    diff_loss_history = train_latent_diffusion(
        denoiser=denoiser, latent_codes=latent_codes, diffusion=diffusion,
        epochs=args.diff_epochs, batch_size=args.batch_size,
        lr=args.diff_lr, device=device,
    )
    diff_elapsed = time.time() - t0
    logger.info("Diffusion training complete in %.1f seconds.", diff_elapsed)

    # ==================================================================
    # PHASE 3: GENERATE SYNTHETIC DATA
    # ==================================================================
    logger.info("")
    logger.info("=" * 65)
    logger.info("  PHASE 3: Generating %d Synthetic Rows", num_rows)
    logger.info("=" * 65)

    t0_gen = time.time()
    synthetic_df = generate_synthetic(
        vae=vae, denoiser=denoiser, diffusion=diffusion,
        preprocessor=preprocessor, n_samples=num_rows, device=device,
    )
    gen_elapsed = time.time() - t0_gen
    logger.info("Generation complete in %.1f seconds.", gen_elapsed)

    # ------------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------------
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    synthetic_df.to_csv(args.output, index=False)
    logger.info("Synthetic data saved -> %s", args.output)

    model_path = args.save_model or os.path.join(
        "saved_models", "tabsyn_{}.pt".format(args.dataset)
    )
    os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
    torch.save({
        "vae_state_dict": vae.state_dict(),
        "denoiser_state_dict": denoiser.state_dict(),
        "d_in": d_in, "d_latent": d_latent,
        "vae_hidden": args.vae_hidden, "vae_layers": args.vae_layers,
        "diff_hidden": args.diff_hidden, "diff_layers": args.diff_layers,
        "dropout": args.dropout, "timesteps": args.timesteps,
        "schedule": args.schedule,
        "vae_loss_history": vae_loss_history,
        "diff_loss_history": diff_loss_history,
        "vae_recon_mse": recon_mse,
        "dataset": args.dataset, "seed": args.seed,
    }, model_path)
    logger.info("Model checkpoint saved -> %s", model_path)

    prep_path = model_path.replace(".pt", "_preprocessor.json")
    preprocessor.save(prep_path)
    logger.info("Preprocessor config saved -> %s", prep_path)

    # ------------------------------------------------------------------
    # Sanity check
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

    logger.info("")
    logger.info("--- TIMING SUMMARY ---")
    logger.info("  VAE training:       %.1f sec", vae_elapsed)
    logger.info("  Diffusion training: %.1f sec", diff_elapsed)
    logger.info("  Generation:         %.1f sec", gen_elapsed)
    logger.info("  Total:              %.1f sec", vae_elapsed + diff_elapsed + gen_elapsed)

    logger.info("=" * 65)
    logger.info("  TabSyn [%s] — DONE", args.dataset.upper())
    logger.info("=" * 65)


if __name__ == "__main__":
    main()

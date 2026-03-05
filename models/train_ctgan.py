"""
SynthoGen AI — CTGAN Training Script
======================================
Trains a fresh CTGANSynthesizer (from SDV) on a single processed CSV,
generates synthetic data, and saves both the model and the output.

Usage:
    python models/train_ctgan.py \
        --dataset diabetes \
        --input data/processed/diabetes_clean.csv \
        --output data/synthetic/ctgan_diabetes.csv \
        --epochs 300 \
        --num_rows 5000

CLI Arguments:
    --dataset        Name tag for logging (e.g. synthea, diabetes, framingham)
    --input          Path to the cleaned CSV produced by data_prep.py
    --output         Path where the synthetic CSV will be saved
    --epochs         Number of CTGAN training epochs (default: 300)
    --batch_size     Mini-batch size (default: 500)
    --num_rows       How many synthetic rows to generate (default: same as input)
    --save_model     Path to save the trained model (default: saved_models/ctgan_<dataset>.pkl)
    --cuda           Use GPU if available (flag)
"""

import argparse
import logging
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from sdv.metadata import SingleTableMetadata
from sdv.single_table import CTGANSynthesizer

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train_ctgan")


def parse_args():
    p = argparse.ArgumentParser(description="Train CTGAN on a clinical CSV")
    p.add_argument("--dataset", type=str, required=True,
                   help="Dataset name tag (synthea / diabetes / framingham)")
    p.add_argument("--input", type=str, required=True,
                   help="Path to processed CSV file")
    p.add_argument("--output", type=str, required=True,
                   help="Path to save generated synthetic CSV")
    p.add_argument("--epochs", type=int, default=300,
                   help="Training epochs (default: 300)")
    p.add_argument("--batch_size", type=int, default=500,
                   help="Mini-batch size (default: 500)")
    p.add_argument("--generator_dim", type=int, nargs="+", default=[256, 256],
                   help="Generator hidden layer dims (default: 256 256)")
    p.add_argument("--discriminator_dim", type=int, nargs="+", default=[256, 256],
                   help="Discriminator hidden layer dims (default: 256 256)")
    p.add_argument("--embedding_dim", type=int, default=128,
                   help="Embedding dimensionality (default: 128)")
    p.add_argument("--generator_lr", type=float, default=2e-4,
                   help="Generator learning rate (default: 2e-4)")
    p.add_argument("--discriminator_lr", type=float, default=2e-4,
                   help="Discriminator learning rate (default: 2e-4)")
    p.add_argument("--discriminator_steps", type=int, default=1,
                   help="Discriminator steps per generator step (default: 1)")
    p.add_argument("--pac", type=int, default=10,
                   help="PAC size for training (default: 10)")
    p.add_argument("--num_rows", type=int, default=None,
                   help="Synthetic rows to generate (default: same as input)")
    p.add_argument("--save_model", type=str, default=None,
                   help="Model save path (default: saved_models/ctgan_<dataset>.pkl)")
    p.add_argument("--cuda", action="store_true",
                   help="Use GPU if available")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for reproducibility (default: 42)")
    return p.parse_args()


def main():
    args = parse_args()

    # ------------------------------------------------------------------
    # 1. Validate input & set seed
    # ------------------------------------------------------------------
    if not os.path.isfile(args.input):
        logger.error("Input file not found: %s", args.input)
        logger.error("Make sure data_prep.py has been run first.")
        sys.exit(1)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    logger.info("Random seed set to %d.", args.seed)

    logger.info("=" * 60)
    logger.info("CTGAN Training — Dataset: %s", args.dataset.upper())
    logger.info("=" * 60)

    # ------------------------------------------------------------------
    # 2. Load data
    # ------------------------------------------------------------------
    logger.info("Loading data from: %s", args.input)
    real_data = pd.read_csv(args.input)
    logger.info("Loaded %d rows × %d columns", *real_data.shape)
    logger.info("Columns: %s", list(real_data.columns))

    num_rows = args.num_rows if args.num_rows else len(real_data)

    # ------------------------------------------------------------------
    # 3. Detect metadata
    # ------------------------------------------------------------------
    logger.info("Auto-detecting SDV metadata …")
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(data=real_data)
    logger.info("Metadata detected for %d columns.", len(real_data.columns))

    # ------------------------------------------------------------------
    # 4. Initialize a FRESH CTGAN model (brand-new weights)
    # ------------------------------------------------------------------
    logger.info("Initializing fresh CTGAN (epochs=%d, batch=%d, emb=%d, cuda=%s)",
                args.epochs, args.batch_size, args.embedding_dim, args.cuda)

    synthesizer = CTGANSynthesizer(
        metadata=metadata,
        epochs=args.epochs,
        batch_size=args.batch_size,
        generator_dim=tuple(args.generator_dim),
        discriminator_dim=tuple(args.discriminator_dim),
        embedding_dim=args.embedding_dim,
        generator_lr=args.generator_lr,
        discriminator_lr=args.discriminator_lr,
        discriminator_steps=args.discriminator_steps,
        pac=args.pac,
        cuda=args.cuda,
    )

    # ------------------------------------------------------------------
    # 5. Train
    # ------------------------------------------------------------------
    logger.info("Training started …")
    t0 = time.time()
    synthesizer.fit(real_data)
    elapsed = time.time() - t0
    logger.info("Training complete in %.1f seconds (%.1f min).", elapsed, elapsed / 60)

    # ------------------------------------------------------------------
    # 6. Generate synthetic data
    # ------------------------------------------------------------------
    logger.info("Generating %d synthetic rows …", num_rows)
    synthetic_data = synthesizer.sample(num_rows=num_rows)
    logger.info("Generated shape: %s", synthetic_data.shape)

    # ------------------------------------------------------------------
    # 7. Save synthetic CSV
    # ------------------------------------------------------------------
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    synthetic_data.to_csv(args.output, index=False)
    logger.info("Synthetic data saved → %s", args.output)

    # ------------------------------------------------------------------
    # 8. Save trained model
    # ------------------------------------------------------------------
    model_path = args.save_model
    if model_path is None:
        model_path = os.path.join("saved_models", f"ctgan_{args.dataset}.pkl")
    os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
    synthesizer.save(model_path)
    logger.info("Model checkpoint saved → %s", model_path)

    # ------------------------------------------------------------------
    # 9. Quick sanity check
    # ------------------------------------------------------------------
    logger.info("")
    logger.info("─── SANITY CHECK ───")
    logger.info("Real  data — shape: %s  |  mean(col0): %.4f",
                real_data.shape,
                real_data.select_dtypes("number").iloc[:, 0].mean())
    logger.info("Synth data — shape: %s  |  mean(col0): %.4f",
                synthetic_data.shape,
                synthetic_data.select_dtypes("number").iloc[:, 0].mean())
    logger.info("=" * 60)
    logger.info("✅ CTGAN [%s] — DONE", args.dataset.upper())
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

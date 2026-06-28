"""
SynthoGen AI — Patient Generator
==================================
Generates synthetic patient records by filtering + sampling from
real data and adding controlled noise. This mirrors the existing
Live Generator approach in app.py.

Strategy:
  1. Load the real dataset.
  2. Apply constraint-based filters (age, gender, conditions).
  3. Sample N patients (with replacement if needed).
  4. Inject small Gaussian noise to create synthetic variation.
  5. Round integer-typed columns back.
  6. Compute summary statistics.
"""

import os

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_DATA_PATH = "data/synthetic/tabddpm_diabetes.csv"
NOISE_SCALE = 0.05  # std multiplier for Gaussian noise


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def generate_patients(constraints, data_path=None):
    """
    Generate synthetic patients matching the given constraints.

    Parameters:
        constraints (dict): Normalized constraints from prompt_parser.
        data_path (str): Optional override for the CSV path.

    Returns:
        (pd.DataFrame, dict)  —  generated patients + summary statistics.

    Raises:
        FileNotFoundError — if the dataset CSV is missing.
        RuntimeError      — if generation fails.
    """
    if data_path is None:
        data_path = DEFAULT_DATA_PATH

    if not os.path.isfile(data_path):
        raise FileNotFoundError(
            "Dataset not found at: {}. "
            "Run data preprocessing first.".format(data_path)
        )

    df = pd.read_csv(data_path)
    original_count = len(df)

    # ------------------------------------------------------------------
    # Apply filters
    # ------------------------------------------------------------------
    mask = pd.Series([True] * len(df), index=df.index)

    # Gender filter
    if constraints.get("gender") is not None and "Sex" in df.columns:
        mask = mask & (df["Sex"] == constraints["gender"])

    # Age filter
    if constraints.get("age_min") is not None and "Age" in df.columns:
        mask = mask & (df["Age"] >= constraints["age_min"])
    if constraints.get("age_max") is not None and "Age" in df.columns:
        mask = mask & (df["Age"] <= constraints["age_max"])

    # Condition filters
    for cond_name, col_name, filter_fn in constraints.get("condition_filters", []):
        if col_name in df.columns:
            mask = mask & df[col_name].apply(filter_fn)

    # Severity filter (diabetes-specific)
    severity_filter = constraints.get("severity_filter")
    if severity_filter is not None:
        col_name, target_val = severity_filter
        if col_name in df.columns:
            mask = mask & (df[col_name] == target_val)

    df_filtered = df[mask]
    matched_count = len(df_filtered)

    if matched_count == 0:
        return pd.DataFrame(), {
            "total_real_rows": original_count,
            "matched_rows": 0,
            "generated_rows": 0,
            "message": "No patients matched the specified criteria. Try broader constraints.",
        }

    # ------------------------------------------------------------------
    # Sample with replacement if needed
    # ------------------------------------------------------------------
    num_patients = constraints.get("num_patients", 100)
    replace = num_patients > matched_count
    synth_df = df_filtered.sample(n=num_patients, replace=replace).reset_index(drop=True)

    # ------------------------------------------------------------------
    # Add synthetic noise to numeric columns
    # ------------------------------------------------------------------
    for col in synth_df.select_dtypes(include=[np.number]).columns:
        std = synth_df[col].std()
        if std > 0:
            noise = np.random.normal(0, std * NOISE_SCALE, size=len(synth_df))
            synth_df[col] = synth_df[col] + noise
            # Preserve integer types
            if str(df[col].dtype).startswith("int"):
                synth_df[col] = synth_df[col].round().astype(int)

    # ------------------------------------------------------------------
    # Compute summary statistics
    # ------------------------------------------------------------------
    stats = _compute_stats(synth_df, original_count, matched_count)

    return synth_df, stats


def _compute_stats(df, total_real, matched_real):
    """Compute a summary dict for display."""
    stats = {
        "total_real_rows": total_real,
        "matched_rows": matched_real,
        "generated_rows": len(df),
        "num_columns": len(df.columns),
    }

    if "Age" in df.columns:
        stats["age_mean"] = round(float(df["Age"].mean()), 1)
        stats["age_min"] = int(df["Age"].min())
        stats["age_max"] = int(df["Age"].max())

    if "Sex" in df.columns:
        male_pct = (df["Sex"] == 1).sum() * 100.0 / max(len(df), 1)
        stats["male_pct"] = round(male_pct, 1)
        stats["female_pct"] = round(100.0 - male_pct, 1)

    if "Diabetes_Target" in df.columns:
        diabetes_pct = (df["Diabetes_Target"] >= 1).sum() * 100.0 / max(len(df), 1)
        stats["diabetes_pct"] = round(diabetes_pct, 1)

    if "Has_Hypertension" in df.columns:
        ht_pct = (df["Has_Hypertension"] == 1).sum() * 100.0 / max(len(df), 1)
        stats["hypertension_pct"] = round(ht_pct, 1)

    if "Has_Hypothyroidism" in df.columns:
        hypo_pct = (df["Has_Hypothyroidism"] == 1).sum() * 100.0 / max(len(df), 1)
        stats["hypothyroidism_pct"] = round(hypo_pct, 1)

    return stats

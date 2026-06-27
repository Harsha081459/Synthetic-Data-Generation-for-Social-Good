"""
SynthoGen AI — Evaluation Suite
=================================
Calculates Utility and Privacy metrics for generated synthetic datasets.

Metrics:
  1. TSTR (Train on Synthetic, Test on Real) - LightGBM Classification
  2. DCR (Distance to Closest Record) - Privacy Risk Analysis
  3. Feature Correlation Similarity - Pearson matrix difference

Usage:
  python eval/evaluate.py --real data/processed/diabetes_mcdd_clean.csv \
                          --synth data/synthetic/tabddpm_diabetes.csv \
                          --target Diabetes_012
"""

import argparse
import json
import logging
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s")
logger = logging.getLogger("evaluator")

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate Synthetic Data")
    p.add_argument("--real", type=str, required=True, help="Path to real CSV")
    p.add_argument("--synth", type=str, required=True, help="Path to synthetic CSV")
    p.add_argument("--target", type=str, default=None, help="Target column for TSTR (classification)")
    p.add_argument("--output", type=str, default="eval/report.json", help="Path to save JSON report")
    return p.parse_args()


def preprocess_for_ml(df_real, df_synth, target_col):
    """Encode categorical columns so LightGBM/KNN can process them."""
    # Ensure columns match
    common_cols = [c for c in df_real.columns if c in df_synth.columns]
    df_r = df_real[common_cols].copy()
    df_s = df_synth[common_cols].copy()

    # Drop NaNs for simple ML evaluation
    df_r = df_r.dropna()
    df_s = df_s.dropna()
    
    encoders = {}
    for col in common_cols:
        if df_r[col].dtype == 'object' or df_r[col].dtype.name == 'category':
            le = LabelEncoder()
            # Fit on both to ensure all classes are known
            le.fit(pd.concat([df_r[col], df_s[col]]).astype(str))
            df_r[col] = le.transform(df_r[col].astype(str))
            df_s[col] = le.transform(df_s[col].astype(str))
            
    return df_r, df_s


def evaluate_tstr(df_real, df_synth, target_col):
    """Train on Synthetic, Test on Real AND Train on Real, Test on Real (Baseline)."""
    logger.info(f"Evaluating TSTR & Baseline on target: {target_col}")
    if target_col not in df_real.columns:
        logger.warning(f"Target {target_col} not found. Skipping TSTR.")
        return None

    # --- 1. Baseline: Real-Train / Real-Test ---
    X_real = df_real.drop(columns=[target_col])
    # Ensure target is discrete for classification
    y_real = df_real[target_col].round().astype(int)
    
    X_r_train, X_r_test, y_r_train, y_r_test = train_test_split(
        X_real, y_real, test_size=0.2, random_state=42, stratify=y_real
    )
    
    clf_base = lgb.LGBMClassifier(n_estimators=100, random_state=42, n_jobs=-1, verbose=-1)
    clf_base.fit(X_r_train, y_r_train)
    base_preds = clf_base.predict(X_r_test)
    
    base_acc = accuracy_score(y_r_test, base_preds)
    base_f1 = f1_score(y_r_test, base_preds, average='weighted')
    logger.info(f"Baseline (Real/Real) -> Accuracy: {base_acc:.4f} | F1: {base_f1:.4f}")

    # --- 2. TSTR: Synth-Train / Real-Test ---
    X_s_train = df_synth.drop(columns=[target_col])
    # Ensure synthetic target is discrete as well
    y_s_train = df_synth[target_col].round().astype(int)
    
    # We test on the exact same 20% holdout set used for the baseline
    clf_tstr = lgb.LGBMClassifier(n_estimators=100, random_state=42, n_jobs=-1, verbose=-1)
    clf_tstr.fit(X_s_train, y_s_train)
    tstr_preds = clf_tstr.predict(X_r_test)
    
    tstr_acc = accuracy_score(y_r_test, tstr_preds)
    tstr_f1 = f1_score(y_r_test, tstr_preds, average='weighted')
    logger.info(f"TSTR (Synth/Real)   -> Accuracy: {tstr_acc:.4f} | F1: {tstr_f1:.4f}")
    
    # --- 3. The Gap ---
    acc_gap = base_acc - tstr_acc
    logger.info(f"Utility Loss Gap    -> {acc_gap:.4f}")

    return {
        "baseline_accuracy": base_acc,
        "baseline_f1": base_f1,
        "tstr_accuracy": tstr_acc,
        "tstr_f1": tstr_f1,
        "utility_loss_gap": acc_gap
    }


def evaluate_privacy_dcr(df_real, df_synth, sample_size=2000):
    """
    Distance to Closest Record (DCR).
    Calculates distance from each synthetic row to the closest real row.
    If distance is 0, the synthetic row is an exact copy (Privacy breach).
    """
    logger.info("Evaluating Privacy (DCR)...")
    
    # Subsample for speed if needed
    if len(df_real) > sample_size:
        df_real = df_real.sample(sample_size, random_state=42)
    if len(df_synth) > sample_size:
        df_synth = df_synth.sample(sample_size, random_state=42)
        
    # Normalize data for KNN (so features contribute equally)
    # Using real data stats to normalize both
    mean = df_real.mean()
    std = df_real.std().replace(0, 1) # Prevent div by zero
    
    r_norm = (df_real - mean) / std
    s_norm = (df_synth - mean) / std
    
    # Fit KNN on Real data
    nn = NearestNeighbors(n_neighbors=1, algorithm='kd_tree', n_jobs=-1)
    nn.fit(r_norm)
    
    # Find distance from Synth to nearest Real
    distances, _ = nn.kneighbors(s_norm)
    
    # If distance is < 1e-6, it's basically an exact match
    exact_matches = (distances < 1e-6).sum()
    pct_exact_matches = exact_matches / len(df_synth) * 100
    
    avg_dcr = float(np.mean(distances))
    
    logger.info(f"Average DCR: {avg_dcr:.4f}")
    logger.info(f"Exact Matches (Privacy Breaches): {exact_matches} ({pct_exact_matches:.2f}%)")
    
    return {
        "avg_dcr": avg_dcr,
        "exact_match_count": int(exact_matches),
        "exact_match_percent": float(pct_exact_matches)
    }

def evaluate_correlation_similarity(df_real, df_synth):
    """Compare the feature correlation matrices."""
    logger.info("Evaluating Correlation Similarity...")
    corr_real = df_real.corr().fillna(0).values
    corr_synth = df_synth.corr().fillna(0).values
    
    # Mean Absolute Error between correlation matrices
    mae = np.mean(np.abs(corr_real - corr_synth))
    logger.info(f"Correlation Matrix MAE: {mae:.4f}")
    
    return {"corr_matrix_mae": float(mae)}


def main():
    args = parse_args()
    
    if not os.path.exists(args.real) or not os.path.exists(args.synth):
        logger.error("Missing input CSV files.")
        sys.exit(1)
        
    df_real = pd.read_csv(args.real)
    df_synth = pd.read_csv(args.synth)
    
    logger.info("Preprocessing data for evaluation...")
    df_r, df_s = preprocess_for_ml(df_real, df_synth, args.target)
    
    report = {
        "dataset_real": args.real,
        "dataset_synth": args.synth,
        "target": args.target
    }
    
    # 1. Utility (TSTR)
    if args.target:
        report["utility"] = evaluate_tstr(df_r, df_s, args.target)
        
    # 2. Privacy (DCR)
    report["privacy"] = evaluate_privacy_dcr(df_r, df_s)
    
    # 3. Fidelity (Correlation)
    report["fidelity"] = evaluate_correlation_similarity(df_r, df_s)
    
    # Save Report
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(report, f, indent=4)
        
    logger.info(f"✅ Evaluation complete. Report saved to {args.output}")

if __name__ == "__main__":
    main()

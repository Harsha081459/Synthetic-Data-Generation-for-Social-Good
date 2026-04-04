"""
SynthoGen AI — Comprehensive Evaluation Suite
===============================================
Calculates Utility, Privacy, and Fidelity metrics for synthetic datasets.

Metrics:
  Utility:
    1. TSTR (Train on Synthetic, Test on Real) with Real/Real Baseline & Gap
  Privacy:
    2. DCR (Distance to Closest Record)
    3. K-Anonymity Score
    4. Re-identification Risk Score
  Fidelity:
    5. Feature Correlation Similarity (Pearson MAE)
    6. Column Shape Similarity (KS-Test per column)
    7. Pair-wise Correlation Trend Accuracy
    8. Class Distribution Comparison (Jensen-Shannon Divergence)

Usage:
  python eval/evaluate.py --real data/processed/diabetes_mcdd_clean.csv \\
                          --synth data/synthetic/tabddpm_diabetes.csv \\
                          --target Diabetes_Target
"""

import argparse
import json
import logging
import os
import sys
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import ks_2samp
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s")
logger = logging.getLogger("evaluator")


# ============================================================================
# CLI
# ============================================================================
def parse_args():
    p = argparse.ArgumentParser(description="Evaluate Synthetic Data")
    p.add_argument("--real", type=str, required=True, help="Path to real CSV")
    p.add_argument("--synth", type=str, required=True, help="Path to synthetic CSV")
    p.add_argument("--target", type=str, default=None, help="Target column for TSTR")
    p.add_argument("--quasi_ids", type=str, default=None,
                   help="Comma-separated quasi-identifier columns for K-Anonymity (e.g. Age,Sex,BMI)")
    p.add_argument("--output", type=str, default="eval/report.json", help="Output JSON path")
    return p.parse_args()


# ============================================================================
# Preprocessing
# ============================================================================
def preprocess_for_ml(df_real, df_synth, target_col):
    """Align columns, drop NaNs, encode categoricals."""
    common_cols = [c for c in df_real.columns if c in df_synth.columns]
    df_r = df_real[common_cols].copy()
    df_s = df_synth[common_cols].copy()

    df_r = df_r.dropna()
    df_s = df_s.dropna()

    for col in common_cols:
        if df_r[col].dtype == "object" or df_r[col].dtype.name == "category":
            le = LabelEncoder()
            le.fit(pd.concat([df_r[col], df_s[col]]).astype(str))
            df_r[col] = le.transform(df_r[col].astype(str))
            df_s[col] = le.transform(df_s[col].astype(str))

    return df_r, df_s


# ============================================================================
# 1. UTILITY — TSTR with Baseline & Gap
# ============================================================================
def evaluate_tstr(df_real, df_synth, target_col):
    """Train on Synthetic, Test on Real AND Train on Real, Test on Real (Baseline)."""
    logger.info(f"Evaluating TSTR & Baseline on target: {target_col}")
    if target_col not in df_real.columns:
        logger.warning(f"Target {target_col} not found. Skipping TSTR.")
        return None

    # --- 1. Baseline: Real-Train / Real-Test ---
    X_real = df_real.drop(columns=[target_col])
    y_real = df_real[target_col].round().astype(int)

    X_r_train, X_r_test, y_r_train, y_r_test = train_test_split(
        X_real, y_real, test_size=0.2, random_state=42, stratify=y_real
    )

    clf_base = lgb.LGBMClassifier(n_estimators=100, random_state=42, n_jobs=-1, verbose=-1)
    clf_base.fit(X_r_train, y_r_train)
    base_preds = clf_base.predict(X_r_test)

    base_acc = accuracy_score(y_r_test, base_preds)
    base_f1 = f1_score(y_r_test, base_preds, average="weighted")
    logger.info(f"Baseline (Real/Real) -> Accuracy: {base_acc:.4f} | F1: {base_f1:.4f}")

    # --- 2. TSTR: Synth-Train / Real-Test ---
    X_s_train = df_synth.drop(columns=[target_col])
    y_s_train = df_synth[target_col].round().astype(int)

    clf_tstr = lgb.LGBMClassifier(n_estimators=100, random_state=42, n_jobs=-1, verbose=-1)
    clf_tstr.fit(X_s_train, y_s_train)
    tstr_preds = clf_tstr.predict(X_r_test)

    tstr_acc = accuracy_score(y_r_test, tstr_preds)
    tstr_f1 = f1_score(y_r_test, tstr_preds, average="weighted")
    logger.info(f"TSTR (Synth/Real)   -> Accuracy: {tstr_acc:.4f} | F1: {tstr_f1:.4f}")

    acc_gap = base_acc - tstr_acc
    logger.info(f"Utility Loss Gap    -> {acc_gap:.4f}")

    return {
        "baseline_accuracy": round(base_acc, 4),
        "baseline_f1": round(base_f1, 4),
        "tstr_accuracy": round(tstr_acc, 4),
        "tstr_f1": round(tstr_f1, 4),
        "utility_loss_gap": round(acc_gap, 4),
    }


# ============================================================================
# 2. PRIVACY — Distance to Closest Record (DCR)
# ============================================================================
def evaluate_privacy_dcr(df_real, df_synth, sample_size=2000):
    """Euclidean distance from each synthetic row to nearest real row."""
    logger.info("Evaluating Privacy (DCR)...")

    r = df_real.sample(min(len(df_real), sample_size), random_state=42)
    s = df_synth.sample(min(len(df_synth), sample_size), random_state=42)

    mean = r.mean()
    std = r.std().replace(0, 1)
    r_norm = (r - mean) / std
    s_norm = (s - mean) / std

    nn = NearestNeighbors(n_neighbors=1, algorithm="kd_tree", n_jobs=-1)
    nn.fit(r_norm)
    distances, _ = nn.kneighbors(s_norm)

    exact_matches = int((distances < 1e-6).sum())
    pct_exact = exact_matches / len(s) * 100
    avg_dcr = float(np.mean(distances))
    median_dcr = float(np.median(distances))
    min_dcr = float(np.min(distances))

    logger.info(f"Average DCR: {avg_dcr:.4f} | Median: {median_dcr:.4f} | Min: {min_dcr:.4f}")
    logger.info(f"Exact Matches (Privacy Breaches): {exact_matches} ({pct_exact:.2f}%)")

    return {
        "avg_dcr": round(avg_dcr, 4),
        "median_dcr": round(median_dcr, 4),
        "min_dcr": round(min_dcr, 4),
        "exact_match_count": exact_matches,
        "exact_match_percent": round(pct_exact, 2),
    }


# ============================================================================
# 3. PRIVACY — K-Anonymity
# ============================================================================
def evaluate_k_anonymity(df_synth, quasi_ids):
    """
    K-Anonymity: for a set of quasi-identifiers, find the smallest group
    of records sharing the same quasi-identifier values.
    A k=1 means a record is uniquely identifiable.
    """
    logger.info(f"Evaluating K-Anonymity on quasi-identifiers: {quasi_ids}")

    # Bin continuous quasi-identifiers into buckets for meaningful grouping
    df_binned = df_synth.copy()
    for col in quasi_ids:
        if df_binned[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
            nunique = df_binned[col].nunique()
            if nunique > 20:
                df_binned[col] = pd.qcut(df_binned[col], q=10, duplicates="drop").astype(str)

    group_sizes = df_binned.groupby(quasi_ids).size()
    k_min = int(group_sizes.min())
    k_median = int(group_sizes.median())
    k_mean = round(float(group_sizes.mean()), 2)
    pct_vulnerable = round((group_sizes[group_sizes <= 5].sum() / len(df_synth)) * 100, 2)

    # Create bins for visualization: 1-5, 6-10, 11-20, 21-50, 51+
    bins = [0, 5, 10, 20, 50, float('inf')]
    labels = ["1-5", "6-10", "11-20", "21-50", "51+"]
    binned_counts = pd.cut(group_sizes, bins=bins, labels=labels).value_counts().sort_index()
    distribution = {str(k): int(v) for k, v in binned_counts.items()}

    logger.info(f"K-Anonymity -> k_min: {k_min} | k_median: {k_median} | k_mean: {k_mean}")
    logger.info(f"Records with k<=5 (vulnerable): {pct_vulnerable}%")

    return {
        "k_min": k_min,
        "k_median": k_median,
        "k_mean": k_mean,
        "pct_records_k_leq_5": pct_vulnerable,
        "distribution": distribution
    }


# ============================================================================
# 4. PRIVACY — Re-identification Risk Score
# ============================================================================
def evaluate_reidentification_risk(df_real, df_synth, sample_size=2000):
    """
    For each synthetic record, calculate a risk score (0-1) based on
    how close it is to the nearest real record relative to the dataset spread.
    """
    logger.info("Evaluating Re-identification Risk Score...")

    r = df_real.sample(min(len(df_real), sample_size), random_state=42)
    s = df_synth.sample(min(len(df_synth), sample_size), random_state=42)

    mean = r.mean()
    std = r.std().replace(0, 1)
    r_norm = (r - mean) / std
    s_norm = (s - mean) / std

    nn = NearestNeighbors(n_neighbors=1, algorithm="kd_tree", n_jobs=-1)
    nn.fit(r_norm)
    distances, _ = nn.kneighbors(s_norm)

    # Convert distance to risk: risk = exp(-distance). Closer = higher risk.
    risk_scores = np.exp(-distances.flatten())

    avg_risk = round(float(np.mean(risk_scores)), 4)
    max_risk = round(float(np.max(risk_scores)), 4)
    high_risk_count = int((risk_scores > 0.5).sum())
    high_risk_pct = round(high_risk_count / len(risk_scores) * 100, 2)

    logger.info(f"Avg Risk: {avg_risk} | Max Risk: {max_risk}")
    logger.info(f"High-Risk Records (>0.5): {high_risk_count} ({high_risk_pct}%)")

    return {
        "avg_risk_score": avg_risk,
        "max_risk_score": max_risk,
        "high_risk_count": high_risk_count,
        "high_risk_percent": high_risk_pct,
    }


# ============================================================================
# 5. FIDELITY — Correlation Matrix MAE
# ============================================================================
def evaluate_correlation_similarity(df_real, df_synth):
    """Mean Absolute Error between Pearson correlation matrices."""
    logger.info("Evaluating Correlation Similarity...")
    num_r = df_real.select_dtypes(include=np.number)
    num_s = df_synth.select_dtypes(include=np.number)
    common = [c for c in num_r.columns if c in num_s.columns]

    corr_real = num_r[common].corr().fillna(0).values
    corr_synth = num_s[common].corr().fillna(0).values

    mae = float(np.mean(np.abs(corr_real - corr_synth)))
    logger.info(f"Correlation Matrix MAE: {mae:.4f}")

    return {"corr_matrix_mae": round(mae, 4)}


# ============================================================================
# 6. FIDELITY — Column Shape Similarity (KS-Test)
# ============================================================================
def evaluate_column_shapes(df_real, df_synth):
    """
    Run a 2-sample Kolmogorov-Smirnov test on each numerical column.
    KS statistic close to 0 = distributions are identical.
    """
    logger.info("Evaluating Column Shape Similarity (KS-Test)...")
    num_cols = [c for c in df_real.select_dtypes(include=np.number).columns
                if c in df_synth.columns]

    ks_results = {}
    for col in num_cols:
        stat, pval = ks_2samp(df_real[col].dropna(), df_synth[col].dropna())
        ks_results[col] = {"ks_statistic": round(stat, 4), "p_value": round(pval, 4)}

    # Average KS statistic across all columns
    avg_ks = round(np.mean([v["ks_statistic"] for v in ks_results.values()]), 4)
    # % of columns that pass (p > 0.05 means distributions are statistically similar)
    passed = sum(1 for v in ks_results.values() if v["p_value"] > 0.05)
    pct_passed = round(passed / max(len(ks_results), 1) * 100, 1)

    logger.info(f"Avg KS Statistic: {avg_ks} | Columns Passed: {passed}/{len(ks_results)} ({pct_passed}%)")

    return {
        "avg_ks_statistic": avg_ks,
        "columns_passed_pct": pct_passed,
        "per_column": ks_results,
    }


# ============================================================================
# 7. FIDELITY — Pair-wise Correlation Trend Accuracy
# ============================================================================
def evaluate_pairwise_trends(df_real, df_synth):
    """
    For every pair of numeric features, check if the sign of the correlation
    (positive/negative/near-zero) is preserved between real and synthetic.
    """
    logger.info("Evaluating Pair-wise Correlation Trends...")
    num_cols = [c for c in df_real.select_dtypes(include=np.number).columns
                if c in df_synth.columns]

    if len(num_cols) < 2:
        return {"trend_accuracy": None, "note": "Not enough numeric columns"}

    corr_r = df_real[num_cols].corr()
    corr_s = df_synth[num_cols].corr()

    matches = 0
    total = 0
    for c1, c2 in combinations(num_cols, 2):
        sign_r = np.sign(corr_r.loc[c1, c2])
        sign_s = np.sign(corr_s.loc[c1, c2])
        if sign_r == sign_s:
            matches += 1
        total += 1

    trend_acc = round(matches / max(total, 1) * 100, 1)
    logger.info(f"Pair-wise Trend Accuracy: {trend_acc}% ({matches}/{total} pairs preserved)")

    return {"trend_accuracy_pct": trend_acc, "pairs_matched": matches, "pairs_total": total}


# ============================================================================
# 8. FIDELITY — Class Distribution Comparison
# ============================================================================
def evaluate_class_distribution(df_real, df_synth, target_col):
    """
    Compare target class distributions using Jensen-Shannon Divergence.
    JSD = 0 means identical distributions, JSD = 1 means completely different.
    """
    logger.info(f"Evaluating Class Distribution for: {target_col}")
    if target_col not in df_real.columns or target_col not in df_synth.columns:
        return None

    real_dist = df_real[target_col].round().astype(int).value_counts(normalize=True).sort_index()
    synth_dist = df_synth[target_col].round().astype(int).value_counts(normalize=True).sort_index()

    # Align indices
    all_classes = sorted(set(real_dist.index) | set(synth_dist.index))
    real_probs = [real_dist.get(c, 0) for c in all_classes]
    synth_probs = [synth_dist.get(c, 0) for c in all_classes]

    jsd = round(float(jensenshannon(real_probs, synth_probs)), 4)

    class_report = {}
    for c in all_classes:
        class_report[str(c)] = {
            "real_pct": round(real_dist.get(c, 0) * 100, 2),
            "synth_pct": round(synth_dist.get(c, 0) * 100, 2),
        }

    logger.info(f"Jensen-Shannon Divergence: {jsd}")
    for c, v in class_report.items():
        logger.info(f"  Class {c}: Real={v['real_pct']}% | Synth={v['synth_pct']}%")

    return {"jensen_shannon_divergence": jsd, "per_class": class_report}


# ============================================================================
# MAIN
# ============================================================================
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
        "target": args.target,
    }

    # 1. Utility (TSTR)
    if args.target:
        report["utility"] = evaluate_tstr(df_r, df_s, args.target)

    # 2. Privacy (DCR)
    report["privacy_dcr"] = evaluate_privacy_dcr(df_r, df_s)

    # 3. Privacy (K-Anonymity)
    if args.quasi_ids:
        qi = [q.strip() for q in args.quasi_ids.split(",")]
        qi_valid = [q for q in qi if q in df_s.columns]
        if qi_valid:
            report["privacy_k_anonymity"] = evaluate_k_anonymity(df_s, qi_valid)
    else:
        # Auto-detect quasi-identifiers: use columns like Age, Sex, BMI if present
        auto_qi = [c for c in ["Age", "Sex", "BMI"] if c in df_s.columns]
        if auto_qi:
            report["privacy_k_anonymity"] = evaluate_k_anonymity(df_s, auto_qi)

    # 4. Privacy (Re-identification Risk)
    report["privacy_reidentification"] = evaluate_reidentification_risk(df_r, df_s)

    # 5. Fidelity (Correlation)
    report["fidelity_correlation"] = evaluate_correlation_similarity(df_r, df_s)

    # 6. Fidelity (Column Shapes - KS Test)
    report["fidelity_column_shapes"] = evaluate_column_shapes(df_r, df_s)

    # 7. Fidelity (Pair-wise Trends)
    report["fidelity_pairwise_trends"] = evaluate_pairwise_trends(df_r, df_s)

    # 8. Fidelity (Class Distribution)
    if args.target:
        report["fidelity_class_distribution"] = evaluate_class_distribution(df_r, df_s, args.target)

    # Save Report
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(report, f, indent=4)

    logger.info(f"✅ Evaluation complete. Report saved to {args.output}")


if __name__ == "__main__":
    main()

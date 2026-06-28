"""
SynthoGen AI — Balanced Synthetic Data Generator
==================================================
Addresses class imbalance by generating class-balanced synthetic datasets.

Given an imbalanced real dataset (e.g., 85% healthy, 15% diabetic), this script:
  1. Detects the class distribution.
  2. Uses the already-generated synthetic data.
  3. Over-samples minority classes and under-samples majority classes.
  4. Outputs a perfectly balanced synthetic dataset.
  5. Proves that a classifier trained on balanced synthetic data is
     MORE FAIR to minority classes than one trained on imbalanced real data.

Usage:
  python models/balanced_generator.py --synth data/synthetic/tabddpm_diabetes.csv \\
                                      --real data/processed/diabetes_mcdd_clean.csv \\
                                      --target Diabetes_Target \\
                                      --output data/synthetic/tabddpm_diabetes_balanced.csv
"""

import argparse
import logging
import os

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
import lightgbm as lgb

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s")
logger = logging.getLogger("balanced_gen")


def parse_args():
    p = argparse.ArgumentParser(description="Generate Class-Balanced Synthetic Data")
    p.add_argument("--synth", type=str, required=True, help="Path to unbalanced synthetic CSV")
    p.add_argument("--real", type=str, required=True, help="Path to real CSV (for fairness comparison)")
    p.add_argument("--target", type=str, required=True, help="Target column name")
    p.add_argument("--output", type=str, required=True, help="Output path for balanced CSV")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def analyze_distribution(df, target_col, label="Dataset"):
    """Print and return class distribution."""
    dist = df[target_col].round().astype(int).value_counts().sort_index()
    total = len(df)
    logger.info(f"\n{label} Class Distribution:")
    for cls, count in dist.items():
        logger.info(f"  Class {cls}: {count} ({count/total*100:.1f}%)")
    return dist


def balance_synthetic_data(df_synth, target_col, seed=42):
    """
    Create a perfectly balanced synthetic dataset by over-sampling minority
    classes and under-sampling majority classes to the median class size.
    """
    logger.info("Balancing synthetic data...")
    target = df_synth[target_col].round().astype(int)
    classes = target.unique()
    class_counts = target.value_counts()

    # Target size: use the median class count so we don't lose too much data
    target_size = int(class_counts.median())
    logger.info(f"Target samples per class: {target_size}")

    balanced_frames = []
    for cls in sorted(classes):
        cls_df = df_synth[target == cls]
        if len(cls_df) >= target_size:
            # Under-sample
            balanced_frames.append(cls_df.sample(target_size, random_state=seed))
        else:
            # Over-sample with replacement
            balanced_frames.append(cls_df.sample(target_size, replace=True, random_state=seed))

    balanced = pd.concat(balanced_frames, ignore_index=True)
    # Shuffle the final dataset
    balanced = balanced.sample(frac=1, random_state=seed).reset_index(drop=True)

    return balanced


def fairness_comparison(df_real, df_synth_balanced, target_col):
    """
    Prove that a classifier trained on balanced synthetic data is fairer
    to minority classes than one trained on imbalanced real data.
    """
    logger.info("\n--- Fairness Comparison ---")

    target_real = df_real[target_col].round().astype(int)
    X_real = df_real.drop(columns=[target_col])

    # Split real data 80/20 for testing
    X_r_train, X_r_test, y_r_train, y_r_test = train_test_split(
        X_real, target_real, test_size=0.2, random_state=42, stratify=target_real
    )

    # --- Model 1: Trained on Imbalanced Real Data ---
    clf_real = lgb.LGBMClassifier(n_estimators=100, random_state=42, n_jobs=-1, verbose=-1)
    clf_real.fit(X_r_train, y_r_train)
    preds_real = clf_real.predict(X_r_test)

    logger.info("Model trained on IMBALANCED REAL data:")
    logger.info(f"  Overall Accuracy: {accuracy_score(y_r_test, preds_real):.4f}")
    logger.info(f"  Overall F1:       {f1_score(y_r_test, preds_real, average='weighted'):.4f}")

    # Per-class F1
    report_real = classification_report(y_r_test, preds_real, output_dict=True, zero_division=0)

    # --- Model 2: Trained on Balanced Synthetic Data ---
    target_synth = df_synth_balanced[target_col].round().astype(int)
    X_s_train = df_synth_balanced.drop(columns=[target_col])

    clf_synth = lgb.LGBMClassifier(n_estimators=100, random_state=42, n_jobs=-1, verbose=-1)
    clf_synth.fit(X_s_train, target_synth)
    preds_synth = clf_synth.predict(X_r_test)

    logger.info("\nModel trained on BALANCED SYNTHETIC data:")
    logger.info(f"  Overall Accuracy: {accuracy_score(y_r_test, preds_synth):.4f}")
    logger.info(f"  Overall F1:       {f1_score(y_r_test, preds_synth, average='weighted'):.4f}")

    report_synth = classification_report(y_r_test, preds_synth, output_dict=True, zero_division=0)

    # --- Compare minority class performance ---
    minority_class = str(target_real.value_counts().idxmin())
    f1_real_minority = report_real.get(minority_class, {}).get("f1-score", 0)
    f1_synth_minority = report_synth.get(minority_class, {}).get("f1-score", 0)
    fairness_gain = f1_synth_minority - f1_real_minority

    logger.info(f"\n--- MINORITY CLASS ({minority_class}) FAIRNESS ---")
    logger.info(f"  F1 (Real-trained):     {f1_real_minority:.4f}")
    logger.info(f"  F1 (Balanced-trained): {f1_synth_minority:.4f}")
    logger.info(f"  Fairness Gain:         {fairness_gain:+.4f}")

    if fairness_gain > 0:
        logger.info("  ✅ Balanced synthetic data IMPROVED fairness for minority class!")
    else:
        logger.info("  ⚠️  Balanced synthetic data did not improve minority F1 (may need tuning).")

    return {
        "real_trained_overall_acc": round(accuracy_score(y_r_test, preds_real), 4),
        "real_trained_overall_f1": round(f1_score(y_r_test, preds_real, average="weighted"), 4),
        "real_trained_minority_f1": round(f1_real_minority, 4),
        "synth_balanced_overall_acc": round(accuracy_score(y_r_test, preds_synth), 4),
        "synth_balanced_overall_f1": round(f1_score(y_r_test, preds_synth, average="weighted"), 4),
        "synth_balanced_minority_f1": round(f1_synth_minority, 4),
        "minority_class": minority_class,
        "fairness_gain": round(fairness_gain, 4),
    }


def main():
    args = parse_args()
    np.random.seed(args.seed)

    df_synth = pd.read_csv(args.synth)
    df_real = pd.read_csv(args.real)

    # Align columns
    common = [c for c in df_real.columns if c in df_synth.columns]
    df_real = df_real[common].dropna()
    df_synth = df_synth[common].dropna()

    # Encode categoricals
    from sklearn.preprocessing import LabelEncoder
    for col in common:
        if df_real[col].dtype == "object" or df_real[col].dtype.name == "category":
            le = LabelEncoder()
            le.fit(pd.concat([df_real[col], df_synth[col]]).astype(str))
            df_real[col] = le.transform(df_real[col].astype(str))
            df_synth[col] = le.transform(df_synth[col].astype(str))

    # 1. Show original distributions
    analyze_distribution(df_real, args.target, "Real (Original)")
    analyze_distribution(df_synth, args.target, "Synthetic (Original)")

    # 2. Balance the synthetic data
    balanced = balance_synthetic_data(df_synth, args.target, args.seed)
    analyze_distribution(balanced, args.target, "Synthetic (Balanced)")

    # 3. Save balanced dataset
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    balanced.to_csv(args.output, index=False)
    logger.info(f"\n✅ Balanced synthetic data saved to {args.output}")

    # 4. Run fairness comparison
    fairness = fairness_comparison(df_real, balanced, args.target)

    # Save fairness report
    import json
    report_path = args.output.replace(".csv", "_fairness.json")
    with open(report_path, "w") as f:
        json.dump(fairness, f, indent=4)
    logger.info(f"✅ Fairness report saved to {report_path}")


if __name__ == "__main__":
    main()

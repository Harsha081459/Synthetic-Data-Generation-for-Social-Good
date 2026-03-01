import pandas as pd
import numpy as np
import argparse
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
from scipy.spatial.distance import cdist
import lightgbm as lgb

def evaluate_ml_utility(real_df, synthetic_df, target_col, task_type="multiclass"):
    print(f"\n--- ML Utility (TSTR) | Target: {target_col} ---")
    common_cols = [c for c in real_df.columns if c in synthetic_df.columns]
    real_df = real_df[common_cols].copy()
    synthetic_df = synthetic_df[common_cols].copy()

    # Encode categoricals
    for col in real_df.select_dtypes(include=["object"]).columns:
        le = LabelEncoder()
        combined = pd.concat([real_df[col], synthetic_df[col]]).astype(str)
        le.fit(combined)
        real_df[col] = le.transform(real_df[col].astype(str))
        synthetic_df[col] = le.transform(synthetic_df[col].astype(str))

    X_real = real_df.drop(columns=[target_col])
    y_real = real_df[target_col]
    X_syn  = synthetic_df.drop(columns=[target_col])
    y_syn  = synthetic_df[target_col]

    _, X_test, _, y_test = train_test_split(X_real, y_real, test_size=0.2, random_state=42, stratify=y_real)
    X_train_real, _, y_train_real, _ = train_test_split(X_real, y_real, test_size=0.2, random_state=42, stratify=y_real)

    avg = "macro" if task_type == "multiclass" else "binary"

    # TRTR: Train on Real, Test on Real (baseline ceiling)
    clf_real = lgb.LGBMClassifier(random_state=42, n_jobs=-1, verbose=-1)
    clf_real.fit(X_train_real, y_train_real)
    f1_trtr = f1_score(y_test, clf_real.predict(X_test), average=avg)

    # TSTR: Train on Synthetic, Test on Real (our metric)
    clf_syn = lgb.LGBMClassifier(random_state=42, n_jobs=-1, verbose=-1)
    clf_syn.fit(X_syn, y_syn)
    f1_tstr = f1_score(y_test, clf_syn.predict(X_test), average=avg)

    print(f"  TRTR F1 (real train baseline): {f1_trtr:.4f}")
    print(f"  TSTR F1 (synthetic train):     {f1_tstr:.4f}")
    print(f"  Utility gap:                   {f1_trtr - f1_tstr:.4f}")
    return {"TRTR_F1": f1_trtr, "TSTR_F1": f1_tstr, "Utility_Gap": f1_trtr - f1_tstr}


def evaluate_privacy_dcr(real_df, synthetic_df, sample_size=500):
    print(f"\n--- Privacy Evaluation (DCR) ---")
    n = min(sample_size, len(real_df), len(synthetic_df))
    real_s = real_df.sample(n=n, random_state=42)
    syn_s  = synthetic_df.sample(n=n, random_state=42)

    num_cols = real_s.select_dtypes(include=[np.number]).columns
    real_n = real_s[num_cols].fillna(0)
    syn_n  = syn_s[num_cols].fillna(0)

    mean = real_n.mean(); std = real_n.std() + 1e-8
    real_norm = (real_n - mean) / std
    syn_norm  = (syn_n  - mean) / std

    dists = cdist(syn_norm.values, real_norm.values, metric="euclidean")
    min_dists = np.min(dists, axis=1)

    mean_dcr = float(np.mean(min_dists))
    p5_dcr   = float(np.percentile(min_dists, 5))
    print(f"  Mean DCR:              {mean_dcr:.4f}")
    print(f"  5th percentile DCR:    {p5_dcr:.4f}  (low = privacy risk)")
    return {"Mean_DCR": mean_dcr, "P5_DCR": p5_dcr}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate synthetic data: ML utility (TSTR) + Privacy (DCR)")
    parser.add_argument("--real",      required=True, help="Path to real data CSV")
    parser.add_argument("--synthetic", required=True, help="Path to synthetic data CSV")
    parser.add_argument("--target",    required=True, help="Target column name")
    parser.add_argument("--task",      default="multiclass", choices=["binary", "multiclass"])
    parser.add_argument("--outdir",    default="data/processed", help="Where to save results")
    args = parser.parse_args()

    real_df = pd.read_csv(args.real)
    syn_df  = pd.read_csv(args.synthetic)

    ml  = evaluate_ml_utility(real_df, syn_df, args.target, args.task)
    prv = evaluate_privacy_dcr(real_df, syn_df)

    results = {**ml, **prv}
    print("\n=== FINAL RESULTS ===")
    for k, v in results.items():
        print(f"  {k}: {v:.4f}")

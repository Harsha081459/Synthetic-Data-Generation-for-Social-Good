"""
Comprehensive evaluator — generates _full.json reports with all metrics
the dashboard expects: utility, privacy_dcr, privacy_reidentification,
privacy_k_anonymity, fidelity_correlation, fidelity_column_shapes,
fidelity_pairwise_trends, fidelity_class_distribution.
"""
import json, os, sys
import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import ks_2samp
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import warnings
warnings.filterwarnings("ignore")

def evaluate_full(real_path, synth_path, target_col, output_path):
    print(f"  Evaluating: {synth_path}")
    df_real = pd.read_csv(real_path)
    df_synth = pd.read_csv(synth_path)

    # Align columns
    common = [c for c in df_real.columns if c in df_synth.columns]
    df_r = df_real[common].copy()
    df_s = df_synth[common].copy()

    # Encode categoricals
    for col in common:
        if df_r[col].dtype == 'object' or df_r[col].dtype.name == 'category':
            le = LabelEncoder()
            le.fit(pd.concat([df_r[col], df_s[col]]).astype(str))
            df_r[col] = le.transform(df_r[col].astype(str))
            df_s[col] = le.transform(df_s[col].astype(str))

    df_r = df_r.dropna()
    df_s = df_s.dropna()

    report = {"dataset_real": real_path, "dataset_synth": synth_path, "target": target_col}

    # ============================================================
    # 1. UTILITY (TSTR)
    # ============================================================
    if target_col and target_col in df_r.columns and target_col in df_s.columns:
        X_real = df_r.drop(columns=[target_col])
        y_real = df_r[target_col].round().astype(int)

        X_train, X_test, y_train, y_test = train_test_split(
            X_real, y_real, test_size=0.2, random_state=42, stratify=y_real
        )
        # Baseline
        clf_base = lgb.LGBMClassifier(n_estimators=100, random_state=42, n_jobs=-1, verbose=-1)
        clf_base.fit(X_train, y_train)
        base_preds = clf_base.predict(X_test)
        base_acc = accuracy_score(y_test, base_preds)
        base_f1 = f1_score(y_test, base_preds, average='weighted')

        # TSTR
        X_s = df_s.drop(columns=[target_col])
        y_s = df_s[target_col].round().astype(int)
        clf_tstr = lgb.LGBMClassifier(n_estimators=100, random_state=42, n_jobs=-1, verbose=-1)
        clf_tstr.fit(X_s, y_s)
        tstr_preds = clf_tstr.predict(X_test)
        tstr_acc = accuracy_score(y_test, tstr_preds)
        tstr_f1 = f1_score(y_test, tstr_preds, average='weighted')

        report["utility"] = {
            "baseline_accuracy": round(base_acc, 4),
            "baseline_f1": round(base_f1, 4),
            "tstr_accuracy": round(tstr_acc, 4),
            "tstr_f1": round(tstr_f1, 4),
            "utility_loss_gap": round(base_acc - tstr_acc, 4),
        }
        print(f"    Utility: Baseline={base_acc:.4f} TSTR={tstr_acc:.4f}")

    # ============================================================
    # 2. PRIVACY - DCR
    # ============================================================
    sample_n = min(2000, len(df_r), len(df_s))
    dr = df_r.select_dtypes(include=[np.number]).sample(sample_n, random_state=42) if len(df_r) > sample_n else df_r.select_dtypes(include=[np.number])
    ds = df_s.select_dtypes(include=[np.number]).sample(sample_n, random_state=42) if len(df_s) > sample_n else df_s.select_dtypes(include=[np.number])

    mean = dr.mean()
    std = dr.std().replace(0, 1)
    r_norm = (dr - mean) / std
    s_norm = (ds - mean) / std

    nn = NearestNeighbors(n_neighbors=1, algorithm='auto', n_jobs=-1)
    nn.fit(r_norm)
    distances, _ = nn.kneighbors(s_norm)
    distances = distances.flatten()

    exact = (distances < 1e-6).sum()
    report["privacy_dcr"] = {
        "avg_dcr": round(float(np.mean(distances)), 4),
        "median_dcr": round(float(np.median(distances)), 4),
        "min_dcr": round(float(np.min(distances)), 4),
        "max_dcr": round(float(np.max(distances)), 4),
        "exact_match_count": int(exact),
        "exact_match_percent": round(float(exact / len(ds) * 100), 2),
    }
    print(f"    DCR: avg={np.mean(distances):.4f} exact={exact}")

    # ============================================================
    # 3. PRIVACY - Re-identification Risk
    # ============================================================
    # Risk = 1 / (1 + DCR) for each synthetic record
    risk_scores = 1.0 / (1.0 + distances)
    report["privacy_reidentification"] = {
        "avg_risk_score": round(float(np.mean(risk_scores)), 4),
        "max_risk_score": round(float(np.max(risk_scores)), 4),
        "min_risk_score": round(float(np.min(risk_scores)), 4),
        "pct_high_risk": round(float((risk_scores > 0.5).sum() / len(risk_scores) * 100), 2),
    }

    # ============================================================
    # 4. PRIVACY - K-Anonymity
    # ============================================================
    # Quasi-identifiers: use a few key columns
    qi_candidates = ["Age", "Sex", "age", "gender", "male"]
    qi_cols = [c for c in qi_candidates if c in df_s.columns]
    if qi_cols:
        # Bin numeric QIs
        ks = df_s[qi_cols].copy()
        for c in qi_cols:
            if ks[c].dtype in [np.float64, np.float32]:
                ks[c] = pd.cut(ks[c], bins=10, labels=False)
            ks[c] = ks[c].astype(str)
        group_sizes = ks.groupby(list(qi_cols)).size()
        k_values = group_sizes.values

        bins = {"1-5": 0, "6-10": 0, "11-20": 0, "21-50": 0, "51+": 0}
        for k in k_values:
            if k <= 5: bins["1-5"] += int(k)
            elif k <= 10: bins["6-10"] += int(k)
            elif k <= 20: bins["11-20"] += int(k)
            elif k <= 50: bins["21-50"] += int(k)
            else: bins["51+"] += int(k)

        report["privacy_k_anonymity"] = {
            "k_min": int(k_values.min()),
            "k_max": int(k_values.max()),
            "k_mean": round(float(k_values.mean()), 1),
            "pct_records_k_leq_5": round(float(bins["1-5"] / len(df_s) * 100), 2),
            "distribution": bins,
        }

    # ============================================================
    # 5. FIDELITY - Correlation
    # ============================================================
    num_r = df_r.select_dtypes(include=[np.number])
    num_s = df_s.select_dtypes(include=[np.number])
    common_num = [c for c in num_r.columns if c in num_s.columns]
    if common_num:
        corr_r = num_r[common_num].corr().fillna(0).values
        corr_s = num_s[common_num].corr().fillna(0).values
        mae = np.mean(np.abs(corr_r - corr_s))
        report["fidelity_correlation"] = {"corr_matrix_mae": round(float(mae), 4)}

    # ============================================================
    # 6. FIDELITY - Column Shapes (KS Test)
    # ============================================================
    ks_stats = []
    for col in common_num:
        try:
            stat, _ = ks_2samp(num_r[col].dropna(), num_s[col].dropna())
            ks_stats.append(stat)
        except:
            pass
    if ks_stats:
        report["fidelity_column_shapes"] = {
            "avg_ks_statistic": round(float(np.mean(ks_stats)), 4),
            "max_ks_statistic": round(float(np.max(ks_stats)), 4),
        }

    # ============================================================
    # 7. FIDELITY - Pairwise Trends
    # ============================================================
    if len(common_num) >= 2:
        correct = 0
        total = 0
        for i in range(min(len(common_num), 10)):
            for j in range(i + 1, min(len(common_num), 10)):
                c1, c2 = common_num[i], common_num[j]
                real_corr = num_r[c1].corr(num_r[c2])
                synth_corr = num_s[c1].corr(num_s[c2])
                if (real_corr > 0 and synth_corr > 0) or (real_corr < 0 and synth_corr < 0) or (abs(real_corr) < 0.05 and abs(synth_corr) < 0.05):
                    correct += 1
                total += 1
        if total > 0:
            report["fidelity_pairwise_trends"] = {
                "trend_accuracy_pct": round(correct / total * 100, 1),
                "pairs_checked": total,
            }

    # ============================================================
    # 8. FIDELITY - Class Distribution
    # ============================================================
    if target_col and target_col in df_r.columns and target_col in df_s.columns:
        real_dist = df_r[target_col].round().astype(int).value_counts(normalize=True).sort_index()
        synth_dist = df_s[target_col].round().astype(int).value_counts(normalize=True).sort_index()
        all_classes = sorted(set(real_dist.index) | set(synth_dist.index))
        r_vec = np.array([real_dist.get(c, 0) for c in all_classes])
        s_vec = np.array([synth_dist.get(c, 0) for c in all_classes])
        jsd = float(jensenshannon(r_vec, s_vec))

        per_class = {}
        for c in all_classes:
            per_class[str(c)] = {
                "real_pct": round(float(real_dist.get(c, 0) * 100), 2),
                "synth_pct": round(float(synth_dist.get(c, 0) * 100), 2),
            }
        report["fidelity_class_distribution"] = {
            "jensen_shannon_divergence": round(jsd, 4),
            "per_class": per_class,
        }

    # Save
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"    Saved: {output_path}")
    return report


if __name__ == "__main__":
    os.chdir(os.path.expanduser("~/DATAPORT_HACKATHON"))

    EVALS = [
        ("tvae",    "data/processed/diabetes_mcdd_clean.csv", "data/synthetic/tvae_diabetes.csv",    "Diabetes_Target"),
        ("ctgan",   "data/processed/diabetes_mcdd_clean.csv", "data/synthetic/ctgan_diabetes.csv",   "Diabetes_Target"),
        ("tabddpm", "data/processed/diabetes_mcdd_clean.csv", "data/synthetic/tabddpm_diabetes.csv", "Diabetes_Target"),
        ("tabsyn",  "data/processed/diabetes_mcdd_clean.csv", "data/synthetic/tabsyn_diabetes.csv",  "Diabetes_Target"),
    ]

    for model, real, synth, target in EVALS:
        out = f"eval/report_{model}_diabetes_full.json"
        try:
            evaluate_full(real, synth, target, out)
        except Exception as e:
            print(f"  ERROR for {model}: {e}")

    # Also generate for Framingham and Synthea (single best model: tabsyn)
    EXTRA = [
        ("tabsyn_framingham", "data/processed/framingham_clean.csv", "data/synthetic/tabsyn_framingham.csv", "TenYearCHD"),
        ("tabsyn_synthea",    "data/processed/synthea_flattened.csv", "data/synthetic/tabsyn_synthea.csv",   "Cond_Essential_hypertension"),
    ]
    for name, real, synth, target in EXTRA:
        out = f"eval/report_{name}_full.json"
        try:
            evaluate_full(real, synth, target, out)
        except Exception as e:
            print(f"  ERROR for {name}: {e}")

    print("\nDONE — All comprehensive reports generated.")

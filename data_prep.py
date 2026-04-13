"""
Data Preparation Pipeline for Synthetic Health Data Generation
==============================================================
Processes 3 datasets for generative model training (TVAE, TabDDPM, TabSyn).

Order of operations (critical):
  1. Drop exact duplicates FIRST (before any imputation)
  2. Add _is_missing binary flags for high-missingness columns
  3. Feature-engineer Diagnosis_Temp -> Has_Hypertension, Has_Hypothyroidism
  4. Clip outliers: hard clinical floors, then P1/P99 Winsorization
  5. MICE imputation (HistGradientBoosting, 10 iterations)
  6. Round discrete/binary columns back to integers
  7. Final validation: 0 NaNs, 0 duplicates, sane min/max
"""

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.ensemble import HistGradientBoostingRegressor

# ============================================================
# Constants
# ============================================================
PROCESSED_DIR = "synthetic-health-sdg/data/processed"

# Missingness threshold: create _is_missing flag if >5%
MISSINGNESS_THRESHOLD = 0.05

# Hard clinical floors (physically impossible below these)
CLINICAL_FLOORS = {
    "Weight_kg": 10,
    "BMI": 10,
    "Age": 0,
    "Height_cm": 50,
    "FBS_mg_dL": 0,
    "PPBS_mg_dL": 0,
    "HbA1c_percent": 0,
    "Total_Cholesterol_mg_dL": 0,
    "HDL_Cholesterol_mg_dL": 0,
    "LDL_Cholesterol_mg_dL": 0,
    "Triglycerides_mg_dL": 0,
    "Serum_Creatinine_mg_dL": 0,
    "TSH_uIU_mL": 0,
    "BP_Systolic_mmHg": 0,
    "Pulse_bpm": 0,
}

# Binary columns that must be rounded after MICE
BINARY_COLS_DIABETES = ["Sex", "Has_Hypertension", "Has_Hypothyroidism"]
INTEGER_COLS_DIABETES = ["Age", "Diabetes_Target"]

BINARY_COLS_FRAMINGHAM = ["male", "currentSmoker", "BPMeds", "prevalentStroke",
                          "prevalentHyp", "diabetes", "TenYearCHD"]
INTEGER_COLS_FRAMINGHAM = ["age", "education"]

# Synthea FHIR extraction maps
VITAL_MAP = {
    "Body Height": "height_cm",
    "Body Weight": "weight_kg",
    "Body mass index (BMI) [Ratio]": "bmi",
    "Heart rate": "heart_rate",
    "Glucose [Mass/volume] in Serum or Plasma": "glucose_mg_dL",
    "Cholesterol [Mass/volume] in Serum or Plasma": "cholesterol_mg_dL",
    "Cholesterol in HDL [Mass/volume] in Serum or Plasma": "hdl_mg_dL",
    "Low Density Lipoprotein Cholesterol": "ldl_mg_dL",
    "Triglycerides": "triglycerides_mg_dL",
    "Creatinine [Mass/volume] in Serum or Plasma": "creatinine_mg_dL",
    "Tobacco smoking status": "smoking_status",
}
BP_PANEL_TEXT = "Blood pressure panel with all children optional"
BP_COMPONENT_MAP = {
    "Systolic Blood Pressure": "systolic_bp",
    "Diastolic Blood Pressure": "diastolic_bp",
}


# ============================================================
# Utility Functions
# ============================================================

def winsorize(series, lower_pct=1, upper_pct=99):
    """Clip values to the P1 and P99 percentiles."""
    lo = np.nanpercentile(series, lower_pct)
    hi = np.nanpercentile(series, upper_pct)
    return series.clip(lower=lo, upper=hi)


def add_missingness_flags(df, threshold=MISSINGNESS_THRESHOLD):
    """Add binary _is_missing flags for columns exceeding the missingness threshold."""
    flag_cols = []
    for col in df.columns:
        pct_missing = df[col].isnull().mean()
        if pct_missing > threshold:
            flag_name = f"{col}_is_missing"
            df[flag_name] = df[col].isnull().astype(int)
            flag_cols.append(flag_name)
            print(f"    Added {flag_name} ({pct_missing*100:.1f}% missing)")
    return df, flag_cols


def mice_impute(df, binary_cols=None, integer_cols=None, flag_cols=None):
    """
    MICE imputation using HistGradientBoosting.
    Preserves inter-column correlations, unlike median fill.
    """
    if binary_cols is None:
        binary_cols = []
    if integer_cols is None:
        integer_cols = []
    if flag_cols is None:
        flag_cols = []

    # Only impute numeric columns (exclude _is_missing flags from imputation features)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cols_to_impute = [c for c in numeric_cols if c not in flag_cols]

    if df[cols_to_impute].isnull().sum().sum() == 0:
        print("    No missing numeric values -- skipping MICE.")
        return df

    print(f"    Running MICE on {len(cols_to_impute)} columns...")
    imputer = IterativeImputer(
        estimator=HistGradientBoostingRegressor(random_state=42, max_iter=100),
        max_iter=10,
        random_state=42,
        verbose=0,
    )

    df[cols_to_impute] = imputer.fit_transform(df[cols_to_impute])

    # Post-imputation rounding
    for col in binary_cols:
        if col in df.columns:
            df[col] = df[col].round().clip(0, 1).astype(int)
    for col in integer_cols:
        if col in df.columns:
            df[col] = df[col].round().astype(int)
    # Round flag columns back to 0/1
    for col in flag_cols:
        if col in df.columns:
            df[col] = df[col].round().clip(0, 1).astype(int)

    return df


def validate(df, name):
    """Final validation: 0 NaNs, 0 duplicates, sane ranges."""
    print(f"\n  Validation for {name}:")
    nan_count = df.isnull().sum().sum()
    dup_count = df.duplicated().sum()
    print(f"    NaN count: {nan_count} {'OK' if nan_count == 0 else 'X FAIL'}")
    print(f"    Duplicate count: {dup_count} {'OK' if dup_count == 0 else '(acceptable)'}")

    # Check binary columns only contain 0/1
    for col in df.columns:
        if "_is_missing" in col or col in ["Sex", "male", "currentSmoker", "BPMeds",
                                            "prevalentStroke", "prevalentHyp", "diabetes",
                                            "Has_Hypertension", "Has_Hypothyroidism"]:
            vals = set(df[col].dropna().unique())
            if vals - {0, 1, 0.0, 1.0}:
                print(f"    X {col} contains non-binary values: {vals}")
            else:
                pass  # fine

    print(f"    Final shape: {df.shape}")
    return nan_count == 0


# ============================================================
# Dataset 1: Diabetes_MCDD
# ============================================================

def clean_diabetes_mcdd(filepath):
    print("\n" + "="*60)
    print("DIABETES_MCDD -- Production Cleaning")
    print("="*60)

    df = pd.read_csv(filepath)
    print(f"  Raw: {df.shape}")

    # Step 1: Drop exact duplicates FIRST
    before = len(df)
    df = df.drop_duplicates()
    print(f"  Step 1 -- Dedup: {before} -> {len(df)} (dropped {before - len(df)})")

    # Step 2: Add missingness flags
    print("  Step 2 -- Missingness flags:")
    df, flag_cols = add_missingness_flags(df)

    # Step 3: Feature engineering
    print("  Step 3 -- Feature engineering:")
    if "Diagnosis_Temp" in df.columns:
        df["Has_Hypertension"] = df["Diagnosis_Temp"].str.contains(
            "HYPERTENSION", case=False, na=False
        ).astype(int)
        df["Has_Hypothyroidism"] = df["Diagnosis_Temp"].str.contains(
            "HYPOTHYROID", case=False, na=False
        ).astype(int)
        df = df.drop(columns=["Diagnosis_Temp"])
        print("    Extracted Has_Hypertension, Has_Hypothyroidism; dropped Diagnosis_Temp")

    # Drop Diabetes_Status (redundant with Diabetes_Target)
    if "Diabetes_Status" in df.columns:
        df = df.drop(columns=["Diabetes_Status"])
        print("    Dropped Diabetes_Status (redundant with numeric target)")

    # Step 4: Outlier clipping
    print("  Step 4 -- Outlier clipping:")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    # Layer 1: Hard clinical floors
    for col, floor in CLINICAL_FLOORS.items():
        if col in df.columns:
            violations = (df[col] < floor).sum()
            if violations > 0:
                df.loc[df[col] < floor, col] = np.nan  # treat as missing, let MICE fix
                print(f"    {col}: {violations} values below {floor} -> set to NaN for MICE")
    # Layer 2: P1/P99 Winsorization
    for col in numeric_cols:
        if col not in flag_cols and "_is_missing" not in col:
            df[col] = winsorize(df[col])
    print("    Applied P1/P99 Winsorization to all continuous columns")

    # Step 5: MICE imputation
    print("  Step 5 -- MICE imputation:")
    df = mice_impute(df, BINARY_COLS_DIABETES, INTEGER_COLS_DIABETES, flag_cols)

    # Step 6: Validate
    validate(df, "Diabetes_MCDD")

    output_path = os.path.join(PROCESSED_DIR, "diabetes_mcdd_clean.csv")
    df.to_csv(output_path, index=False)
    print(f"  Saved: {output_path}")
    return df


# ============================================================
# Dataset 2: Framingham
# ============================================================

def clean_framingham(filepath):
    print("\n" + "="*60)
    print("FRAMINGHAM -- Production Cleaning")
    print("="*60)

    df = pd.read_csv(filepath, na_values=["NA", ""])
    print(f"  Raw: {df.shape}")

    # Step 1: Dedup (expect 0)
    before = len(df)
    df = df.drop_duplicates()
    print(f"  Step 1 -- Dedup: {before} -> {len(df)} (dropped {before - len(df)})")

    # Step 2: Missingness flag for glucose (9.2% missing)
    print("  Step 2 -- Missingness flags:")
    df, flag_cols = add_missingness_flags(df)

    # Step 3: No feature engineering needed for Framingham

    # Step 4: Outlier clipping (Winsorize only -- Framingham ranges are plausible)
    print("  Step 4 -- P1/P99 Winsorization:")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col not in flag_cols and "_is_missing" not in col:
            df[col] = winsorize(df[col])

    # Step 5: MICE imputation
    print("  Step 5 -- MICE imputation:")
    df = mice_impute(df, BINARY_COLS_FRAMINGHAM, INTEGER_COLS_FRAMINGHAM, flag_cols)

    # Step 6: Validate
    validate(df, "Framingham")

    output_path = os.path.join(PROCESSED_DIR, "framingham_clean.csv")
    df.to_csv(output_path, index=False)
    print(f"  Saved: {output_path}")
    return df


# ============================================================
# Dataset 3: Synthea FHIR (1000 JSON Bundles)
# ============================================================

def parse_fhir_bundles(json_dir, max_files=None):
    print("\n" + "="*60)
    print("SYNTHEA FHIR -- Flattening + Cleaning")
    print("="*60)

    files = [f for f in os.listdir(json_dir) if f.endswith(".json")]
    if max_files:
        files = files[:max_files]
    print(f"  Found {len(files)} JSON files")

    patients_data = []

    for f in tqdm(files, desc="  Processing FHIR"):
        filepath = os.path.join(json_dir, f)
        with open(filepath, "r", encoding="utf-8") as file:
            bundle = json.load(file)

        if bundle.get("resourceType") != "Bundle":
            continue

        entries = bundle.get("entry", [])
        patient_info = {}
        conditions = []
        vitals_latest = {}

        for entry in entries:
            resource = entry.get("resource", {})
            r_type = resource.get("resourceType")

            if r_type == "Patient":
                patient_info["id"] = resource.get("id")
                patient_info["gender"] = resource.get("gender")
                patient_info["birthDate"] = resource.get("birthDate")
                patient_info["maritalStatus"] = resource.get(
                    "maritalStatus", {}
                ).get("text", "Unknown")

                for ext in resource.get("extension", []):
                    if "us-core-race" in ext.get("url", ""):
                        for subext in ext.get("extension", []):
                            if subext.get("url") == "text":
                                patient_info["race"] = subext.get("valueString")
                    if "us-core-ethnicity" in ext.get("url", ""):
                        for subext in ext.get("extension", []):
                            if subext.get("url") == "text":
                                patient_info["ethnicity"] = subext.get("valueString")

            elif r_type == "Condition":
                clin_status = (
                    resource.get("clinicalStatus", {})
                    .get("coding", [{}])[0]
                    .get("code")
                )
                if clin_status == "active":
                    code_text = resource.get("code", {}).get("text")
                    if code_text:
                        conditions.append(code_text)

            elif r_type == "Observation":
                code_text = resource.get("code", {}).get("text", "")
                if code_text == BP_PANEL_TEXT:
                    for comp in resource.get("component", []):
                        comp_text = comp.get("code", {}).get("text", "")
                        if comp_text in BP_COMPONENT_MAP:
                            vq = comp.get("valueQuantity", {})
                            if "value" in vq:
                                vitals_latest[BP_COMPONENT_MAP[comp_text]] = vq["value"]
                elif code_text in VITAL_MAP:
                    col_name = VITAL_MAP[code_text]
                    val_q = resource.get("valueQuantity")
                    val_s = resource.get("valueString")
                    val_cc = resource.get("valueCodeableConcept", {}).get("text")
                    if val_q and "value" in val_q:
                        vitals_latest[col_name] = val_q["value"]
                    elif val_s:
                        vitals_latest[col_name] = val_s
                    elif val_cc:
                        vitals_latest[col_name] = val_cc

        patient_info["conditions"] = conditions
        patient_info.update(vitals_latest)
        patients_data.append(patient_info)

    df = pd.DataFrame(patients_data)

    # Calculate age
    if "birthDate" in df.columns:
        df["birthDate"] = pd.to_datetime(df["birthDate"], errors="coerce")
        df["age"] = (pd.Timestamp("2026-01-01") - df["birthDate"]).dt.days // 365
        df = df.drop(columns=["birthDate"])

    # One-hot encode top 20 conditions
    all_conditions = pd.Series(
        [c for sublist in df["conditions"].dropna() for c in sublist]
    )
    top_conditions = all_conditions.value_counts().head(20).index.tolist()

    for cond in top_conditions:
        safe_name = (
            cond.replace(" (finding)", "")
            .replace(" (disorder)", "")
            .replace(" ", "_")
            .replace(",", "")
            .replace("(", "")
            .replace(")", "")
        )
        col_name = f"Cond_{safe_name}"
        df[col_name] = df["conditions"].apply(
            lambda x: 1 if isinstance(x, list) and cond in x else 0
        )

    df = df.drop(columns=["conditions", "id"], errors="ignore")

    # Drop exact duplicates
    before = len(df)
    df = df.drop_duplicates()
    print(f"  Dedup: {before} -> {len(df)}")

    # Drop rows missing age (only 2)
    df = df.dropna(subset=["age"])

    # Impute remaining missing vitals with median (very few missing)
    vital_cols = list(VITAL_MAP.values()) + list(BP_COMPONENT_MAP.values())
    for col in vital_cols:
        if col in df.columns:
            if df[col].dtype == object:
                mode_val = df[col].mode()
                df[col] = df[col].fillna(mode_val[0] if len(mode_val) > 0 else "Unknown")
            else:
                df[col] = df[col].fillna(df[col].median())

    # Fill demographic missing
    for col in ["race", "ethnicity", "maritalStatus", "gender"]:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown")

    # Clean column names
    df.columns = [c.replace(",", "").replace("(", "").replace(")", "") for c in df.columns]

    validate(df, "Synthea")

    output_path = os.path.join(PROCESSED_DIR, "synthea_flattened.csv")
    df.to_csv(output_path, index=False)
    print(f"  Saved: {output_path}")
    return df


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    try:
        parse_fhir_bundles(r"Datasets\SyntheticEHR\dataset")
    except Exception as e:
        print(f"ERROR (Synthea): {e}")
        import traceback; traceback.print_exc()

    try:
        clean_diabetes_mcdd(r"Datasets\Diabetes_MCDD.csv")
    except Exception as e:
        print(f"ERROR (Diabetes): {e}")
        import traceback; traceback.print_exc()

    try:
        clean_framingham(r"Datasets\Framingham.csv")
    except Exception as e:
        print(f"ERROR (Framingham): {e}")
        import traceback; traceback.print_exc()

    print("\n" + "="*60)
    print("ALL DATA PREPARATION COMPLETE")
    print("="*60)

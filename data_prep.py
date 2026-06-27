import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

def create_dirs():
    dirs = [
        "synthetic-health-sdg/data/raw",
        "synthetic-health-sdg/data/processed",
        "synthetic-health-sdg/models",
        "synthetic-health-sdg/eval"
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)


# Key vitals to extract — map FHIR observation text → column name
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
# BP panel uses components, handled separately
BP_PANEL_TEXT = "Blood pressure panel with all children optional"
BP_COMPONENT_MAP = {
    "Systolic Blood Pressure": "systolic_bp",
    "Diastolic Blood Pressure": "diastolic_bp",
}

def parse_fhir_bundles(json_dir, max_files=None):
    print(f"Parsing FHIR bundles from {json_dir}...")
    files = [f for f in os.listdir(json_dir) if f.endswith('.json')]
    if max_files:
        files = files[:max_files]

    patients_data = []

    for f in tqdm(files, desc="Processing FHIR"):
        filepath = os.path.join(json_dir, f)
        with open(filepath, 'r', encoding='utf-8') as file:
            bundle = json.load(file)

        if bundle.get('resourceType') != 'Bundle':
            continue

        entries = bundle.get('entry', [])

        patient_info = {}
        conditions = []
        vitals_latest = {}  # store latest value per vital type

        for entry in entries:
            resource = entry.get('resource', {})
            r_type = resource.get('resourceType')

            if r_type == 'Patient':
                patient_info['id'] = resource.get('id')
                patient_info['gender'] = resource.get('gender')
                patient_info['birthDate'] = resource.get('birthDate')
                patient_info['maritalStatus'] = resource.get('maritalStatus', {}).get('text', 'Unknown')

                for ext in resource.get('extension', []):
                    if 'us-core-race' in ext.get('url', ''):
                        for subext in ext.get('extension', []):
                            if subext.get('url') == 'text':
                                patient_info['race'] = subext.get('valueString')
                    if 'us-core-ethnicity' in ext.get('url', ''):
                        for subext in ext.get('extension', []):
                            if subext.get('url') == 'text':
                                patient_info['ethnicity'] = subext.get('valueString')

            elif r_type == 'Condition':
                clin_status = resource.get('clinicalStatus', {}).get('coding', [{}])[0].get('code')
                if clin_status == 'active':
                    code_text = resource.get('code', {}).get('text')
                    if code_text:
                        conditions.append(code_text)

            elif r_type == 'Observation':
                code_text = resource.get('code', {}).get('text', '')
                # Handle BP panel with components
                if code_text == BP_PANEL_TEXT:
                    for comp in resource.get('component', []):
                        comp_text = comp.get('code', {}).get('text', '')
                        if comp_text in BP_COMPONENT_MAP:
                            vq = comp.get('valueQuantity', {})
                            if 'value' in vq:
                                vitals_latest[BP_COMPONENT_MAP[comp_text]] = vq['value']
                elif code_text in VITAL_MAP:
                    col_name = VITAL_MAP[code_text]
                    val_q = resource.get('valueQuantity')
                    val_s = resource.get('valueString')
                    val_cc = resource.get('valueCodeableConcept', {}).get('text')
                    if val_q and 'value' in val_q:
                        vitals_latest[col_name] = val_q['value']
                    elif val_s:
                        vitals_latest[col_name] = val_s
                    elif val_cc:
                        vitals_latest[col_name] = val_cc

        patient_info['conditions'] = conditions
        patient_info.update(vitals_latest)
        patients_data.append(patient_info)

    df = pd.DataFrame(patients_data)

    # Calculate age
    if 'birthDate' in df.columns:
        df['birthDate'] = pd.to_datetime(df['birthDate'], errors='coerce')
        df['age'] = (pd.Timestamp('2026-01-01') - df['birthDate']).dt.days // 365
        df = df.drop(columns=['birthDate'])

    # One-hot encode top 20 conditions
    all_conditions = pd.Series([c for sublist in df['conditions'].dropna() for c in sublist])
    top_conditions = all_conditions.value_counts().head(20).index.tolist()

    for cond in top_conditions:
        col_name = f"Cond_{cond.replace(' (finding)', '').replace(' (disorder)', '').replace(' ', '_')}"
        df[col_name] = df['conditions'].apply(lambda x: 1 if isinstance(x, list) and cond in x else 0)

    df = df.drop(columns=['conditions', 'id'], errors='ignore')

    # Impute missing vitals with median
    vital_cols = list(VITAL_MAP.values())
    for col in vital_cols:
        if col in df.columns:
            if df[col].dtype == object:
                df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown')
            else:
                df[col] = df[col].fillna(df[col].median())

    # Fill demographic missing
    for col in ['race', 'ethnicity', 'maritalStatus', 'gender']:
        if col in df.columns:
            df[col] = df[col].fillna('Unknown')

    # Clean column names
    df.columns = [c.replace(',', '').replace('(', '').replace(')', '') for c in df.columns]

    output_path = "synthetic-health-sdg/data/processed/synthea_flattened.csv"
    df.to_csv(output_path, index=False)
    print(f"Saved {output_path} with shape {df.shape}")
    return df


def clean_diabetes_mcdd(filepath):
    print(f"Cleaning {filepath}...")
    df = pd.read_csv(filepath)
    
    # Handle specific encodings
    if 'Diabetes_Target' in df.columns:
        # 0=Non-Diabetic, 1=Pre-Diabetic, 2=Diabetic
        pass
        
    # Impute missing values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())
        
    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown')
        
    # Simplify Diagnosis_Temp if present
    if 'Diagnosis_Temp' in df.columns:
        df['Has_Hypertension'] = df['Diagnosis_Temp'].str.contains('HYPERTENSION', case=False, na=False).astype(int)
        df['Has_Hypothyroidism'] = df['Diagnosis_Temp'].str.contains('HYPOTHYROIDISM', case=False, na=False).astype(int)
        df = df.drop(columns=['Diagnosis_Temp'])
        
    # Drop near duplicates (same everything except Diagnosis_Temp which we just parsed)
    df = df.drop_duplicates()
    
    output_path = "synthetic-health-sdg/data/processed/diabetes_mcdd_clean.csv"
    df.to_csv(output_path, index=False)
    print(f"Saved {output_path} with shape {df.shape}")
    return df

def clean_framingham(filepath):
    print(f"Cleaning {filepath}...")
    df = pd.read_csv(filepath, na_values=['NA', ''])
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())
        
    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown')
        
    output_path = "synthetic-health-sdg/data/processed/framingham_clean.csv"
    df.to_csv(output_path, index=False)
    print(f"Saved {output_path} with shape {df.shape}")
    return df

if __name__ == "__main__":
    create_dirs()
    
    try:
        # Full 1000 JSONs processing
        df_syn = parse_fhir_bundles(r"c:\Users\jagat\Desktop\IEEE\Datasets\SyntheticEHR\dataset")
    except Exception as e:
        print(f"Error parsing FHIR: {e}")
        
    try:
        df_diab = clean_diabetes_mcdd(r"c:\Users\jagat\Desktop\IEEE\Datasets\Diabetes_MCDD.csv")
    except Exception as e:
        print(f"Error cleaning Diabetes: {e}")
        
    try:
        df_fram = clean_framingham(r"c:\Users\jagat\Desktop\IEEE\Datasets\Framingham.csv")
    except Exception as e:
        print(f"Error cleaning Framingham: {e}")

    print("Data preparation complete.")

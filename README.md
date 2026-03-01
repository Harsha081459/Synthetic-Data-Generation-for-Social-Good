# Synthetic Health Data Generation for Social Good (SDG)

## Abstract

Healthcare data is hard to access for research because of strict privacy regulations like HIPAA — hospitals can't just share patient records even when researchers genuinely need them. Our team built a pipeline to solve this by generating synthetic patient data that is statistically realistic enough for machine learning, but mathematically private enough to share safely.

Our primary dataset is a collection of 1000 longitudinal FHIR patient records (Synthea), which we flatten from raw JSON bundles into structured tabular data — demographics, active conditions, and vitals. We also run the same pipeline on a diabetes classification dataset and the Framingham heart disease study to show it generalises. For the generative model, we use TabDDPM, a diffusion-based approach that consistently outperforms older methods like TVAE and CTGAN on structured tabular data. We compare it against a TVAE baseline to quantify the improvement.

To measure utility objectively, we run Train-on-Synthetic, Test-on-Real (TSTR) experiments—training a classifier entirely on synthetic data and evaluating it on real held-out records. We also integrate Differential Privacy (DP-SGD via Opacus) directly into the generative training loops. Ultimately, our interactive dashboard explicitly visualizes the critical privacy-utility tradeoff: demonstrating exactly how tightening mathematical privacy bounds (lower epsilon) impacts downstream machine learning accuracy.

## Repository Structure

```
synthetic-health-sdg/
├── data_prep.py                  ← Run this first to generate clean CSVs
├── requirements.txt              ← pip install -r requirements.txt
├── app.py                        ← streamlit run app.py
├── data/
│   └── processed/
│       ├── synthea_flattened.csv      ← 1000 patients × 38 columns (MAIN)
│       ├── diabetes_mcdd_clean.csv    ← 6874 × 20
│       └── framingham_clean.csv       ← 4240 × 16
├── models/
│   ├── train_tvae.py             ← TVAE baseline
│   ├── compare_baselines.py      ← CTGAN vs TVAE
│   └── dp_tvae.py                ← Differential Privacy tradeoff
└── eval/
    └── evaluate.py               ← TSTR + DCR evaluation metrics
```

## Setup

```bash
pip install -r requirements.txt
python data_prep.py          # generates data/processed/ CSVs
```

## Team Workflow

| Person | Task | Script |
|--------|------|--------|
| Friend 1 | TVAE + CTGAN baselines on 3 datasets | `models/train_tvae.py`, `models/compare_baselines.py` |
| Friend 2 | TabDDPM (main model) — clone https://github.com/rotot0/tab-ddpm | Feed processed CSVs |
| You | Evaluation + Dashboard | `eval/evaluate.py` → `app.py` |

# 🧬 SynthoGen AI: Privacy-Preserving Synthetic Healthcare Data

![CI](https://github.com/Harsha081459/Synthetic-Data-Generation-for-Social-Good/actions/workflows/ci.yml/badge.svg)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://python.org)
[![PyTorch 2.1+](https://img.shields.io/badge/PyTorch-2.1+-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?logo=streamlit&logoColor=white)](https://synthetic-data-generation-for-social-good.streamlit.app)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![IEEE DataPort](https://img.shields.io/badge/IEEE_DataPort-Published-00629B?logo=ieee&logoColor=white)](https://ieee-dataport.org/documents/provably-private-synthetic-ehr-cohorts-latent-diffusion-tabsyn)

> **IEEE DataPort Hackathon 2026** — A research prototype comparing synthetic healthcare-data generators, classifier utility and empirical privacy diagnostics. Includes an experimental Opacus DP-SGD training path; no end-to-end privacy or clinical-safety guarantee is established.

---

## 🌐 Live Demo

**🔗 [synthetic-data-generation-for-social-good.streamlit.app](https://synthetic-data-generation-for-social-good.streamlit.app)**

**📄 [IEEE DataPort Publication (DOI: 10.21227/64c7-vj34)](https://ieee-dataport.org/documents/provably-private-synthetic-ehr-cohorts-latent-diffusion-tabsyn)**

---

## 🎯 Problem Statement

Healthcare AI is critically bottlenecked by **patient privacy regulations** (HIPAA, GDPR). Researchers cannot freely share or use real Electronic Health Records (EHR) for machine learning without risking re-identification of patients.

**SynthoGen AI** explores this problem through four model families, a comparison dashboard and a cohort-sampling demo:
- Compare distributions and classifier performance using saved evaluation artifacts.
- Inspect sampled near-duplicate counts and distance-based privacy heuristics.
- Run the dashboard from committed outputs without model weights or API credentials.
- Train TVAE, CTGAN, TabDDPM and TabSyn separately; their standard training runs are not DP-SGD runs.

The saved reports contain up to **94.59% classifier accuracy**, not 94.59% retained utility. Generator-level train/test separation is not documented, so these historical numbers must not be presented as an independently validated, leakage-free benchmark.

---

## 🏗️ Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                    SynthoGen AI Pipeline                       │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  Raw EHR Data ──► Data Preprocessing ──► Feature Engineering   │
│       │              (data_prep.py)         (Cleaning, Norm)   │
│       │                                                        │
│       ▼                                                        │
│  ┌──────────────────────────────────────────────────────┐      │
│  │           Generative Model Training Suite            │      │
│  │  ┌─────────┐ ┌─────────┐ ┌──────────┐ ┌──────────┐ │      │
│  │  │  TVAE   │ │  CTGAN  │ │ TabDDPM  │ │  TabSyn  │ │      │
│  │  │ (VAE)   │ │ (GAN)   │ │(Diffusion│ │ (Latent  │ │      │
│  │  │         │ │         │ │  Model)  │ │Diffusion)│ │      │
│  │  └─────────┘ └─────────┘ └──────────┘ └──────────┘ │      │
│  └──────────────────────────────────────────────────────┘      │
│       │                                                        │
│       ▼                                                        │
│  Comprehensive Evaluation Engine                               │
│  ├── Utility: TSTR (Train-Synthetic, Test-Real)                │
│  ├── Privacy: DCR, K-Anonymity, Re-Identification Risk         │
│  ├── Fidelity: Correlation MAE, Distribution Matching          │
│  └── Fairness: Bias Auditing across demographics               │
│       │                                                        │
│       ▼                                                        │
│  Production Dashboard (Streamlit)                              │
│  ├── Interactive Leaderboard & Metrics                         │
│  ├── Live Patient Generator (SDV + Diffusion Pool)             │
│  ├── Prompt-to-Patient (Groq LLM → Structured Generation)     │
│  └── DP-SGD Privacy-Utility Tradeoff Ablation                  │
└────────────────────────────────────────────────────────────────┘
```

---

## 📊 Key Results

### Per-Dataset TSTR Accuracy (Train on Synthetic, Test on Real)

| Model | Diabetes MCDD | Framingham Heart | Synthea EHR |
|-------|:------------:|:----------------:|:-----------:|
| **TabDDPM** | **94.59%** | 84.32% | 66.0% |
| **TabSyn** | 94.10% | **84.79%** | **83.5%** |
| **TVAE** | 94.47% | 84.79% | 80.5% |
| **CTGAN** | 88.94% | 83.02% | 74.0% |

### Privacy Metrics (Averaged Across All Datasets)

| Model | Avg DCR ↑ | Mean group size ↑ | Distance-risk score ↓ | Sampled near-duplicates |
|-------|:---------:|:-------------:|:------------:|:----------------:|
| **TabSyn** | 7.11 | 61.2 | 0.218 | **0** |
| **TabDDPM** | 7.99 | 176.7 | 0.224 | **0** |
| **CTGAN** | 3.98 | 59.6 | 0.233 | **0** |
| **TVAE** | 2.41 | 60.4 | 0.336 | **0** |

> These historical reports found zero near-duplicates in sampled comparisons (at most 2,000 reference and synthetic rows). That is **not** zero privacy breaches. Mean group size is not minimum k-anonymity, and the distance-risk score is not a calibrated re-identification probability.

---

## 📂 Repository Structure

```
├── app.py                      # Production Streamlit dashboard
├── gemini_parser.py            # LLM prompt parser (Groq/Gemini API)
├── prompt_parser.py            # Natural language → structured constraints
├── patient_generator.py        # Synthetic patient generation engine
├── data_prep.py                # Data cleaning and preprocessing pipeline
├── requirements.txt            # Python dependencies
├── architecture_pipeline.md    # Detailed architecture documentation
│
├── data/
│   ├── processed/              # Cleaned real-world datasets
│   │   ├── diabetes_mcdd_clean.csv
│   │   ├── framingham_clean.csv
│   │   └── synthea_flattened.csv
│   └── synthetic/              # Generated synthetic datasets (4 models × 3 datasets)
│       ├── tabsyn_*.csv
│       ├── tabddpm_*.csv
│       ├── ctgan_*.csv
│       └── tvae_*.csv
│
├── eval/                       # Evaluation reports and scripts
│   ├── evaluate.py             # Core evaluation engine
│   ├── ablation_study.py       # DP-SGD epsilon ablation
│   ├── ablation_results.json   # Ablation study results
│   └── report_*_full.json      # Per-model per-dataset evaluation reports
│
├── models/                     # Model training scripts
│   ├── train_tvae.py
│   ├── train_ctgan.py
│   ├── train_tabddpm.py
│   ├── train_tabsyn.py
│   ├── dp_tvae.py              # Differentially-private TVAE
│   └── balanced_generator.py   # Class-balanced generation
│
└── saved_models/               # Trained model weights (NOT committed — see Quick Start)
    ├── tvae_*.pkl              # SDV pickle models (live inference)
    ├── ctgan_*.pkl
    ├── tabddpm_*.pt            # PyTorch diffusion weights
    └── tabsyn_*.pt             # Latent diffusion weights
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.12 (tested runtime and Community Cloud deployment setting)
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/Harsha081459/Synthetic-Data-Generation-for-Social-Good.git
cd Synthetic-Data-Generation-for-Social-Good

# Create a virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS / Linux

# Install dependencies
pip install -r requirements.txt

# Run the dashboard
streamlit run app.py
```

> **What works from a clean clone:** the Leaderboard, Expanded Metrics,
> Bias & Fairness, and DP-SGD Ablation tabs read the committed
> `eval/report_*_full.json` / `eval/ablation_results.json` files, and the
> generator tabs sample from the committed `data/synthetic/*.csv` pools.
> `saved_models/*.pkl` (TVAE/CTGAN weights for true live re-inference) are
> **not committed** — when absent the app transparently falls back to the
> cached pools. For training or loading SDV checkpoint pickles, first install
> `pip install -r requirements-training.txt`, then use `models/train_tvae.py`
> or `models/train_ctgan.py`. The default cloud runtime deliberately excludes
> PyTorch, SDV, Opacus and the evaluation stack.
> Prompt-to-Patient works offline with the regex parser when `GROQ_API_KEY`
> is absent. Supplying a key enables an external Groq call containing the prompt;
> do not enter confidential patient information. Unsupported dataset conditions
> are rejected rather than silently ignored.

### Community Cloud deployment

Configure repository `Harsha081459/Synthetic-Data-Generation-for-Social-Good`, branch `main`, entrypoint `app.py`, and Python **3.12**. Community Cloud reads root `requirements.txt`, which now includes only `requirements-demo.txt`. No API key is needed for the cached dashboard or regex-based cohort demo.

If the host displays **Oh no / Error running app**, open **Manage app → Logs** to check whether installation failed or the process exceeded resources. Reboot the app after the latest commit is deployed. A successful GitHub test run is not proof that the hosted process restarted successfully. Do not install `requirements-training.txt` on the small dashboard instance unless you specifically need live SDV checkpoint inference.

### Tests

```bash
pip install -r requirements-dev.txt
python -m pytest tests/test_dashboard.py tests/test_cohort.py -q
```

For the full evaluation suite, also install `lightgbm==4.7.0 scikit-learn==1.9.1 scipy==1.17.1`, then run `python -m pytest -q`.

The suite covers prompt parsers, constraint enforcement, metric functions and Streamlit interaction tests. Dashboard tests use the committed datasets to navigate all six pages for all three datasets, sample all four model pools and generate a constrained cohort offline. No model checkpoints, credentials or external API calls are required.

For a small local demo install `requirements-demo.txt` instead of the training stack, then run `streamlit run app.py`. Select Live Generator, choose a model and generate 10 rows; Method must say `Cached model output` when checkpoints are absent. Try `Generate 5 female patients over age 30 with diabetes` in Prompt-to-Patient. The regex fallback handles simple positive constraints, not unrestricted clinical language.

The committed processed datasets total **13,285 rows** (8,047 + 4,240 + 998). The older 258K-scale claim is not supported by these files. Outputs can contain implausible clinical values; do not use this demo for clinical decisions. `patient_generator.py` is a separate legacy real-row perturbation helper, not a privacy-preserving release mechanism.

### Environment Variables (Optional — for Prompt-to-Patient feature)

Create a `.env` file in the project root:
```env
GROQ_API_KEY=your_groq_api_key_here
XAI_API_KEY=your_xai_api_key_here
```

---

## 🔬 Datasets

| Dataset | Source | Records | Target Variable | Domain |
|---------|--------|:-------:|-----------------|--------|
| **Diabetes MCDD** | MCDD-derived processed artifact; original source/provenance needs confirmation | 8,047 committed rows | Diabetes Status (3-class) | Metabolic Disease |
| **Framingham Heart** | NHLBI | 4,240 | 10-Year CHD Risk (binary) | Cardiovascular |
| **Synthea EHR** | Synthea™ | 998 | Hypertension (binary) | General Practice |

All synthetic datasets are published on **[IEEE DataPort](https://ieee-dataport.org/documents/provably-private-synthetic-ehr-cohorts-latent-diffusion-tabsyn)** under DOI: `10.21227/64c7-vj34`.

---

## Privacy diagnostics and scope

1. **DCR:** nearest-neighbour distance after numeric scaling, on sampled rows.
2. **Grouped quasi-identifiers:** minimum/mean group sizes after binning; this does not prove anonymisation of original records.
3. **Distance-risk heuristic:** `eval_comprehensive.py` uses `1/(1+DCR)`; `evaluate.py` uses `exp(-DCR)`. Their scores are not interchangeable and neither measures actual attack probability.
4. **Near-duplicate count:** distances below `1e-6` in the evaluated subset; not an exhaustive privacy-breach assessment.
5. **DP-SGD experiment:** Opacus accounts for training gradients. `models/dp_tvae.py` fits a quantile transformer and categorical encoders on private data, then uses that transformer to generate outputs without accounting for its privacy loss. This prevents an end-to-end DP claim. Infinite epsilon is explicitly non-private.

---

## 🧪 Evaluation Methodology

- **TSTR:** LightGBM trained on synthetic data, scored on a classifier-level real-data split. The generator scripts train on their whole input CSV; generator-level holdout provenance is missing. The dashboard's `_full.json` schema comes from `eval/eval_comprehensive.py`, not the alternate `evaluate.py`.
- **Correlation Matrix MAE:** Measures how well inter-feature correlations are preserved
- **Distribution Fidelity:** KDE-based comparison of marginal distributions
- **Bias & Fairness Audit:** Demographic parity analysis across protected attributes

---

## ⚠️ Limitations

- **Privacy metrics are empirical audits, not proofs.** DCR, k-anonymity, and
  re-identification risk in `eval/evaluate.py` measure distance/coverage on the
  evaluated samples — "0 privacy breaches" means zero synthetic rows within
  1e-6 of a sampled reference row, not a guarantee against memorization. The
  DP-SGD ablation accounts only for training, not its private preprocessing.
- **TSTR uses a single classifier family.** Utility is measured with
  LightGBM only (`eval/evaluate.py`); results may differ for other model
  families.
- **The Synthea "real" dataset is itself simulated.** Evaluating against it
  is not the same as evaluating against real EHR, and its evaluation subset is
  small (998 records — see Datasets table).
- **Dashboard "live generation" is model-dependent.** TabDDPM/TabSyn sample
  from committed model-output pools without adding noise by default
  (`app.py` `generate_with_model`); only TVAE/CTGAN run true inference, and
  only when `saved_models/*.pkl` exist (not committed).

---

## 🛠️ Technology Stack

| Component | Technology |
|-----------|-----------|
| **Generative Models** | TVAE, CTGAN (SDV), TabDDPM, TabSyn (PyTorch) |
| **Evaluation** | LightGBM, Scikit-learn, SciPy |
| **Dashboard** | Streamlit, Plotly |
| **LLM Integration** | Groq (Llama 3.3 70B) for natural language parsing |
| **Training Infrastructure** | NVIDIA GPU Server (CUDA 12.x) |
| **Deployment** | Streamlit Community Cloud |
| **Data Publication** | IEEE DataPort |

---

## 👥 Team

| Name | 
|------|
| **Harsha Vardhan D** |
| **Lohith P** |
| **Anish Reddy** |
| **Vishal Sriram K** |

---

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details on:
- Setting up the development environment
- Running tests
- Submitting pull requests
- Code style guidelines

---

## 📜 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## 📖 Citation

If you use our synthetic datasets or methodology, please cite:

```bibtex
@misc{synthogen_ai_2026,
  title   = {Provably Private Synthetic EHR Cohorts via Latent Diffusion (TabSyn)},
  author  = {Harsha Vardhan D and Lohith P and Anish Reddy and Vishal Sriram K},
  year    = {2026},
  doi     = {10.21227/64c7-vj34},
  url     = {https://ieee-dataport.org/documents/provably-private-synthetic-ehr-cohorts-latent-diffusion-tabsyn},
  note    = {IEEE DataPort}
}
```

---

<p align="center">
  <b>Built with ❤️ for the IEEE DataPort Hackathon 2026</b><br>
  <i>Generating privacy-safe healthcare data so researchers don't have to choose between innovation and patient safety.</i>
</p>

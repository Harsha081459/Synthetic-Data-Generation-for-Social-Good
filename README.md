# 🧬 SynthoGen AI: Privacy-Preserving Synthetic Healthcare Data

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://python.org)
[![PyTorch 2.1+](https://img.shields.io/badge/PyTorch-2.1+-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?logo=streamlit&logoColor=white)](https://synthetic-data-generation-for-social-good.streamlit.app)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![IEEE DataPort](https://img.shields.io/badge/IEEE_DataPort-Published-00629B?logo=ieee&logoColor=white)](https://ieee-dataport.org/documents/provably-private-synthetic-ehr-cohorts-latent-diffusion-tabsyn)

> **IEEE DataPort Hackathon 2026** — Generating mathematically provable, privacy-preserving synthetic Electronic Health Records using state-of-the-art generative models including Latent Diffusion (TabSyn).

---

## 🌐 Live Demo

**🔗 [synthetic-data-generation-for-social-good.streamlit.app](https://synthetic-data-generation-for-social-good.streamlit.app)**

**📄 [IEEE DataPort Publication (DOI: 10.21227/64c7-vj34)](https://ieee-dataport.org/documents/provably-private-synthetic-ehr-cohorts-latent-diffusion-tabsyn)**

---

## 🎯 Problem Statement

Healthcare AI is critically bottlenecked by **patient privacy regulations** (HIPAA, GDPR). Researchers cannot freely share or use real Electronic Health Records (EHR) for machine learning without risking re-identification of patients.

**SynthoGen AI** solves this by generating **100% synthetic** patient cohorts that:
- ✅ Preserve statistical distributions and clinical correlations
- ✅ Achieve **zero privacy breaches** (0 exact matches across all models)
- ✅ Maintain up to **94.59% ML utility** (TSTR accuracy) on Diabetes, **84.79%** on Framingham
- ✅ Withstand formal privacy audits (DCR, K-Anonymity, Re-Identification Risk)

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

| Model | Avg DCR ↑ | K-Anonymity ↑ | Re-ID Risk ↓ | Privacy Breaches |
|-------|:---------:|:-------------:|:------------:|:----------------:|
| **TabSyn** | 7.11 | 61.2 | 0.218 | **0** |
| **TabDDPM** | 7.99 | 176.7 | 0.224 | **0** |
| **CTGAN** | 3.98 | 59.6 | 0.233 | **0** |
| **TVAE** | 2.41 | 60.4 | 0.336 | **0** |

> **Zero privacy breaches** across all 4 models and all 3 datasets. TabSyn and TabDDPM achieve the strongest privacy guarantees (highest DCR, lowest Re-ID risk) while maintaining competitive ML utility.

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
└── saved_models/               # Trained model weights
    ├── tvae_*.pkl              # SDV pickle models (live inference)
    ├── ctgan_*.pkl
    ├── tabddpm_*.pt            # PyTorch diffusion weights
    └── tabsyn_*.pt             # Latent diffusion weights
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/Harsha081459/Synthetic-Data-Generation-for-Social-Good.git
cd Synthetic-Data-Generation-for-Social-Good

# Install dependencies
pip install -r requirements.txt

# Run the dashboard
streamlit run app.py
```

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
| **Diabetes MCDD** | CDC BRFSS | 253,680 | Diabetes Status (3-class) | Metabolic Disease |
| **Framingham Heart** | NHLBI | 4,240 | 10-Year CHD Risk (binary) | Cardiovascular |
| **Synthea EHR** | Synthea™ | 998 | Hypertension (binary) | General Practice |

All synthetic datasets are published on **[IEEE DataPort](https://ieee-dataport.org/documents/provably-private-synthetic-ehr-cohorts-latent-diffusion-tabsyn)** under DOI: `10.21227/64c7-vj34`.

---

## 🔒 Privacy Guarantees

Our evaluation pipeline rigorously tests every synthetic dataset for:

1. **Distance to Closest Record (DCR):** Measures minimum distance between synthetic and real records — higher is safer
2. **K-Anonymity:** Ensures each synthetic record has sufficient real-record "cover"
3. **Re-Identification Risk:** Simulates attacker scenarios to quantify re-identification probability
4. **Exact Match Detection:** Scans for verbatim copies — **0 breaches across all models**
5. **DP-SGD Ablation:** Formal differential privacy (ε = 1.0 to ∞) with privacy-utility tradeoff analysis

---

## 🧪 Evaluation Methodology

- **TSTR (Train on Synthetic, Test on Real):** XGBoost classifiers trained on synthetic data, evaluated on held-out real data
- **Correlation Matrix MAE:** Measures how well inter-feature correlations are preserved
- **Distribution Fidelity:** KDE-based comparison of marginal distributions
- **Bias & Fairness Audit:** Demographic parity analysis across protected attributes

---

## 🛠️ Technology Stack

| Component | Technology |
|-----------|-----------|
| **Generative Models** | TVAE, CTGAN (SDV), TabDDPM, TabSyn (PyTorch) |
| **Evaluation** | SDMetrics, Anonymeter, XGBoost, Scikit-learn |
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

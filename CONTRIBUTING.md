# Contributing to SynthoGen AI

Thank you for your interest in contributing to **SynthoGen AI**! This guide will help you get started.

## 🚀 Getting Started

### Prerequisites

- Python 3.10 or higher
- pip
- (Optional) NVIDIA GPU with CUDA 12.x for training diffusion models

### Setting Up the Development Environment

```bash
# 1. Clone the repository
git clone https://github.com/Lohith248/Synthetic-Data-Generation-for-Social-Good.git
cd Synthetic-Data-Generation-for-Social-Good

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Optional) Set up API keys for the Prompt-to-Patient feature
cp .env.example .env
# Edit .env and add your API keys
```

### Running the Dashboard Locally

```bash
streamlit run app.py
```

The dashboard will be available at `http://localhost:8501`.

---

## 📂 Project Structure Overview

| Directory / File | Purpose |
|---|---|
| `app.py` | Streamlit production dashboard |
| `data_prep.py` | Data cleaning & preprocessing pipeline |
| `gemini_parser.py` | LLM prompt parser (Groq API + offline fallback) |
| `prompt_parser.py` | Natural language → structured constraints |
| `patient_generator.py` | Synthetic patient generation engine |
| `models/` | Training scripts for TVAE, CTGAN, TabDDPM, TabSyn |
| `eval/` | Evaluation scripts and result reports |
| `data/processed/` | Cleaned real-world datasets |
| `data/synthetic/` | Generated synthetic datasets |
| `tests/` | Unit and smoke tests |

---

## 🧪 Running Tests

We use `pytest` for testing. Run the test suite with:

```bash
python -m pytest tests/ -v
```

Please ensure all existing tests pass before submitting a PR.

---

## 🔬 Running Evaluations

To evaluate a synthetic dataset against the real data:

```bash
python eval/evaluate.py \
    --real data/processed/diabetes_mcdd_clean.csv \
    --synth data/synthetic/tabddpm_diabetes.csv \
    --target Diabetes_Target \
    --output eval/report_tabddpm_diabetes_full.json
```

---

## 🛠️ How to Contribute

### Reporting Bugs

1. Check existing [Issues](https://github.com/Lohith248/Synthetic-Data-Generation-for-Social-Good/issues) to avoid duplicates.
2. Open a new issue with:
   - A clear, descriptive title
   - Steps to reproduce the bug
   - Expected vs. actual behavior
   - Your environment (OS, Python version, GPU if relevant)

### Suggesting Features

Open an issue tagged with `enhancement` describing:
- The problem your feature would solve
- Your proposed solution
- Any alternatives you've considered

### Submitting Pull Requests

1. **Fork** the repository and create a feature branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```
2. Make your changes with clear, descriptive commit messages.
3. Add or update tests if your change affects functionality.
4. Ensure all tests pass: `python -m pytest tests/ -v`
5. Push to your fork and open a Pull Request against `main`.

### Code Style Guidelines

- Follow [PEP 8](https://peps.python.org/pep-0008/) for Python code.
- Use descriptive variable names and add docstrings for public functions.
- Keep functions focused — prefer small, composable functions over monolithic ones.
- Add type hints where practical, especially for public APIs.
- Use `logging` instead of `print()` in library code (scripts may use `print`).

---

## 📜 Code of Conduct

We are committed to providing a welcoming and inclusive experience for everyone. Please be respectful and constructive in all interactions.

---

## 💬 Questions?

If you have questions about contributing, feel free to open an issue or reach out to the maintainers.

Thank you for helping make healthcare AI more accessible and privacy-preserving! 🧬

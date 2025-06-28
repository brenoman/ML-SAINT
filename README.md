# SAINT vs Gradient Boosting & AutoML for Tabular Data

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

SAINT vs. Gradient Boosting & AutoML for Tabular Data

Official repository for the project comparing the SAINT (Self-Attention and Intersample Transformer) model with traditional models (LightGBM, XGBoost, CatBoost) and AutoML tools (AutoGluon, auto-sklearn) across 30 tabular datasets from the OpenML-CC18 benchmark.

🔍 Context

This is the final project for the Machine Learning course (2025), with the following objectives:

    To implement and evaluate SAINT, a Transformer-based model for tabular data.

    To compare its performance against Gradient Boosting and AutoML benchmarks.

    To apply a rigorous statistical protocol for comparison (Demšar's test).

🚀 Key Results

    ✅ AutoGluon achieved the best overall performance (accuracy).

    ⚡ LightGBM was the most efficient in terms of execution time.

    🧠 SAINT demonstrated competitive potential, particularly in One-vs-One AUC.

    📊 CatBoost showed the best generalization, with the lowest overfitting.

📊 Analyzed Metrics

    Accuracy

    One-vs-One AUC (AUC-OVO)

    Cross-Entropy

    Execution Time

🛠️ How to Reproduce

    Clone the repository:
    Bash

git clone https://github.com/brenoman/ML-SAINT.git

Install the dependencies (Anaconda is recommended):
Bash

    pip install -r requirements.txt

    Run the scripts (see the documentation for more details).

## 📄 Complete Documentation
- [Technical Report](docs/report.pdf)
- [Presentation](docs/presentation.pdf)

# SAINT vs. Gradient Boosting & AutoML for Tabular Data

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Official repository for the project comparing the **SAINT (Self-Attention and Intersample Transformer)** model with traditional models (LightGBM, XGBoost, CatBoost) and AutoML tools (AutoGluon, auto-sklearn) across 30 tabular datasets from the OpenML-CC18 benchmark.

## 🔍 Context
This is the final project for the Machine Learning course (2025), with the following objectives:
- To implement and evaluate SAINT, a Transformer-based model for tabular data.
- To compare its performance against Gradient Boosting and AutoML benchmarks.
- To apply a rigorous statistical protocol for comparison (Demšar's test).

## 🚀 Key Results
- ✅ **AutoGluon** achieved the best overall performance (accuracy).
- ⚡ **LightGBM** was the most efficient in terms of execution time.
- 🧠 **SAINT** demonstrated competitive potential, particularly in One-vs-One AUC.
- 📊 **CatBoost** showed the best generalization, with the lowest overfitting.

## 📊 Analyzed Metrics
- Accuracy
- One-vs-One AUC (AUC-OVO)
- Cross-Entropy
- Execution Time

## 🛠️ Setup and Installation

To ensure full reproducibility, it is highly recommended to use a Conda environment. This will manage all dependencies and guarantee that the correct package versions are used.

### Step 1: Create and Activate the Conda Environment

First, create a new Conda environment. We will name it `saint-project` and use Python 3.9, which is compatible with all required libraries.

```bash
# Create a new environment named 'saint-project' with Python 3.9
conda create -n saint-project python=3.8 -y
```

Next, activate the newly created environment. You must do this every time you work on the project in a new terminal session.

```bash
# Activate the environment
conda activate saint-project
```

### Step 2: Install Dependencies

With the environment activated, install all the necessary libraries from the `requirements.txt` file. This single command will handle the entire installation process.

```bash
# Install all required packages
pip install -r requirements.txt
```

### Step 3: Run the Experiments

You are now ready to run the project scripts. For example, to execute the main comparison script, you would run:

```bash
# Example of how to run a script
python total_hyper_comparator.py
```

By following these steps, you will have a clean and isolated environment with all the tools needed to reproduce the results of this analysis.


## 📄 Complete Documentation

The detailed findings, methodology, and statistical analysis are available in the full project report and presentation slides.

- [Technical Report](docs/report.pdf)
- [Presentation](docs/presentation.pdf)

# Explainable AI — Practical Notebooks

> **A hands-on exploration of six state-of-the-art Explainable AI (XAI) techniques, applied to computer vision, natural language processing, graph neural networks, and tabular data.**

---

## Table of Contents

- [Overview](#overview)
- [Notebooks](#notebooks)
  - [1. SHAP Values](#1-shap-values)
  - [2. Counterfactual Explanations — DiCE](#2-counterfactual-explanations--dice)
  - [3. Grad-CAM](#3-grad-cam)
  - [4. Layer-wise Relevance Propagation (LRP)](#4-layer-wise-relevance-propagation-lrp)
  - [5. GNN-LRP](#5-gnn-lrp)
  - [6. Interpreting BERT](#6-interpreting-bert)
- [Dependencies](#dependencies)
- [How to Run](#how-to-run)
- [Authors](#authors)

---

## Overview

This project is part of the **Explainable AI** module (M2 MALIA). Each notebook independently implements and demonstrates a different XAI method, covering the major families of explainability techniques:

| Family | Technique | Data Type |
|--------|-----------|-----------|
| **Feature attribution** | SHAP (Shapley values) | Tabular |
| **Counterfactual** | DiCE | Tabular |
| **Gradient-based** | Grad-CAM | Image |
| **Backpropagation-based** | Layer-wise Relevance Propagation | Tabular / FC networks |
| **Graph-based** | GNN-LRP | Graph |
| **Attention-based** | Layer Integrated Gradients (Captum) | Text (BERT) |

The notebooks are self-contained and runnable in Google Colab or any local Jupyter environment.

---

## Notebooks

### 1. SHAP Values

**File:** `SHAPLEY_Values.ipynb`

Introduces SHAP (SHapley Additive exPlanations), rooted in cooperative game theory. Each feature is treated as a "player" contributing to the model's prediction.

**Key concepts:**
- Formal derivation of Shapley values and the four axioms: Efficiency, Symmetry, Dummy, Additivity
- Step-by-step computation on a 3-feature credit risk example
- `TreeExplainer` applied to an **XGBoost** classifier on the **Breast Cancer Wisconsin** dataset (binary: malignant vs. benign)
- Beeswarm plots, force plots, and global feature importance visualizations

**Libraries:** `shap`, `xgboost`, `scikit-learn`, `matplotlib`

---

### 2. Counterfactual Explanations — DiCE

**File:** `Counterfactuals_pour_données_tabulaires.ipynb`

Implements **DiCE** (Diverse Counterfactual Explanations), which generates hypothetical "what-if" scenarios showing how a model's prediction would change if input features were different.

**Key concepts:**
- Loss function balancing proximity, diversity (via Determinantal Point Process), and prediction flip
- Real-world constraints (feature upper/lower bounds) integrated into generation
- Applied to **Heart Disease Prediction** (Cleveland UCI dataset, 303 patients) using a **Random Forest** classifier
- Clinical interpretation: which actionable risk factors would change the prediction for a given patient?

**Libraries:** `dice-ml`, `scikit-learn`, `pandas`, `matplotlib`

---

### 3. Grad-CAM

**File:** `GradCam_pour_analyse_d_images.ipynb`

Implements **Gradient-weighted Class Activation Mapping (Grad-CAM)**, which highlights the image regions most influential in a CNN's prediction by leveraging gradients flowing into the final convolutional layer.

**Key concepts:**
- Isolation of the last convolutional layer (`conv5_block3_out`) in **ResNet50**
- Gradient computation via `tf.GradientTape`
- Heatmap generation and overlay on original image
- Supports `prediction_rank=n` to visualize the nth top predicted class

**Libraries:** `tensorflow`, `keras`, `numpy`, `matplotlib`

---

### 4. Layer-wise Relevance Propagation (LRP)

**File:** `Layer-wise_Relevance_Propagation.ipynb`

A from-scratch implementation of **LRP** on a small fully connected neural network (3 inputs → 5 hidden neurons → 1 output), explaining how relevance scores are backpropagated from the output to each input neuron.

**Key concepts:**
- Architecture: ReLU hidden layer + sum-pooling output layer
- Derivation of LRP propagation rules
- Visualization of per-neuron relevance scores as a heatmap
- Users can explore custom weights and inputs to build intuition

**Libraries:** `scipy`, `matplotlib`

---

### 5. GNN-LRP

**File:** `GNN-LRP.ipynb`

Extends LRP to **Graph Neural Networks**, implementing walk-level relevance computation over graph structures. Based on the paper [*Explainability Methods for Graph Convolutional Neural Networks*](https://arxiv.org/abs/2006.03589).

**Key concepts:**
- Scale-free graph generation via the **Barabási–Albert model**
- Walk enumeration of length 3 over the adjacency matrix
- Relevance curve visualization (curved lines surrounding graph edges)
- Kamada-Kawai layout for graph rendering

**Libraries:** `python-igraph`, `torch`, `numpy`, `matplotlib`

---

### 6. Interpreting BERT

**File:** `Interpretation_de_BERT.ipynb`

Uses **Captum** (PyTorch's interpretability library) to explain predictions of a BERT-based question-answering model, applied to a **biomedical QA** context.

**Key concepts:**
- Fine-tuned `bert-base-uncased` on SQuAD2 for extractive QA
- `LayerIntegratedGradients` and `LayerConductance` from Captum
- Separate attribution of start and end answer span tokens
- Clinical use case: localizing answers in dense medical texts

**Libraries:** `transformers`, `captum`, `torch`, `matplotlib`, `seaborn`

---

## Dependencies

Each notebook installs its own dependencies via `pip`. For reference, the full list across all notebooks is:

```bash
pip install shap xgboost scikit-learn pandas numpy matplotlib seaborn
pip install dice-ml
pip install tensorflow keras
pip install scipy
pip install python-igraph torch
pip install transformers captum
```

> **Note:** All notebooks are compatible with **Google Colab** (recommended for GPU support on Grad-CAM and BERT notebooks).

---

## How to Run

### Option 1 — Google Colab (recommended)

1. Upload the notebook of your choice to [colab.research.google.com](https://colab.research.google.com)
2. Run all cells — dependencies are installed automatically via `!pip install` cells

### Option 2 — Local Jupyter

```bash
# Clone the repo
git clone https://github.com/SkanderHAJMABROUK/explainable-ai-notebooks
cd YOUR_REPO

# Install dependencies
pip install jupyter
pip install shap xgboost scikit-learn dice-ml tensorflow transformers captum torch python-igraph scipy matplotlib seaborn pandas numpy

# Launch Jupyter
jupyter notebook
```

---

## Authors

**Maram NASR** & **Skander HAJ MABROUK**
M2 MALIA — Université Lumière Lyon 2
Explainable AI Module
Academic Year 2025–2026

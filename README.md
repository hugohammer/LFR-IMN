# LFR-IMN: Local Fidelity Regularized Interpretable Mesomorphic Neural Networks

Official repository for the paper: **[Insert Paper Title Here]**.

## Overview
The **Interpretable Mesomorphic Neural Network (IMN)** is a powerful architecture that bridges the gap between the predictive performance of deep learning and the transparency of linear models for tabular data. It uses a neural network backbone to dynamically generate instance-wise linear weights, providing local feature attributions.

However, unregularized IMNs are prone to discovering degenerate solutions—collapsing predictive magnitude into the intercept and yielding unfaithful explanations. 

This repository implements **Local Fidelity Regularization (LFR)**, a computationally efficient penalty inspired by SMOTE interpolation. LFR grounds the network's generated explanations in the true geometric structure of the data, ensuring mathematically faithful interpretations while simultaneously improving overall predictive performance.

## Repository Structure
```text
LFR-IMN/
├── lfr_imn/                 # Core package directory
│   ├── __init__.py
│   ├── backbones.py         # PyTorch neural network backbones (MLP, TabResNet)
│   ├── model.py             # Main LFR_IMN wrapper class (fit, predict, explain)
│   └── metrics.py           # Evaluation metrics (Scaled Infidelity, Tuning Score S)
├── examples/                # Example scripts and tutorials
│   └── synthetic_example.py # End-to-end quickstart guide
├── requirements.txt
└── README.md
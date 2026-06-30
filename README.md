# LFR-IMN: Local Fidelity Regularized Interpretable Mesomorphic Neural Networks

<!--
Official repository for the paper: **[Insert Paper Title Here]**.
-->

## Overview
The [Local Fidelity Regularized Interpretable Mesomorphic Neural Network (LFR-IMN)](https://arxiv.org/abs/2606.29951) is a powerful architecture that bridges the gap between the predictive performance of deep learning and the transparency of linear models for tabular data. It uses a neural network backbone to dynamically generate instance-wise linear weights, providing local feature attributions. **Local Fidelity Regularization (LFR)**, a novel and computationally efficient penalty inspired by SMOTE interpolation, ensures that the network's generated explanations align with the true geometric structure of the data, guaranteeing mathematically faithful interpretations while simultaneously improving overall predictive performance.

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
```

## Installation

We recommend using a Python virtual environment to avoid dependency conflicts. To install and set up the repository, run the following commands in your terminal:

1. Clone the repository
```bash
git clone https://github.com/hugohammer/LFR-IMN.git
cd LFR-IMN
```

2. Create and activate a virtual environment (optional but recommended)
```bash
python3 -m venv venv
source venv/bin/activate
# On Windows, use: 
venv\Scripts\activate
```

3. Install the required dependencies
```bash
pip install -r requirements.txt
```

## Running the Example

We provide a ready-to-use synthetic data example to demonstrate how Local Fidelity Regularization (LFR) prevents degenerate weight collapse and perfectly recovers ground-truth feature attributions. The example can be run using either a MLP or TabResNet backbone, the latter used in the original IMN paper. To run the code with the TabResNet backbone, you will need access to the original [IMN codebase](https://github.com/ArlindKadra/IMN). 

To run the default example, simply execute:

```bash
python examples/synthetic_example.py
```

This script will:

1. Train an unregularized Baseline IMN.
2. Train an LFR-IMN and compute the tuning score $S(\gamma)$.
3. Output a comparison of the local feature attributions against the theoretical ground truth for a sample data point.

## Citation

If you use this code for your research, please cite the following paper:

> Hammer, H. L., Thambawita, V., Hellton, K., H., & Halvorsen, P. (2026). Improved Predictive Performance and Interpretability for Mesomorphic Neural Networks Using Local Fidelity Regularization. arXiv preprint arXiv:2606.29951.

**BibTeX:**
```bibtex

@article{hammer2026improved,
  title={Improved Predictive Performance and Interpretability for Mesomorphic Neural Networks Using Local Fidelity Regularization},
  author={Hammer, Hugo L, Thambawita, Vajira, Hellton, Kristoffer Herland and Halvorsen, P{\aa}l},
  journal={arXiv preprint arXiv:2606.29951},
  year={2026}
}
```

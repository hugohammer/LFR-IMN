# LFR-IMN: Local Fidelity Regularized Interpretable Mesomorphic Neural Networks

<!--
Official repository for the paper: **[Insert Paper Title Here]**.
-->

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

## Running the Examples

We provide a ready-to-use synthetic data example to demonstrate how Local Fidelity Regularization (LFR) prevents degenerate weight collapse and perfectly recovers ground-truth feature attributions.

To run the default example (which uses a Multilayer Perceptron backbone), simply execute:

```bash
python examples/synthetic_example.py
```

This script will:

1. Train an unregularized Baseline IMN.
2. Train an LFR-IMN and compute the tuning score $S$.
3. Output a comparison of the local feature attributions against the theoretical ground truth for a sample data point.

## Switching to the TabResNet Backbone

The LFR-IMN framework is designed to be completely modular. While the default example uses an MLPBackbone, you can easily swap it for the highly expressive TabResNetBackbone used in the original IMN paper.

To do this, you will need access to the original [IMN codebase](https://github.com/ArlindKadra/IMN). Open examples/synthetic_example.py and modify the imports and model initialization as follows:

1. Replace the current import block at the very top of examples/synthetic_example.py with the following code. This swaps the backbone and ensures Python can find both repositories:

```bash
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Add the original IMN directory to your path (adjust the path if necessary)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../IMN')))

import torch
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

from lfr_imn.backbones import TabResNetBackbone
from lfr_imn.model import LFR_IMN
from lfr_imn.metrics import compute_infidelity, compute_tuning_score

from models.hypernetwork import HyperNet
```


2. The example script trains two distinct models (Baseline and LFR-IMN). You will need to replace the `MLPBackbone` initialization in **both** sections to ensure they use separate TabResNet architectures. In **Section 2 (TRAINING BASELINE IMN)**, find these lines:
```python
backbone_base = MLPBackbone(num_features=P)
imn_model = LFR_IMN(backbone_base, device=device)
```

And replace them with:

```python
# Initialize a new TabResNet HyperNet and wrap it
hypernet_base = HyperNet(nr_features=P, nr_classes=1, nr_blocks=2, hidden_size=64)
backbone_base = TabResNetBackbone(hypernet_model=hypernet_base)
imn_model = LFR_IMN(backbone_base, device=device)
```

3. In **Section 3 (TRAINING LFR-IMN)**, find these lines:

```python
backbone_lfr = MLPBackbone(num_features=P)
lfr_model = LFR_IMN(backbone_lfr, device=device)
```
And replace them with:

```python
# Initialize a new, separate TabResNet HyperNet and wrap it
hypernet_lfr = HyperNet(nr_features=P, nr_classes=1, nr_blocks=2, hidden_size=64)
backbone_lfr = TabResNetBackbone(hypernet_model=hypernet_lfr)
lfr_model = LFR_IMN(backbone_lfr, device=device)
```

This ensures that users know exactly what to look for and guarantees their code will run successfully on the first try.

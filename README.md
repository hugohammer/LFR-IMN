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
python3 -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate

3. Install the required dependencies
pip install -r requirements.txt

## Running the Examples

We provide a ready-to-use synthetic data example to demonstrate how Local Fidelity Regularization (LFR) prevents degenerate weight collapse and perfectly recovers ground-truth feature attributions.

To run the default example (which uses a Multilayer Perceptron backbone), simply execute:

python examples/synthetic_example.py

This script will:

1. Train an unregularized Baseline IMN.
2. Train an LFR-IMN and compute the tuning score $S$.
3. Output a comparison of the local feature attributions against the theoretical ground truth for a sample data point.

## Switching to the TabResNet Backbone

The LFR-IMN framework is designed to be completely modular. While the default example uses an MLPBackbone, you can easily swap it for the highly expressive TabResNetBackbone used in the original IMN paper.

To do this, you will need access to the original IMN codebase. Open examples/synthetic_example.py and modify the imports and model initialization as follows:

1. Update the imports at the top of the file:

```bash
from lfr_imn.backbones import TabResNetBackbone
# Add the original IMN directory to your path (adjust the path as needed)
import sys
import os
sys.path.append(os.path.abspath('../IMN')) 
from models.hypernetwork import HyperNet
```

2. Update the model initialization:

```bash
# Create the original TabResNet HyperNet
raw_hypernet = HyperNet(nr_features=P, nr_classes=1, nr_blocks=2, hidden_size=64)

# Wrap it in the LFR-IMN compatible backbone
backbone = TabResNetBackbone(hypernet_model=raw_hypernet)

# Initialize the LFR-IMN model
model = LFR_IMN(backbone)
````

Because of the modular design, the .fit(), .predict(), and .explain() functions will work exactly the same way regardless of the underlying network architecture.


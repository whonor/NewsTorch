# NewsTorch

<p align="center">
    <br>
    <img src="docs/NewsTorch%20icon.png" width="400"/>
    <br>
<p>

[//]: # (<p align="center">)

[//]: # (<a href="README_CN.md">中文</a> &nbsp ｜ &nbsp English &nbsp)

[//]: # (</p>)

<p align="center">
<img src="https://img.shields.io/badge/python-3.9+-5be.svg">
<a href="https://pytorch.org/"><img src="https://img.shields.io/badge/pytorch-%E2%89%A52.10-orange.svg"></a>
<a href="https://github.com/whonor/NewsTorch/pulls"><img src="https://img.shields.io/badge/PR-welcome-55EB99.svg"></a>
<a href="https://github.com/whonor/NewsTorch/LICENSE"><img src="https://img.shields.io/badge/License-MIT-green.svg?labelColor=gray"></a>
</p>

___

## 📖 Table of Contents
- [Introduction](#-introduction)
- [Key Features](#-key-features)
- [Implemented Models](#-implemented-models)
- [Installation](#-installation)
- [Dataset Preparation](#-dataset-preparation)
- [Quick Start](#-quick-start)
- [Project Structure](#-project-structure)
- [Configuration](#-configuration)
- [Training](#-training)
- [Evaluation](#-evaluation)
- [Contributing](#-contributing)
- [License](#-license)
- [Citation](#-citation)

## 📝 Introduction

NewsTorch is a comprehensive PyTorch-based library for news recommendation research. It provides implementations of various neural news recommendation models, including both traditional deep learning-based methods and modern graph-based and large language model (LLM)-based approaches.

The framework is designed to facilitate the development, training, and evaluation of news recommendation systems. Built on top of PyTorch, it offers a unified interface for training and evaluating different models while including utilities for data preprocessing, feature extraction, and evaluation metrics.

With NewsTorch, researchers and practitioners can quickly implement and evaluate different news recommendation models with an original PyTorch environment.

## 🌟 Key Features

- **Diverse Model Implementations**: Supports a wide range of news recommendation models from classical approaches to cutting-edge techniques
- **Unified Framework**: Consistent API for training, validation, and testing across different models
- **Multi-Dataset Support**: Easy integration with popular datasets like MIND and EBNeRD
- **Flexible Configuration**: YAML-based configuration system for easy experimentation
- **Extensible Design**: Modular architecture for adding new models and datasets
- **Comprehensive Evaluation**: Built-in support for multiple evaluation metrics (AUC, MRR, nDCG, etc.)
- **Experiment Tracking**: Integration with Weights & Biases for experiment management

## 🍎 Implemented Models

| Model | Paper | Year | Category |
|-------|-------|------|----------|
| NPA | NPA: neural news recommendation with personalized attention | 2019 | DL-based |
| DKN | DKN: Deep knowledge-aware network for news recommendation | 2018 | DL-based |
| LSTUR | Neural news recommendation with long-and short-term user representations | 2019 | DL-based |
| NAML | Neural News Recommendation with Attentive Multi-View Learning | 2019 | DL-based |
| NRMS | Neural news recommendation with multi-head self-attention | 2019 | DL-based |
| FIM | Fine-grained Interest Matching for Neural News Recommendation| 2020 | DL-based |
| TANR | Neural news recommendation with topic-aware news representation | 2019 | DL-based |
| CenNewsRec | Privacy-Preserving News Recommendation Model Learning | 2020 | DL-based |
| MINS | News recommendation via multi-interest news sequence modelling | 2022 | DL-based |
| CNE-SUE | Neural News Recommendation with Collaborative News Encoding and Structural User Encoding | 2021 | Graph-based |
| IPNR | Intention-aware user modeling for personalized news recommendation | 2023 | Graph-based |
| MANNeR | Train once, use flexibly: A modular framework for multi-aspect neural news recommendation | 2024 | DL-based |
| LKPNR | LKPNR: Large Language Models and Knowledge Graph for Personalized News Recommendation Framework | 2024 | LLM-based |


## 🛠️ Installation

### Requirements
- Python 3.9+
- PyTorch 2.1.0+
- CUDA (for GPU training)

### Installation Steps

1. Create a new conda environment:
```bash
conda create -n newstorch python=3.9
conda activate newstorch
```

2. Install other dependencies:
```bash
cd scripts
./install_dependencies.sh
```

## 📦 Dataset Preparation

NewsTorch supports multiple datasets like MIND and EB-NeRD. Dataset preparation is divided into two steps:

1. Downloading the dataset by running the python files in the `dataset_download_prepare` directory
2. Preparing the dataset by running the python files in the `dataset_download_prepare` directory

### MIND Dataset Example

```bash
# Download and prepare the MIND dataset
python dataset_download_prepare/download_mind.py
python dataset_download_prepare/MIND_dataset_prepare.py
```

The dataset files should be organized in the following structure:
```
MIND-small/
├── train/
│   ├── news.tsv
│   ├── behaviors.tsv
│   ├── context_embedding.vec
│   └── entity_embedding.vec
├── dev/
│   ├── news.tsv
│   ├── behaviors.tsv
│   ├── context_embedding.vec
│   └── entity_embedding.vec
├── test/
│   ├── news.tsv
│   ├── behaviors.tsv
│   ├── context_embedding.vec
│   └── entity_embedding.vec
└── download/
    ├── MINDsmall_train.zip
    ├── MINDsmall_dev.zip
    └── wikidata-graph.zip
```

## 🚀 Quick Start

To train a model (e.g., NRMS) on the MIND dataset:

```bash
python general_runner.py --model=NRMS --batch_size=64 --epoch=10
```

To test a trained model:
```bash
python general_runner.py --mode=test --model=NRMS --test_model_path=path/to/your/model
```

## 📁 Project Structure

```
NewsTorch/
├── config/                 # Model configurations (YAML files)
├── dataset_corpus_preprocessing/ # Data preprocessing modules
├── dataset_download_prepare/     # Dataset download and preparation scripts
├── docs/                   # Documentation and images
├── models/                 # Model implementations
│   ├── modules/            # Reusable model components
│   └── [model_name].py     # Individual model files
├── utils/                  # Utility functions
├── general_runner.py       # Main training/evaluation script
├── config.py               # Configuration class
└── README.md
```

## ⚙️ Configuration

NewsTorch uses a combination of command-line arguments and YAML configuration files for model-specific settings:

1. **Command-line arguments**: Basic settings like model name, batch size, and epochs
2. **YAML files**: Model-specific hyperparameters in `config/[model_name].yaml`

Example configuration file (`config/nrms.yaml`):
```yaml
# NRMS specific hyperparameters
head_num: 20
head_dim: 20
attention_dim: 200
```

## 🏃 Training

The training process is handled by `general_runner.py` which supports different modes:

- **Train mode**: Train a model from scratch
- **Dev mode**: Evaluate on development set
- **Test mode**: Evaluate on test set

### Training a Model
```bash
python general_runner.py --model=NRMS --mode=train
```

### Key Training Features
- Automatic checkpoint saving
- Early stopping based on validation performance
- Multi-GPU support
- Experiment tracking with Weights & Biases

## 📊 Evaluation

NewsTorch computes multiple evaluation metrics during validation and testing:

- **AUC**: Area Under the ROC Curve
- **MRR**: Mean Reciprocal Rank
- **nDCG@5/10**: Normalized Discounted Cumulative Gain
- **Recall@5/10**: Recall at K
- **Hit@5/10**: Hit Rate at K
- **Precision@5/10**: Precision at K
- **MAE**: Mean Absolute Error
- **RMSE**: Root Mean Square Error

Results are automatically saved to CSV files for further analysis.

## 🤝 Contributing

We welcome contributions to NewsTorch! Here's how you can contribute:

1. Fork the repository
2. Create a new branch for your feature
3. Add your model or improvement
4. Write tests if applicable
5. Submit a pull request

Please ensure your code follows the existing style and includes appropriate documentation.

## 🏛 License

This framework is licensed under the [MIT License](LICENSE).

## 📎 Citation

If you use NewsTorch in your research, please cite our work:

```bibtex
@software{newstorch2025,
  author = {Rongyao Wang},
  title = {NewsTorch: A PyTorch Library for News Recommendation},
  year = {2025},
  url = {https://github.com/whonor/NewsTorch}
}
```
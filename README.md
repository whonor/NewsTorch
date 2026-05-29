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
- **GUI Support**: GUI for model training and evaluation

## 🍎 Implemented Models

| Model | Paper | Year | Category | Core Code Files                                                      | Config File           | Note                                                                                                                                                            |
|-------|-------|------|----------|----------------------------------------------------------------------|-----------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------|
| NPA | NPA: neural news recommendation with personalized attention | 2019 | DL-based | models/NPA.py general_runner.py                                      | config/npa.yaml       | None                                                                                                                                                            |
| DKN | DKN: Deep knowledge-aware network for news recommendation | 2018 | DL-based | models/DKN.py general_runner.py                                      | config/dkn.yaml       | None                                                                                                                                                            |
| LSTUR | Neural news recommendation with long-and short-term user representations | 2019 | DL-based | models/LSTUR.py  general_runner.py                                   | config/lstur.yaml     | None                                                                                                                                                            |
| NAML | Neural News Recommendation with Attentive Multi-View Learning | 2019 | DL-based | models/NAML.py  general_runner.py                                    | config/naml.yaml      | None                                                                                                                                                            |
| NRMS | Neural news recommendation with multi-head self-attention | 2019 | DL-based | models/NRMS.py  general_runner.py                                    | config/nrms.yaml      | None                                                                                                                                                            |
| FIM | Fine-grained Interest Matching for Neural News Recommendation| 2020 | DL-based | models/FIM.py  general_runner.py                                     | config/fim.yaml       | None                                                                                                                                                            |
| TANR | Neural news recommendation with topic-aware news representation | 2019 | DL-based | models/TANR.py  general_runner.py                                    | config/tanr.yaml      | None                                                                                                                                                            |
| CenNewsRec | Privacy-Preserving News Recommendation Model Learning | 2020 | DL-based | models/CenNewsRec.py  general_runner.py                              | config/cennewsrec.yaml | None                                                                                                                                                            |
| MINS | News recommendation via multi-interest news sequence modelling | 2022 | DL-based | models/MINS.py  general_runner.py                                    | config/mins.yaml      | None                                                                                                                                                            |
| CNE-SUE | Neural News Recommendation with Collaborative News Encoding and Structural User Encoding | 2021 | Graph-based | models/CNE_SUE.py  general_runner.py                                 | config/cne-sue.yaml   | None                                                                                                                                                            |
| IPNR | Intention-aware user modeling for personalized news recommendation | 2023 | Graph-based | models/IPNR.py models/modules/ipnr Conceptgraph general_runner.py    | config/ipnr.yaml      | First to download and execute Conceptgraph to generate graph data [here](https://drive.google.com/file/d/1rih15bSlXTHZg-JNdtxSoS3uiVfjjIyX/view?usp=sharing)    |
| LKPNR | LKPNR: Large Language Models and Knowledge Graph for Personalized News Recommendation Framework | 2024 | LLM-based | models/LKPNR.py models/modules/LKPNR  KGraph_LKPNR general_runner.py | config/lkpnr.yaml | First to download and execute KGraph_LKPNR to generate required data [here](https://drive.google.com/file/d/1eGiw6Cg7yH-bdjcXJnIBmRjjPlVde69s/view?usp=sharing) |
| ONCE | ONCE: Boosting Content-based Recommendation with Both Open- and Closed-source Large Language Models | 2024 | LLM-based | models/ONCE_DIRE_LLAMA1_NAML.py general_runner.py | config/once.yaml | Reproduces ONCE-DIRE-LLAMA with NAML-style user modeling. Requires local or Hugging Face-accessible LLaMA weights. Builds a cached lower-layer hidden-state store before training. |


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

⚠️ Note: Due to the public URL of wikidata-graph.zip missing, we provide a direct download link. You have to manually download it from [here](https://drive.google.com/file/d/1zoH9Aj03KM4KMa5FqJx9UV5CRwSrUjN0/view?usp=sharing) and then put it into the `download` directory before executing the `download_mind.py`.

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

To train the ONCE-DIRE reproduction on MIND, set the LLaMA checkpoint in `config/once.yaml` or set `ONCE_LLAMA_MODEL`. For gated Meta checkpoints, request Hugging Face access and authenticate with `huggingface-cli login` or `HF_TOKEN`.

```bash
python general_runner.py --dataset_name=MIND --DATASET_ROOT=MIND-small --dataset_size=small --model=ONCE --batch_size=64 --epoch=10
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
├── README.md
├── templates /
│   └──index.html           # Web UI template
├── run_webui.py            # Web UI starter
└── webui.py                # Web UI terminal
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
- **DCE@5/10**: Discounted Clickbait Exposure over valid per-article clickbait scores
- **MAE**: Mean Absolute Error
- **RMSE**: Root Mean Square Error

Results are automatically saved to CSV files for further analysis.


## Limitations

NewsTorch is still under active development. The current version has the following limitations.

### LLM Reproducibility

LLM-assisted news recommendation is difficult to reproduce exactly, especially when API-based LLMs are used. API-based models may change over time due to backend updates, decoding randomness, provider-specific changes, rate limits, pricing changes, and access availability.

To improve reproducibility, we recommend recording the LLM provider, model name, prompt template, decoding parameters, and API/local checkpoint version. Users are also encouraged to cache generated responses, embeddings, or user/news profiles. However, fully deterministic reproduction of API-based LLM results cannot be guaranteed.

### Dataset Coverage

The current validation mainly focuses on MIND and EB-NeRD. Although these are widely used datasets in news recommendation research, they do not cover all possible data schemas, languages, metadata fields, behavior logs, licensing requirements, or temporal evaluation protocols.

Additional dataset-specific adapters may be required to support other news recommendation datasets.

### Model Coverage

NewsTorch currently covers representative news recommendation models, including classical neural models, graph-based models, and initial LLM-assisted models. It does not aim to exhaustively implement all state-of-the-art architectures.

Some recent models require specialized preprocessing, external knowledge sources, proprietary modules, or large pretrained backbones. These models may require additional engineering work before being integrated into NewsTorch.

NewsTorch should therefore be viewed as an extensible benchmarking framework for controlled comparison, rather than a complete leaderboard of all news recommendation methods.

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

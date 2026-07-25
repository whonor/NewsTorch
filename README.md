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
| PLM-NR | PLM-empowered News Recommendation | 2021 | PLM-based | models/PLM_NR.py models/modules/plm_nr.py general_runner.py | config/plm-nr.yaml | RoBERTa-NRMS reproduction: title-only `roberta-base` news encoding with attention pooling; the final two RoBERTa layers are fine-tuned. |
| FIM | Fine-grained Interest Matching for Neural News Recommendation| 2020 | DL-based | models/FIM.py  general_runner.py                                     | config/fim.yaml       | None                                                                                                                                                            |
| TANR | Neural news recommendation with topic-aware news representation | 2019 | DL-based | models/TANR.py  general_runner.py                                    | config/tanr.yaml      | None                                                                                                                                                            |
| CenNewsRec | Privacy-Preserving News Recommendation Model Learning | 2020 | DL-based | models/CenNewsRec.py  general_runner.py                              | config/cennewsrec.yaml | None                                                                                                                                                            |
| MINS | News recommendation via multi-interest news sequence modelling | 2022 | DL-based | models/MINS.py  general_runner.py                                    | config/mins.yaml      | None                                                                                                                                                            |
| CNE-SUE | Neural News Recommendation with Collaborative News Encoding and Structural User Encoding | 2021 | Graph-based | models/CNE_SUE.py  general_runner.py                                 | config/cne-sue.yaml   | None                                                                                                                                                            |
| IPNR | Intention-aware user modeling for personalized news recommendation | 2023 | Graph-based | models/IPNR.py models/modules/ipnr Conceptgraph general_runner.py    | config/ipnr.yaml      | First to download and execute Conceptgraph to generate graph data [here](https://drive.google.com/file/d/1rih15bSlXTHZg-JNdtxSoS3uiVfjjIyX/view?usp=sharing)    |
| LKPNR | LKPNR: Large Language Models and Knowledge Graph for Personalized News Recommendation Framework | 2024 | LLM-based | models/LKPNR.py models/modules/LKPNR general_runner.py | config/lkpnr.yaml | Ported from the [official implementation](https://github.com/Xuan-ZW/LKPNR). EB-NeRD and Adressa use native semantic and linked-metadata features; the original KGraph_LKPNR files remain optional for legacy MIND experiments. |
| ONCE | ONCE: Boosting Content-based Recommendation with Both Open- and Closed-source Large Language Models | 2024 | LLM-based | models/ONCE_DIRE_QWEN3_NAML.py general_runner.py | config/once.yaml | Qwen3 adaptation of ONCE-DIRE with NAML-style user modeling. Builds a cached lower-layer hidden-state store and fine-tunes the final Qwen3 block with LoRA. |
| S2LENR | Towards S2-Challenges Underlying LLM-Based Augmentation for Personalized News Recommendation | 2025 | LLM-based | models/S2LENR.py scripts/generate_s2lenr_qwen_news.py general_runner.py | config/s2lenr.yaml | Structure-aware and semantic-aware LLM augmentation. Generate the JSONL cache locally with the model configured by `s2lenr_llm_model` before training for the full model. |
| PNR-LLM | Enhancing News Recommendation with Hierarchical LLM Prompting | 2025 | LLM-based | models/PNR_LLM.py scripts/generate_pnr_llm_enrichment.py general_runner.py | config/pnr-llm.yaml | Hierarchical title and entity enrichment followed by CNN, entity self-attention, multi-view attention, and attentive user pooling. |


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

To reproduce the paper's RoBERTa-NRMS PLM-NR model on MIND-small (the
configuration uses the paper's `1e-5` Adam learning rate and per-GPU batch size
of 32):

```bash
python general_runner.py --dataset_name=MIND --DATASET_ROOT=MIND-small --dataset_size=small --model=PLM-NR --word_embedding_dim=300 --batch_size=32
```

The first run downloads `roberta-base` and creates a cached mapping from the
corpus vocabulary to RoBERTa subword embeddings under `cache/nrms_plm/`.

To train SEIN on MIND, select the prepared MIND split and its 300-dimensional GloVe embeddings:

```bash
python general_runner.py --dataset_name=MIND --DATASET_ROOT=MIND-small --dataset_size=small --model=SEIN --batch_size=64 --epoch=10 --word_embedding_dim=300
```

ONCE-DIRE uses the public `Qwen/Qwen3-0.6B` safetensors checkpoint by default.
Set `ONCE_QWEN_MODEL` to use an already downloaded Qwen3 directory; also set
`once_local_files_only: true` in `config/once.yaml` for a fully offline run.
The first run automatically creates the lower-layer cache under
`cache/once_dire_qwen3_naml/`; previous LLaMA caches are not reused.

```bash
python general_runner.py --dataset_name=MIND --DATASET_ROOT=MIND-small --dataset_size=small --model=ONCE --batch_size=64 --epoch=10
```

To train S2LENR, first generate the user-level synthetic-news cache locally with the same model named by `s2lenr_llm_model` (`Qwen/Qwen3-0.6B` by default), then launch training. The generator uses Hugging Face `transformers` and disables Qwen3 thinking mode by default for efficient instruction-style generation:

```bash
python scripts/generate_s2lenr_qwen_news.py --dataset_name=MIND --DATASET_ROOT=MIND-small --model_name_or_path=Qwen/Qwen3-0.6B
python general_runner.py --dataset_name=MIND --DATASET_ROOT=MIND-small --dataset_size=small --model=S2LENR --batch_size=64 --epoch=10
```

For EB-NeRD, use the EB-NeRD root and embedding dimension used by the existing EB-NeRD configs:

```bash
python scripts/generate_s2lenr_qwen_news.py --dataset_name=ebnerd --DATASET_ROOT=ebnerd_demo --model_name_or_path=Qwen/Qwen3-0.6B
python general_runner.py --dataset_name=ebnerd --DATASET_ROOT=ebnerd_demo --dataset_size=demo --model=S2LENR --batch_size=64 --epoch=10 --word_embedding_dim=1024
```

PNR-LLM requires an offline per-article enrichment cache. NewsTorch uses local `Qwen/Qwen3-0.6B` generation instead of the paper's Gemini API, and writes to the cache path resolved by `config/pnr-llm.yaml`:

```bash
python scripts/generate_pnr_llm_enrichment.py --dataset_name=MIND --DATASET_ROOT=MIND-small --model_name_or_path=Qwen/Qwen3-0.6B
python general_runner.py --dataset_name=MIND --DATASET_ROOT=MIND-small --dataset_size=small --model=PNR-LLM --batch_size=64 --epoch=10 --word_embedding_dim=300
```

Generated entities can optionally be canonicalized through Wikidata before training:

```bash
python scripts/generate_pnr_llm_enrichment.py --dataset_name=MIND --DATASET_ROOT=MIND-small --model_name_or_path=Qwen/Qwen3-0.6B --verify_wikidata
```

Set `pnr_llm_require_enrichment_cache: true` for benchmark runs so an incomplete preprocessing setup fails immediately instead of using the title-only ablation.

### Adressa-1week

The official Adressa one-week light release is licensed CC BY-NC-SA 4.0 for non-commercial use. Review the terms, then download, extract, and prepare it in one command:

```bash
python dataset_download_prepare/download_adressa.py --accept-license --prepare
```

The downloader streams the official `one_week.tar.gz` archive into `Adressa-1week/download/`, resumes an existing `.part` file, and safely extracts the daily files into `Adressa-1week/raw/`. Use `--url` for an approved mirror, `--sha256` when a checksum is available, or `--force-download`/`--force-extract` to replace cached artifacts.

To prepare raw files that were obtained separately:

```bash
python dataset_download_prepare/Adressa_dataset_prepare.py --dataset-root Adressa-1week
```

Adressa contains positive click events rather than displayed non-click impressions. The preparer therefore uses days 1-5 as history, day 6 for training, and chronological halves of day 7 for development and testing. It deterministically samples 20 negatives per click by default and records the complete policy in `Adressa-1week/manifest.json`.

Run one model with the unified runner:

```bash
python general_runner.py --dataset_name=Adressa --DATASET_ROOT=Adressa-1week --dataset_size=1week --model=NRMS --word_embedding_dim=300
```

### LKPNR on EB-NeRD small and Adressa-1week

LKPNR requires a CUDA-capable GPU when launched through `general_runner.py`.
Before starting, ensure that the selected dataset has prepared `train`, `dev`,
and `test` directories. For Adressa, use the preparation commands above.

The full LKPNR semantic branch uses one frozen LLM vector per article. The
generator reproduces the official last-hidden-layer, attention-mask mean
pooling. It loads only `safetensors` weights, so it remains safe and usable with
PyTorch versions older than 2.6. `Qwen/Qwen3-0.6B` is the default compatible
encoder. To download it from Hugging Face while generating the cache, run:

```bash
# EB-NeRD small
python scripts/generate_lkpnr_embeddings.py \
  --dataset_name=ebnerd \
  --DATASET_ROOT=ebnerd_small \
  --dataset_size=small \
  --model_name_or_path=Qwen/Qwen3-0.6B \
  --batch_size=4 \
  --device=cuda

# Adressa-1week
python scripts/generate_lkpnr_embeddings.py \
  --dataset_name=Adressa \
  --DATASET_ROOT=Adressa-1week \
  --dataset_size=1week \
  --model_name_or_path=Qwen/Qwen3-0.6B \
  --batch_size=4 \
  --device=cuda
```

For an already downloaded checkpoint, replace the model name with its local
directory and add `--local_files_only`. That directory must contain
`model.safetensors` or sharded `*.safetensors` files.

The official LKPNR code used ChatGLM2-6B, whose official checkpoint contains
legacy `.bin` shards. Recent Transformers versions refuse to load those shards
with PyTorch older than 2.6 because the load path is affected by
CVE-2025-32434. Do not bypass that check. For an exact ChatGLM2 semantic cache,
convert the checkpoint with `safe_serialization=True` in a separate environment
that has PyTorch 2.6 or newer, then point this generator at the converted
directory. The resulting `.npy` cache can be used for LKPNR training in the
original NewsTorch environment.

These commands create:

```text
cache/ebnerd/lkpnr_item_embedding-small.npy
cache/adressa/lkpnr_item_embedding-1week.npy
```

If the cache is absent, LKPNR remains runnable and pools the dataset's
pretrained title and abstract word embeddings as a semantic fallback. For a
benchmark run that must use LLM vectors, set the following in
`config/lkpnr.yaml`:

```yaml
lkpnr_require_semantic_cache: true
```

Train on EB-NeRD small:

```bash
python general_runner.py \
  --model=LKPNR \
  --dataset_name=ebnerd \
  --DATASET_ROOT=ebnerd_small \
  --dataset_size=small \
  --word_embedding_dim=1024 \
  --batch_size=64 \
  --epoch=10
```

Train on Adressa-1week:

```bash
python general_runner.py \
  --model=LKPNR \
  --dataset_name=Adressa \
  --DATASET_ROOT=Adressa-1week \
  --dataset_size=1week \
  --word_embedding_dim=300 \
  --batch_size=64 \
  --epoch=10
```

Reduce the batch size to `16` or `32` if LKPNR does not fit in GPU memory.
Training evaluates the best checkpoint automatically. Checkpoints are stored
under `cache/best_models/<DATASET_ROOT>/LKPNR/`; for example:

```text
cache/best_models/ebnerd_small/LKPNR/#1/LKPNR
cache/best_models/Adressa-1week/LKPNR/#1/LKPNR
```

Evaluate an existing EB-NeRD checkpoint:

```bash
python general_runner.py \
  --mode=test \
  --model=LKPNR \
  --dataset_name=ebnerd \
  --DATASET_ROOT=ebnerd_small \
  --dataset_size=small \
  --word_embedding_dim=1024 \
  --test_model_path='cache/best_models/ebnerd_small/LKPNR/#1/LKPNR'
```

Evaluate an existing Adressa checkpoint:

```bash
python general_runner.py \
  --mode=test \
  --model=LKPNR \
  --dataset_name=Adressa \
  --DATASET_ROOT=Adressa-1week \
  --dataset_size=1week \
  --word_embedding_dim=300 \
  --test_model_path='cache/best_models/Adressa-1week/LKPNR/#1/LKPNR'
```

Prepare optional local Qwen3 caches for the full S2LENR and PNR-LLM paths:

```bash
python scripts/generate_s2lenr_qwen_news.py --dataset_name=Adressa --DATASET_ROOT=Adressa-1week --model_name_or_path=Qwen/Qwen3-0.6B
python scripts/generate_pnr_llm_enrichment.py --dataset_name=Adressa --DATASET_ROOT=Adressa-1week --model_name_or_path=Qwen/Qwen3-0.6B
```

Deploy every model registered by `general_runner.py` sequentially, preserving completed checkpoints on a resumed run:

```bash
python scripts/deploy_all_adressa.py --dataset-root Adressa-1week --resume --continue-on-error
```

The common Adressa adapter derives text, category, named-entity, read-time, popularity, and concept fields from the click log. Since the release has no article images or calibrated sentiment labels, MMRec receives zero image features and sentiment models receive neutral labels. LKPNR derives its semantic and knowledge views from the available Adressa article fields instead of using MIND-specific knowledge files. ONCE uses the Qwen3 checkpoint configured in `config/once.yaml`.

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
- **Hit Rate@5/10**: Whether a relevant item appears in the top K
- **Precision@5/10**: Precision at K
- **Categ-Div@5/10**: Normalized entropy of recommended-news categories (subcategories for Adressa-1week)
- **Categ-Pers@5/10**: Generalized Jaccard similarity between recommended and clicked-history category counts (subcategory counts for Adressa-1week)
- **MAE**: Mean Absolute Error
- **RMSE**: Root Mean Square Error

The category diversity and personalization metrics follow the aspect-based
definitions in [NewsRecLib](https://github.com/andreeaiana/newsreclib). Results
are automatically saved to CSV files for further analysis. Because the
Adressa-1week adapter does not expose informative top-level categories, its
`Categ-*` metrics explicitly use the corpus subcategory taxonomy instead.


## Limitations

NewsTorch is still under active development. The current version has the following limitations.

### LLM Reproducibility

LLM-assisted news recommendation is difficult to reproduce exactly, especially when API-based LLMs are used. API-based models may change over time due to backend updates, decoding randomness, provider-specific changes, rate limits, pricing changes, and access availability.

To improve reproducibility, we recommend recording the LLM provider, model name, prompt template, decoding parameters, and API/local checkpoint version. Users are also encouraged to cache generated responses, embeddings, or user/news profiles. However, fully deterministic reproduction of API-based LLM results cannot be guaranteed.

### Dataset Coverage

The current validation mainly focuses on MIND, EB-NeRD, and the Adressa-1week adapter. These datasets do not cover all possible data schemas, languages, metadata fields, behavior logs, licensing requirements, or temporal evaluation protocols.

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

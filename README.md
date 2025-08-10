# NewsRecTorch

NewsRecTorch is a PyTorch-based library for news recommendation. It provides implementations of various neural news recommendation models, including both ML-based and DL-based methods, as well as graph-based models. 
The framework is designed to facilitate the development and evaluation of news recommendation systems.
It is built on top of PyTorch and provides a unified interface for training and evaluating different models. The library also includes utilities for data preprocessing, feature extraction, and evaluation metrics.
We also provide a general runner for most models, which allows users to easily switch between different models and datasets.
For special models, such as UNBERT, it provides a different interface for training and evaluation.
Our motivation is to provide a comprehensive and easy-to-use framework for news recommendation research, enabling researchers and practitioners to quickly implement and evaluate different models with an original PyTorch environment.


This library includes implementations of the following models:

| Model | Paper | Year |
|-------|-------|------|
| NPA | [NPA](https://arxiv.org/abs/1704.00270) | 2019 |
| DKN | [DKN](https://arxiv.org/abs/1709.02320) | 2018 |
| LSTUR | [LSTUR](https://arxiv.org/abs/1802.03684) | 2019 |
| NAML | [NAML](https://arxiv.org/abs/1802.03401) | 2019 |
| NRMS | [NRMS](https://arxiv.org/abs/1808.09781) | 2019 |
| FIM | [FIM](https://arxiv.org/abs/1904.06690) | 2020 |
| TANR | [TANR](https://arxiv.org/abs/1909.05883) | 2019 |
| CenNewsRec | [CenNewsRec](https://arxiv.org/abs/2004.01463) | 2020 |
| MINS | [MINS](https://arxiv.org/abs/2005.00796) | 2022 |
| UNBERT | [UNBERT](https://arxiv.org/abs/2010.06467) | 2021 |
| CNE-SUE | [CNE-SUE](https://arxiv.org/abs/2101.08604) | 2021 |
| IPNR | [IPNR](https://arxiv.org/abs/2105.01524) | 2023 |
| MANNeR | [MANNeR](https://arxiv.org/abs/2203.12664) | 2022 |
| LKPNR | [LKPNR](https://arxiv.org/abs/2204.08666) | 2024 |
| ONCE | [ONCE](https://arxiv.org/abs/2205.15263) | 2024 |
| Prompt4NR | [Prompt4NR](https://arxiv.org/abs/2303.11364) | 2023 |


 

## Environment Requirements

You can create a conda environment using the following command:
<pre><code>
conda create -n NewsRecTorch python=3.8
conda activate NewsRecTorch
./intall_packages.sh
</code></pre>

## Dataset Preparation

The dataset preparation is divided into two steps: downloading the dataset and preparing the dataset.


## How to Run

Neural news recommendation baselines:

- DL-based models: NPA, DKN, LSTUR, NAML, NRMS, FIM, TANR, CenNewsRec, MINS, UNBERT

- Graph-based models: CNE-SUE, IPNR


You can run the DL-based models like NRMS in the following order:
<pre><code>python general_runner.py --model=NRMS</code></pre>


## To Do
1. Add more datasets: Adressa
2. Add more models: MANNeR, LKPNR, ONCE



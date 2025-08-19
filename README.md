# NewsTorch

NewsTorch is a PyTorch-based library for news recommendation. It provides implementations of various neural news recommendation models, including both ML-based and DL-based methods, as well as graph-based models. 
The framework is designed to facilitate the development and evaluation of news recommendation systems.
It is built on top of PyTorch and provides a unified interface for training and evaluating different models. The library also includes utilities for data preprocessing, feature extraction, and evaluation metrics.
We also provide a general runner for most models, which allows users to easily switch between different models and datasets.
Our motivation is to provide a comprehensive and easy-to-use framework for news recommendation research, enabling researchers and practitioners to quickly implement and evaluate different models with an original PyTorch environment.


This library includes implementations of the following models:

| Model | Paper                                                                                                                         | Year |
|-------|-------------------------------------------------------------------------------------------------------------------------------|------|
| NPA | NPA: neural news recommendation with personalized attention                                                                   | 2019 |
| DKN | DKN: Deep knowledge-aware network for news recommendation                                                                     | 2018 |
| LSTUR | Neural news recommendation with long-and short-term user representations                                                      | 2019 |
| NAML | Neural News Recommendation with Attentive Multi-View Learning                                                                 | 2019 |
| NRMS | Neural news recommendation with multi-head self-attention                                                                     | 2019 |
| FIM | Fine-grained Interest Matching for Neural News Recommendation                                                                 | 2020 |
| TANR | Neural news recommendation with topic-aware news representation                                                               | 2019 |
| CenNewsRec | Privacy-Preserving News Recommendation Model Learning                                          | 2020 |
| MINS | News recommendation via multi-interest news sequence modelling                               | 2022 |
| CNE-SUE | Neural News Recommendation with Collaborative News Encoding and Structural User Encoding       | 2021 |
| IPNR | Intention-aware user modeling for personalized news recommendation                           | 2023 |
| MANNeR | Train once, use flexibly: A modular framework for multi-aspect neural news recommendation    | 2024 |
| LKPNR | LKPNR: Large Language Models and Knowledge Graph for Personalized News Recommendation Framework | 2024 |
| ONCE | Once: Boosting content-based recommendation with both open-and closed-source large language models | 2024 |



 

## Environment Requirements

You can create a conda environment using the following command:
<pre><code>
conda create -n NewsRecTorch python=3.9
conda activate NewsRecTorch
./intall_packages.sh
</code></pre>

## Dataset Preparation

The dataset preparation is divided into two steps: downloading the dataset and preparing the dataset.


## How to Run

Neural news recommendation baselines:

- DL-based models: NPA, DKN, LSTUR, NAML, NRMS, FIM, TANR, CenNewsRec, MINS, MANNeR

- Graph-based models: CNE-SUE, IPNR

- LLM-based models: LKPNR, ONCE


You can run the DL-based models like NRMS in the following order:
<pre><code>python general_runner.py --model=NRMS</code></pre>


## To Do
1. Add more datasets: Adressa;
2. Add more models: MANNeR, ONCE
3. Adapt LKPNR to MIND-large; 



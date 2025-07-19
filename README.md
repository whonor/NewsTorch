# NewsRecTorch


## Dataset Preparation

The dataset preparation is divided into two steps: downloading the dataset and preparing the dataset. 

## Environment Requirements
Dependencies are needed to be installed by
<pre><code>bash install_dependencies.sh</code></pre>
Our experiments require python>=3.7, torch==2.1.0, and torch_scatter. The [torch_scatter](https://github.com/rusty1s/pytorch_scatter) package is necessary, installing as follows:

<pre><code>pip install --no-index torch-scatter -f https://pytorch-geometric.com/whl/torch-2.1.0+cu121.html/code></pre>

## How to Run

Neural news recommendation baselines:

- ML-based models: LibFM, DSSM, Wide&Deep

- DL-based models: NPA, DKN, LSTUR, NAML, NRMS, FIM, TANR, CenNewsRec, MINS, UNBERT

- Graph-based models: CNE-SUE, IPNR, DIAT

<pre><code>cd general_recommendation_methods
python generate_tf_idf_feature_file.py
python generate_libfm_data.py
chmod -R 777 libfm
python libfm_main.py
python DSSM_main.py 
python wide_deep_main.py</code></pre>


## To Do
1. Add more datasets: Adressa, EB-NeRD
2. Add more models: MANNeR, LKPNR, ONCE, Prompt4NR
3. Add more evaluation metrics using TorchMetrics.


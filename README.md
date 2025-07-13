

## Dataset Preparation

The dataset preparation is divided into two steps: downloading the dataset and preparing the dataset. 

## Environment Requirements
Dependencies are needed to be installed by
<pre><code>bash install_dependencies.sh</code></pre>
Our experiments require python>=3.7, torch==1.12.1, and torch_scatter==2.0.9. The [torch_scatter](https://github.com/rusty1s/pytorch_scatter) package is necessary. If the dependency installation fails, please follow [https://github.com/rusty1s/pytorch_scatter](https://github.com/rusty1s/pytorch_scatter) to install the package manually.
<br/><br/>

## Experiment Running

<hr>Neural news recommendation baselines:

- ML-based models: LibFM, DSSM, Wide&Deep

- DL-based models: NPA, DKN, LSTUR, NAML, NRMS, FIM, 

- Graph-based models: CNE-SUE, DIAT, IPNR

<pre><code>python main.py --news_encoder=DAE       --user_encoder=GRU
python main.py --news_encoder=KCNN      --user_encoder=CATT  --word_embedding_dim=100 --entity_embedding_dim=100   --context_embedding_dim=100
python main.py --news_encoder=CNN       --user_encoder=LSTUR
python main.py --news_encoder=NAML      --user_encoder=ATT
python main.py --news_encoder=PNE       --user_encoder=PUE
python main.py --news_encoder=MHSA      --user_encoder=MHSA
python main.py --news_encoder=HDC       --user_encoder=FIM   --click_predictor=FIM</code></pre>

<hr>General news recommendation baselines in Section 4.2
<pre><code>cd general_recommendation_methods
python generate_tf_idf_feature_file.py
python generate_libfm_data.py
chmod -R 777 libfm
python libfm_main.py
python DSSM_main.py 
python wide_deep_main.py</code></pre>


## Distributed Training & Faster Inference
Distributed training is supported. If you would like to train models on N GPUs, please set the config parameter `--world_size=N`. The batch size config parameter `batch_size` should be divisible by `world_size`, as our code equally divides the training batch size into N GPUs. For example,
<pre><code>python main.py --news_encoder=CNE --user_encoder=SUE --batch_size=128 --world_size=4</code></pre>
The command above trains our model on 4 GPUs, each GPU contains the mini-batch data of 32.

For coding simplicity, we do not implement news representation caching in the inference stage. Pre-computing and caching news representation can magnificently accelerate inference. For code of news representation caching, please refer to [https://github.com/Veason-silverbullet/DIGAT/blob/master/util.py](https://github.com/Veason-silverbullet/DIGAT/blob/master/util.py).

## To Do
1. Add more datasets: Adressa, EB-NeRD
2. Add more models: CAUM, MINER, MINS, TANR, MANNeR, CenNewsRec, UNBERT
3. Add more evaluation metrics using TorchMetrics.


#!/bin/bash
# This script install s the required packages for the project. python=3.9
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
pip install torchtext==0.16.0 torchmetrics==1.5.2 transformers==4.46.3 wandb==0.21.0 scikit-learn==1.0.2 pandas==2.0.3 scipy==1.10.1 openai==1.88.0 nltk==3.7 networkx==3.1 retrying==1.4.1 polars matplotlib flask flask-wtf sentencepiece
pip install --no-index torch-sparse -f https://pytorch-geometric.com/whl/torch-2.1.0+cu121.html
pip install --no-index torch-scatter -f https://pytorch-geometric.com/whl/torch-2.1.2+cu121.html
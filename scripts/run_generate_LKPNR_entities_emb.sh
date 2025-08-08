#!/bin/bash
# Prepare the data for LKPNR entities embeddings
#conda init
#conda activate NewsRecTorch

# python ./LKPNR/get_item.py

python ./LKPNR/count_link_counts.py --dataset_path="../../MIND-small"

python ./LKPNR/get_node_emb.py --train_news_path='../../MIND-200k/train/news.tsv' --train_behaviors_path='../../MIND-small/train/behaviors.tsv' --dev_behaviors_path='../../MIND-small/dev/behaviors.tsv' --test_behaviors_path='../../MIND-small/test/behaviors.tsv' --train_entity_path="../../MIND-200k/train/entity_embedding.vec" --dev_entity_path="../../MIND-200k/dev/entity_embedding.vec"

python ./LKPNR/get_all_entities_emb_dict.py --train_entity_path="../../MIND-200k/train/entity_embedding.vec" --dev_entity_path="../../MIND-200k/dev/entity_embedding.vec" --output_path="all_entity_emb_dic.pkl"


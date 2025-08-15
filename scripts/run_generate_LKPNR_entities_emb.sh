#!/bin/bash
# Prepare the data for LKPNR entities embeddings
#conda init
#conda activate NewsRecTorch

# python ./LKPNR/get_item.py

python ./LKPNR/count_link_count.py --dataset_path="../../MIND-large"\
--save_path='KGraph_LKPNR'

python ./LKPNR/get_node_emb.py --train_news_path='../../MIND-large/train/news.tsv' \
--train_behaviors_path='../../MIND-large/train/behaviors.tsv' \
--dev_behaviors_path='../../MIND-large/dev/behaviors.tsv' \
--test_behaviors_path='../../MIND-large/test/behaviors.tsv' \
--train_entity_path="../../MIND-large/train/entity_embedding.vec" --dev_entity_path="../../MIND-large/dev/entity_embedding.vec"

python ./LKPNR/get_all_entities_emb_dict.py --train_entity_path="../../MIND-large/train/entity_embedding.vec" \
--dev_entity_path="../../MIND-large/dev/entity_embedding.vec" \
--output_path="../KGraph_LKPNR/all_entity_emb_dic.pkl"


#!/bin/bash
# Prepare the data for LKPNR entities embeddings
#conda init
#conda activate NewsRecTorch

python ./LKPNR/get_item.py

python ./LKPNR/count_link_counts.py

python ./LKPNR/get_node_emb.py

python ./LKPNR/get_all_entities_emb_dict.py


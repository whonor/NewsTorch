#!/bin/bash
# Prepare the graph data for IPNR after download_mind and MIND_dataset_prepare
#conda init
#conda activate NewsTorch

# python ../Conceptgraph/conceptnet/_extract_english_cpnet.py --conceptnet_path="../Conceptgraph/conceptnet/conceptnet-assertions-5.7.0.csv/assertions.csv" --output_csv_path='../Conceptgraph/conceptnet/conceptnet.en.csv'

python ../Conceptgraph/graph/_build_vocab_and_graph.py --root_path="../MIND-small/download/"

python ../Conceptgraph/graph/_build_edge_index.py --mode="train" --root_path="../MIND-small/download/"
python ../Conceptgraph/graph/_build_edge_index.py --mode="dev" --root_path="../MIND-small/download/"

python ../Conceptgraph/graph/_generate_concept_embeddings.py --root_path="../MIND-small/download/"

python ../Conceptgraph/TF-IDF/_TFIDF_main.py --mode="train" --dst_graph_path="../MIND-small/train/concepts_subgraph.pkl" --root_path="../MIND-small/download/"
python ../Conceptgraph/TF-IDF/_TFIDF_main.py --mode="dev" --dst_graph_path="../MIND-small/test/concepts_subgraph.pkl" --root_path="../MIND-small/download/"

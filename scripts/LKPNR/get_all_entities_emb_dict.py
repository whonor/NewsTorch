import argparse

import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--train_entity_path', type=str, default='../../MIND-200k/train/entity_embedding.vec', help='dataset path')
parser.add_argument('--dev_entity_path', type=str, default='../../MIND-200k/dev/entity_embedding.vec', help='dataset path')
parser.add_argument('--output_path', type=str, default='all_entity_emb_dic.pkl', help='dataset path')

args, _ = parser.parse_known_args()

train_entity_path = args.train_entity_path
dev_entity_path = args.dev_entity_path
output_path = args.output_path

with open(train_entity_path, "r", encoding="utf-8") as file:
    lines = file.readlines()

data = []
for line in lines:
    parts = line.strip().split()
    key = parts[0]
    value = [float(v) for v in parts[1:]]
    data.append([key, value])

node_emb_train = pd.DataFrame(data, columns=["Node", "Embedding"])

with open(dev_entity_path, "r", encoding="utf-8") as file:
    lines = file.readlines()

data = []
for line in lines:
    parts = line.strip().split()
    key = parts[0]
    value = [float(v) for v in parts[1:]]
    data.append([key, value])

node_emb_dev = pd.DataFrame(data, columns=["Node", "Embedding"])

node_emb = pd.concat([node_emb_train, node_emb_dev])
node_emb.drop_duplicates(subset="Node", keep="first", inplace=True)
node_emb.reset_index(drop=True, inplace=True)

my_dict = node_emb.set_index('Node')['Embedding'].to_dict()

import pickle

with open(output_path, 'wb') as file:
    pickle.dump(my_dict, file)

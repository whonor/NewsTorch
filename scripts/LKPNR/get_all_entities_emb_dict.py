
import pandas as pd

with open("../MIND-200k/train/entity_embedding.vec", "r", encoding="utf-8") as file:
    lines = file.readlines()

data = []
for line in lines:
    parts = line.strip().split()
    key = parts[0]
    value = [float(v) for v in parts[1:]]
    data.append([key, value])

node_emb_train = pd.DataFrame(data, columns=["Node", "Embedding"])

with open("../MIND-200k/dev/entity_embedding.vec", "r", encoding="utf-8") as file:
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

with open('all_entity_emb_dic.pkl', 'wb') as file:
    pickle.dump(my_dict, file)

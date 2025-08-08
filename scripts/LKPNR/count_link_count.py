import argparse

import pandas as pd
import pickle

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_path', type=str, default='../../MIND-small', help='dataset path')
args, _ = parser.parse_known_args()
dataset_path = args.dataset_path

linked_node_dic_df = pd.read_csv('linked_node_dic_df.csv', sep='\t')
# dic指明实体有多少相邻节点
entity_link_count = linked_node_dic_df['original_entity'].value_counts().to_dict()
# 相邻节点字典
linke_dic = linked_node_dic_df.groupby('original_entity')['linked_entities'].agg(list).to_dict()

with open('./entity_in_emb_file.pickle', 'rb') as file:
    entity_in_emb_file = pickle.load(file)

# 计算 训练集各新闻包含 entity数
with open(dataset_path + '/train/news.tsv', "r", encoding="utf-8") as file:
    lines = file.readlines()
data = []
for line in lines:
    parts = line.strip().split('\t')
    news_id = parts[0]
    title_entity = parts[-2]
    abstract_entity = parts[-1]
    # 获得新闻中的实体
    title_entity_id = [item["WikidataId"] for item in eval(title_entity) ]
    abstract_entity_id = [item["WikidataId"] for item in eval(abstract_entity) ]
    news_entity_id  = title_entity_id + abstract_entity_id
    news_entity_id_count = len(news_entity_id)

    link_entity = set(news_entity_id)
    # source entity也不一定有emb，也要筛选一下
    for source_entity in news_entity_id:
        if source_entity in linke_dic:
            link_entity = link_entity.union(linke_dic[source_entity])
    # 确保有emb
    link_entity = link_entity.intersection(entity_in_emb_file)
    link_entity = list(link_entity)
    link_entity_count = len(link_entity)

    data.append([news_id, news_entity_id, news_entity_id_count, link_entity, link_entity_count])
    
linked_relation_in_train_news = pd.DataFrame(data, columns=["news_id", "news_entity_id", "news_entity_id_count", 'link_entity_id', 'link_entity_count'])

# 计算 dev集各新闻包含 entity数
with open(dataset_path + '/dev/news.tsv', "r", encoding="utf-8") as file:
    lines = file.readlines()
data = []
for line in lines:
    parts = line.strip().split('\t')
    news_id = parts[0]
    title_entity = parts[-2]
    abstract_entity = parts[-1]
    # 获得新闻中的实体
    title_entity_id = [item["WikidataId"] for item in eval(title_entity) ]
    abstract_entity_id = [item["WikidataId"] for item in eval(abstract_entity) ]
    news_entity_id  = title_entity_id + abstract_entity_id
    news_entity_id_count = len(news_entity_id)

    link_entity = set(news_entity_id)
    for source_entity in news_entity_id:
        if source_entity in linke_dic:
            link_entity = link_entity.union(linke_dic[source_entity])
    # 确保有emb
    link_entity = link_entity.intersection(entity_in_emb_file)        
    link_entity = list(link_entity)
    link_entity_count = len(link_entity)

    data.append([news_id, news_entity_id, news_entity_id_count, link_entity, link_entity_count])
    
linked_relation_in_dev_news = pd.DataFrame(data, columns=["news_id", "news_entity_id", "news_entity_id_count", 'link_entity_id', 'link_entity_count'])

# 计算 dev集各新闻包含 entity数
with open(dataset_path + '/test/news.tsv', "r", encoding="utf-8") as file:
    lines = file.readlines()
data = []
for line in lines:
    parts = line.strip().split('\t')
    news_id = parts[0]
    title_entity = parts[-2]
    abstract_entity = parts[-1]
    # 获得新闻中的实体
    title_entity_id = [item["WikidataId"] for item in eval(title_entity) ]
    abstract_entity_id = [item["WikidataId"] for item in eval(abstract_entity) ]
    news_entity_id  = title_entity_id + abstract_entity_id
    news_entity_id_count = len(news_entity_id)

    link_entity = set(news_entity_id)
    for source_entity in news_entity_id:
        if source_entity in linke_dic:
            link_entity = link_entity.union(linke_dic[source_entity])
    # 确保有emb
    link_entity = link_entity.intersection(entity_in_emb_file)    
    link_entity = list(link_entity)
    link_entity_count = len(link_entity)

    data.append([news_id, news_entity_id, news_entity_id_count, link_entity, link_entity_count])
    
linked_relation_in_test_news = pd.DataFrame(data, columns=["news_id", "news_entity_id", "news_entity_id_count", 'link_entity_id', 'link_entity_count'])

result_df = pd.concat([linked_relation_in_train_news, linked_relation_in_dev_news, linked_relation_in_test_news], ignore_index=True)

result_df = result_df.drop_duplicates(subset='news_id', keep='first')

link_entuty_dic = result_df.set_index('news_id')['link_entity_id'].to_dict()


with open('link_entuty_dic.pkl', 'wb') as file:
    pickle.dump(link_entuty_dic, file)

with open('link_entuty_dic.pkl', 'rb') as file:
    my_dict = pickle.load(file)

with open('./link_entity_dic.pkl', 'rb') as file:
    link_entity_dic = pickle.load(file)

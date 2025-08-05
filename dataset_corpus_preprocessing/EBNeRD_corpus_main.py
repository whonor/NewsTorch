import os
import json
import pickle
import collections
import re

import fasttext
import pandas as pd
from gensim.models import KeyedVectors
from nltk.tokenize import word_tokenize
from torchtext.vocab import GloVe
from config import Config
import torch
import numpy as np


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

pat = re.compile(r"[\w]+|[.,!?;|]")

import urllib.request
import gzip
import shutil

# Step 1: Auto-download and extract FastText Danish vector
def download_fasttext_vec_if_needed(vec_file_path):
    if os.path.exists(vec_file_path):
        print(f"✅ Vector file already exists at: {vec_file_path}")
        return

    vec_gz_url = "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.da.300.vec.gz"
    vec_gz_path = vec_file_path + ".gz"

    print(f"⬇️ Downloading FastText Danish vectors from {vec_gz_url} ...")
    urllib.request.urlretrieve(vec_gz_url, vec_gz_path)

    print(f"📦 Extracting {vec_gz_path} ...")
    with gzip.open(vec_gz_path, 'rb') as f_in:
        with open(vec_file_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

    os.remove(vec_gz_path)
    print(f"✅ Extracted to {vec_file_path}")

# Step 2: Load FastText .vec file into dictionary
def load_fasttext_vec(filepath):
    word_to_vec = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        first_line = f.readline()  # Skip header
        for line in f:
            parts = line.rstrip().split(' ')
            word = parts[0]
            vec = torch.tensor([float(x) for x in parts[1:]], dtype=torch.float32)
            word_to_vec[word] = vec
    return word_to_vec

# Step 3: Build embedding matrix from word_dict
def build_danish_word_embedding(word_dict, vec_file, embedding_dim, output_pkl_path):
    # Auto-download if file missing
    download_fasttext_vec_if_needed(vec_file)

    print("📥 Loading FastText Danish vectors...")
    word_to_vec = load_fasttext_vec(vec_file)

    print("🛠️ Building word embedding matrix...")
    all_vecs = torch.stack(list(word_to_vec.values()))
    mean_vector = torch.mean(all_vecs, dim=0)

    word_embedding_vectors = torch.zeros([len(word_dict), embedding_dim])
    for word, index in word_dict.items():
        if index == 0:
            continue
        if word in word_to_vec:
            word_embedding_vectors[index, :] = word_to_vec[word]
        else:
            random_vec = torch.randn(embedding_dim) * 0.1
            word_embedding_vectors[index, :] = random_vec + mean_vector

    with open(output_pkl_path, 'wb') as f:
        pickle.dump(word_embedding_vectors, f)

    print(f"✅ Saved to {output_pkl_path}, shape: {word_embedding_vectors.shape}")


class EBNeRD_Corpus:
    @staticmethod
    def preprocess(config: Config):
        user_ID_file = 'cache/ebnerd/user_ID-%s.json' % config.dataset
        news_ID_file = 'cache/ebnerd/news_ID-%s.json' % config.dataset
        category_file = 'cache/ebnerd/category-%s.json' % config.dataset
        subCategory_file = 'cache/ebnerd/subCategory-%s.json' % config.dataset
        vocabulary_file = 'cache/ebnerd/vocabulary-' + str(config.word_threshold) + '-' + config.tokenizer + '-' + str(config.max_title_length) + '-' + str(config.max_abstract_length) + '-' + config.dataset + '.json'
        word_embedding_file = 'cache/ebnerd/word_embedding-' + str(config.word_threshold) + '-' + str(config.word_embedding_dim) + '-' + config.tokenizer + '-' + str(config.max_title_length) + '-' + str(config.max_abstract_length) + '-' + config.dataset + '.pkl'
        entity_file = 'cache/ebnerd/entity-%s.json' % config.dataset
        entity_embedding_file = 'cache/ebnerd/entity_embedding-%s.pkl' % config.dataset
        context_embedding_file = 'cache/ebnerd/context_embedding-%s.pkl' % config.dataset
        user_history_graph_file = 'cache/ebnerd/user_history_graph-' + str(config.max_history_num) + ('' if config.no_self_connection else '-self') + ('' if config.no_adjacent_normalization else '-normalize-' + config.gcn_normalization_type) + '-' + config.dataset + '.pkl'
        preprocessed_data_files = [user_ID_file, news_ID_file, category_file, subCategory_file, vocabulary_file, word_embedding_file, entity_file, entity_embedding_file, context_embedding_file, user_history_graph_file]

        if not all(list(map(os.path.exists, preprocessed_data_files))):
            user_ID_dict = {'<UNK>': 0}
            news_ID_dict = {'<PAD>': 0}
            category_dict = {}
            subCategory_dict = {}
            word_dict = {'<PAD>': 0, '<UNK>': 1}
            word_counter = collections.Counter()
            entity_dict = {'<PAD>': 0, '<UNK>': 1}
            news_category_dict = {}

            # 1. user ID dictionay
            behaviors_df = pd.read_parquet(os.path.join(config.train_root, 'behaviors.parquet'))

            for user_id in behaviors_df['uid'].unique():
                user_id = str(user_id)
                if user_id not in user_ID_dict:
                    user_ID_dict[user_id] = len(user_ID_dict)
            with open(user_ID_file, 'w', encoding='utf-8') as user_ID_f:
                json.dump(user_ID_dict, user_ID_f)

            # 2. news ID dictionay & news category dictionay & news subCategory dictionay
            for i, prefix in enumerate([config.train_root, config.dev_root, config.test_root]):
                news_parquet_file = os.path.join(prefix, 'news.parquet')
                df_news = pd.read_parquet(news_parquet_file)

                for _, row in df_news.iterrows():
                    news_ID = str(row['nid']).strip()  # adapt column name if different
                    category = str(row['category']).strip()
                    subCategory = str(row['subcategory']).strip()
                    title = str(row['title']).lower()
                    abstract = str(row['abstract']).lower() if 'abstract' in row else ''

                    if news_ID not in news_ID_dict:
                        news_ID_dict[news_ID] = len(news_ID_dict)
                        if category not in category_dict:
                            category_dict[category] = len(category_dict)
                        if subCategory not in subCategory_dict:
                            subCategory_dict[subCategory] = len(subCategory_dict)

                        words = pat.findall(title) if config.tokenizer == 'MIND' else word_tokenize(title)
                        for word in words:
                            if is_number(word):
                                word_counter['<NUM>'] += 1
                            else:
                                if i == 0:  # training set
                                    word_counter[word] += 1
                                else:
                                    if word in word_counter:
                                        word_counter[word] += 1

                        if abstract:
                            words = pat.findall(abstract) if config.tokenizer == 'MIND' else word_tokenize(abstract)
                            for word in words:
                                if is_number(word):
                                    word_counter['<NUM>'] += 1
                                else:
                                    if i == 0:  # training set
                                        word_counter[word] += 1
                                    else:
                                        if word in word_counter:
                                            word_counter[word] += 1

                    news_category_dict[news_ID] = category_dict[category]

            # Save dictionaries
            with open(news_ID_file, 'w', encoding='utf-8') as news_ID_f:
                json.dump(news_ID_dict, news_ID_f)
            with open(category_file, 'w', encoding='utf-8') as category_f:
                json.dump(category_dict, category_f)
            with open(subCategory_file, 'w', encoding='utf-8') as subCategory_f:
                json.dump(subCategory_dict, subCategory_f)

            # 3. word dictionay
            word_counter_list = [[word, word_counter[word]] for word in word_counter]
            word_counter_list.sort(key=lambda x: x[1], reverse=True) # sort by word frequency
            filtered_word_counter_list = list(filter(lambda x: x[1] >= config.word_threshold, word_counter_list))
            for i, word in enumerate(filtered_word_counter_list):
                word_dict[word[0]] = i + 2
            with open(vocabulary_file, 'w', encoding='utf-8') as vocabulary_f:
                json.dump(word_dict, vocabulary_f)

            # 4. Danish word embedding using fastText
            build_danish_word_embedding(
                word_dict,
                vec_file='cc.da.300.vec',
                embedding_dim=300,
                output_pkl_path=word_embedding_file
            )

            # build graph
            # build graph
            category_num = len(category_dict)
            graph_size = config.max_history_num + category_num  # |V_n| + |V_p|
            prefix_mode = ['train', 'dev', 'test']
            user_history_graph_data = {}

            for prefix_index, prefix in enumerate([config.train_root, config.dev_root, config.test_root]):
                mode = prefix_mode[prefix_index]
                behaviors_parquet = os.path.join(prefix, 'behaviors.parquet')

                df_behaviors = pd.read_parquet(behaviors_parquet)
                user_history_num = len(df_behaviors)

                user_history_graph = np.zeros([user_history_num, graph_size, graph_size], dtype=np.float32)
                user_history_category_mask = np.zeros([user_history_num, category_num + 1], dtype=np.float32)
                user_history_category_indices = np.full([user_history_num, config.max_history_num], category_num,
                                                        dtype=np.int64)

                for line_index, row in df_behaviors.iterrows():
                    history = str(row['history']).strip()

                    if config.no_self_connection:
                        history_graph = np.zeros([graph_size, graph_size], dtype=np.float32)
                    else:
                        history_graph = np.identity(graph_size, dtype=np.float32)

                    history_category_mask = np.zeros(category_num + 1, dtype=np.float32)
                    history_category_indices = np.full(config.max_history_num, category_num, dtype=np.int64)

                    if len(history.strip('[] ')) > 0:
                        history_news_ID = history.strip('[] ').split(',')
                        history_news_ID = [nid.strip() for nid in history_news_ID if nid.strip() != '']

                        offset = max(0, len(history_news_ID) - config.max_history_num)
                        history_news_num = min(len(history_news_ID), config.max_history_num)

                        for i in range(history_news_num):
                            news_id = history_news_ID[i + offset]
                            category_index = news_category_dict.get(news_id, category_num)

                            if category_index >= category_num:
                                continue  # skip invalid category

                            history_category_mask[category_index] = 1.0
                            history_category_indices[i] = category_index

                            history_graph[i, config.max_history_num + category_index] = 1
                            history_graph[config.max_history_num + category_index, i] = 1

                            for j in range(i + 1, history_news_num):
                                other_news_id = history_news_ID[j + offset]
                                other_category_index = news_category_dict.get(other_news_id, category_num)

                                if other_category_index >= category_num:
                                    continue  # skip invalid category

                                if category_index == other_category_index:
                                    history_graph[i, j] = 1
                                    history_graph[j, i] = 1
                                else:
                                    history_graph[
                                        config.max_history_num + category_index, config.max_history_num + other_category_index] = 1
                                    history_graph[
                                        config.max_history_num + other_category_index, config.max_history_num + category_index] = 1

                        if not config.no_adjacent_normalization:
                            degrees = history_graph.sum(axis=1)
                            if config.gcn_normalization_type == 'asymmetric':
                                D_inv = np.diag(1 / np.clip(degrees, a_min=1e-12, a_max=None))
                                history_graph = D_inv @ history_graph
                            else:
                                D_inv_sqrt = np.diag(np.sqrt(1 / np.clip(degrees, a_min=1e-12, a_max=None)))
                                history_graph = D_inv_sqrt @ history_graph @ D_inv_sqrt

                    user_history_graph[line_index] = history_graph
                    user_history_category_mask[line_index] = history_category_mask
                    user_history_category_indices[line_index] = history_category_indices

                user_history_graph_data[f'{mode}_user_history_graph'] = user_history_graph
                user_history_graph_data[f'{mode}_user_history_category_mask'] = user_history_category_mask
                user_history_graph_data[f'{mode}_user_history_category_indices'] = user_history_category_indices

            with open(user_history_graph_file, 'wb') as f:
                pickle.dump(user_history_graph_data, f)

    def __init__(self, config: Config):
        # preprocess cache
        EBNeRD_Corpus.preprocess(config)
        with open('cache/ebnerd/user_ID-%s.json' % config.dataset, 'r', encoding='utf-8') as user_ID_f:
            self.user_ID_dict = json.load(user_ID_f)
            config.user_num = len(self.user_ID_dict)
        with open('cache/ebnerd/news_ID-%s.json' % config.dataset, 'r', encoding='utf-8') as news_ID_f:
            self.news_ID_dict = json.load(news_ID_f)
            self.news_num = len(self.news_ID_dict)
        with open('cache/ebnerd/category-%s.json' % config.dataset, 'r', encoding='utf-8') as category_f:
            self.category_dict = json.load(category_f)
            config.category_num = len(self.category_dict)
        with open('cache/ebnerd/subCategory-%s.json' % config.dataset, 'r', encoding='utf-8') as subCategory_f:
            self.subCategory_dict = json.load(subCategory_f)
            config.subCategory_num = len(self.subCategory_dict)
        with open('cache/ebnerd/vocabulary-' + str(config.word_threshold) + '-' + config.tokenizer + '-' + str(config.max_title_length) + '-' + str(config.max_abstract_length) + '-' + config.dataset + '.json', 'r', encoding='utf-8') as vocabulary_f:
            self.word_dict = json.load(vocabulary_f)
            config.vocabulary_size = len(self.word_dict)
        # with open('cache/ebnerd/entity-%s.json' % config.dataset, 'r', encoding='utf-8') as entity_f:
        #     self.entity_dict = json.load(entity_f)
        #     config.entity_size = len(self.entity_dict)
        with open('cache/ebnerd/user_history_graph-' + str(config.max_history_num) + ('' if config.no_self_connection else '-self') + ('' if config.no_adjacent_normalization else '-normalize-' + config.gcn_normalization_type) + '-' + config.dataset + '.pkl', 'rb') as user_history_graph_f:
            user_history_data = pickle.load(user_history_graph_f)
            self.train_user_history_graph = user_history_data['train_user_history_graph']
            self.train_user_history_category_mask = user_history_data['train_user_history_category_mask']
            self.train_user_history_category_indices = user_history_data['train_user_history_category_indices']
            self.dev_user_history_graph = user_history_data['dev_user_history_graph']
            self.dev_user_history_category_mask = user_history_data['dev_user_history_category_mask']
            self.dev_user_history_category_indices = user_history_data['dev_user_history_category_indices']
            self.test_user_history_graph = user_history_data['test_user_history_graph']
            self.test_user_history_category_mask = user_history_data['test_user_history_category_mask']
            self.test_user_history_category_indices = user_history_data['test_user_history_category_indices']

        # meta cache
        self.negative_sample_num = config.negative_sample_num                                          # negative sample number for training
        self.max_history_num = config.max_history_num                                                   # max history number for each training user
        self.max_title_length = config.max_title_length                                                 # max title length for each news text
        self.max_abstract_length = config.max_abstract_length                                           # max abstract length for each news text
        self.news_category = np.zeros([self.news_num], dtype=np.int32)                                  # [news_num]
        self.news_subCategory = np.zeros([self.news_num], dtype=np.int32)                               # [news_num]
        self.news_title_text = np.zeros([self.news_num, self.max_title_length], dtype=np.int32)         # [news_num, max_title_length]
        self.news_title_mask = np.zeros([self.news_num, self.max_title_length], dtype=np.float32)       # [news_num, max_title_length]
        self.news_title_entity = np.zeros([self.news_num, self.max_title_length], dtype=np.int32)       # [news_num, max_title_length]
        self.news_abstract_text = np.zeros([self.news_num, self.max_abstract_length], dtype=np.int32)   # [news_num, max_abstract_length]
        self.news_abstract_mask = np.zeros([self.news_num, self.max_abstract_length], dtype=np.float32) # [news_num, max_abstract_length]
        self.news_abstract_entity = np.zeros([self.news_num, self.max_abstract_length], dtype=np.int32) # [news_num, max_abstract_length]
        self.train_behaviors = []                                                                       # [user_ID, [history], [history_mask], click impression, [non-click impressions], behavior_index]
        self.dev_behaviors = []                                                                         # [user_ID, [history], [history_mask], candidate_news_ID, behavior_index]
        self.dev_indices = []                                                                           # index for dev
        self.test_behaviors = []                                                                        # [user_ID, [history], [history_mask], candidate_news_ID, behavior_index]
        self.test_indices = []                                                                          # index for test
        self.title_word_num = 0
        self.abstract_word_num = 0

        news_ID_set = set(['<PAD>'])
        news_records = []

        for prefix in [config.train_root, config.dev_root, config.test_root]:
            news_parquet = os.path.join(prefix, 'news.parquet')
            df_news = pd.read_parquet(news_parquet)

            for _, row in df_news.iterrows():
                news_ID = str(row['nid']).strip()
                if news_ID not in news_ID_set:
                    news_records.append(row)
                    news_ID_set.add(news_ID)

        assert self.news_num == len(news_ID_set), f'news num mismatch {self.news_num} v.s. {len(news_ID_set)}'

        for row in news_records:
            news_ID = str(row['nid']).strip()
            category = str(row['category']).strip()
            subCategory = str(row['subcategory']).strip()
            title = str(row['title'])
            abstract = str(row['abstract'])

            index = self.news_ID_dict[news_ID]
            self.news_category[index] = self.category_dict.get(category, 0)
            self.news_subCategory[index] = self.subCategory_dict.get(subCategory, 0)

            words = pat.findall(title.lower()) if config.tokenizer == 'MIND' else word_tokenize(title.lower())
            offsets = [-1] * len(title)
            offset_index = 0
            for i, word in enumerate(words):
                if i == self.max_title_length:
                    break
                if is_number(word):
                    self.news_title_text[index][i] = self.word_dict['<NUM>']
                else:
                    self.news_title_text[index][i] = self.word_dict.get(word, 1)
                self.news_title_mask[index][i] = 1
                while offset_index < len(title) and title[offset_index] in [' ', '\t']:
                    offset_index += 1
                for _ in range(len(word)):
                    if offset_index < len(offsets):
                        offsets[offset_index] = i
                        offset_index += 1

            self.title_word_num += len(words)

            words = pat.findall(abstract.lower()) if config.tokenizer == 'MIND' else word_tokenize(abstract.lower())
            offsets = [-1] * len(abstract)
            offset_index = 0
            for i, word in enumerate(words):
                if i == self.max_abstract_length:
                    break
                if is_number(word):
                    self.news_abstract_text[index][i] = self.word_dict['<NUM>']
                else:
                    self.news_abstract_text[index][i] = self.word_dict.get(word, 1)
                self.news_abstract_mask[index][i] = 1
                while offset_index < len(abstract) and abstract[offset_index] in [' ', '\t']:
                    offset_index += 1
                for _ in range(len(word)):
                    if offset_index < len(offsets):
                        offsets[offset_index] = i
                        offset_index += 1

            self.abstract_word_num += len(words)

        self.news_title_mask[0][0] = 1  # for <PAD> news
        self.news_abstract_mask[0][0] = 1  # for <PAD> news

        # generate behavior meta cache
        def process_behavior_df(df, mode='train'):
            for behavior_index, row in df.iterrows():
                user_ID = str(row['uid'])  # or your actual user ID col name
                history = row['history']  # assumed list or string like '[id1,id2,...]'
                labels = row.get('labels', None)  # might be None for dev/test
                impressions = row['candidates']  # string/list of impression IDs

                # print(type(row['candidates']), row['candidates'])
                # print(type(row['history']), row['history'])
                # Parse impressions & labels for train (labels required to split clicks/non-clicks)
                click_impressions = []
                non_click_impressions = []
                if mode == 'train' and labels is not None:
                    labels_list = [int(l) for l in labels]
                    impressions_list = [str(x).strip() for x in impressions]
                    for impression, label in zip(impressions_list, labels_list):
                        imp_id = self.news_ID_dict[impression.strip()]
                        if label == '0':
                            non_click_impressions.append(imp_id)
                        else:
                            click_impressions.append(imp_id)
                else:
                    # For dev/test, impressions are processed differently
                    impressions_list = [str(x).strip() for x in impressions]

                # Process user history
                if isinstance(history, str):
                    history_list = list(
                        map(lambda x: self.news_ID_dict[x], history.strip('[]').split(','))) if history.strip(
                        '[]') else []
                elif isinstance(history, list):
                    history_list = list(map(lambda x: self.news_ID_dict[x], history)) if history else []
                else:
                    history_list = []

                padding_num = max(0, self.max_history_num - len(history_list))
                user_history = history_list[-self.max_history_num:] + [0] * padding_num
                user_history_mask = np.zeros(self.max_history_num, dtype=np.float32)
                user_history_mask[:min(len(history_list), self.max_history_num)] = 1.0

                if mode == 'train':
                    for click_imp in click_impressions:
                        self.train_behaviors.append([
                            self.user_ID_dict[user_ID],
                            user_history,
                            user_history_mask,
                            click_imp,
                            non_click_impressions,
                            behavior_index
                        ])
                elif mode == 'dev':
                    for impression in impressions_list:
                        imp_id = self.news_ID_dict[impression.strip()]
                        self.dev_indices.append(behavior_index)
                        self.dev_behaviors.append([
                            self.user_ID_dict.get(user_ID, 0),
                            user_history,
                            user_history_mask,
                            imp_id,
                            behavior_index
                        ])
                elif mode == 'test':
                    for impression in impressions_list:
                        imp_id = self.news_ID_dict.get(impression.strip(), 0)
                        self.test_indices.append(behavior_index)
                        self.test_behaviors.append([
                            self.user_ID_dict.get(user_ID, 0),
                            user_history,
                            user_history_mask,
                            imp_id,
                            behavior_index
                        ])

        # Load parquet files and process
        train_path = os.path.join(config.train_root, 'behaviors.parquet')
        dev_path = os.path.join(config.dev_root, 'behaviors.parquet')
        test_path = os.path.join(config.test_root, 'behaviors.parquet')

        train_df = pd.read_parquet(train_path)
        dev_df = pd.read_parquet(dev_path)
        test_df = pd.read_parquet(test_path)

        process_behavior_df(train_df, 'train')
        process_behavior_df(dev_df, 'dev')
        process_behavior_df(test_df, 'test')

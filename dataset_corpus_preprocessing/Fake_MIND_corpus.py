import collections
import csv
import json
import os
import pickle
import re
import time

import numpy as np
import torch
import torch.utils.data as data
from nltk.tokenize import word_tokenize
from numpy.random import randint
from torchtext.vocab import GloVe

from config import Config


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


pat = re.compile(r"[\w]+|[.,!?;|]")


class Fake_MIND_Corpus:
    """GossipCop CSV corpus with the MIND tensor contract used by NRMS."""

    SPLIT_FILES = {'train': 'train.csv', 'dev': 'val.csv', 'test': 'test.csv'}
    CATEGORY_PLACEHOLDER = 'unknown'
    SUBCATEGORY_PLACEHOLDER = 'unknown'

    @staticmethod
    def _source_dir(config: Config):
        return os.path.join(config.root, config.DATASET_ROOT)

    @staticmethod
    def _read_news(config: Config):
        path = os.path.join(Fake_MIND_Corpus._source_dir(config), 'news.csv')
        with open(path, newline='', encoding='utf-8') as news_f:
            for row in csv.DictReader(news_f):
                yield {
                    'news_id': row['news_id'],
                    'category': Fake_MIND_Corpus.CATEGORY_PLACEHOLDER,
                    'subCategory': Fake_MIND_Corpus.SUBCATEGORY_PLACEHOLDER,
                    'label': row.get('label') or '',
                    'publisher': row.get('source') or '',
                    'title': row.get('title') or '',
                    'abstract': row.get('text') or '',
                }

    @staticmethod
    def _read_behaviors(config: Config, split: str):
        path = os.path.join(Fake_MIND_Corpus._source_dir(config), Fake_MIND_Corpus.SPLIT_FILES[split])
        with open(path, newline='', encoding='utf-8') as behavior_f:
            yield from csv.DictReader(behavior_f)

    @staticmethod
    def _tokens(text, config: Config):
        text = text or ''
        return pat.findall(text.lower()) if config.tokenizer == 'MIND' else word_tokenize(text.lower())

    @staticmethod
    def _impressions(row):
        candidates = row['candidates'].split()
        labels = row['clicked'].split()
        if len(candidates) != len(labels):
            raise ValueError(
                f"candidate/label mismatch for impression {row['impression_id']}: "
                f"{len(candidates)} != {len(labels)}"
            )
        return [f'{candidate}-{label}' for candidate, label in zip(candidates, labels)]

    @staticmethod
    def _cache_paths(config: Config):
        suffix = config.dataset_size
        return {
            'user_ID': 'cache/user_ID-%s.json' % suffix,
            'news_ID': 'cache/news_ID-%s.json' % suffix,
            'category': 'cache/category-%s.json' % suffix,
            'subCategory': 'cache/subCategory-%s.json' % suffix,
            'vocabulary': 'cache/vocabulary-%d-%s-%d-%d-%s.json' % (
                config.word_threshold,
                config.tokenizer,
                config.max_title_length,
                config.max_abstract_length,
                suffix,
            ),
            'word_embedding': 'cache/word_embedding-%d-%d-%s-%d-%d-%s.pkl' % (
                config.word_threshold,
                config.word_embedding_dim,
                config.tokenizer,
                config.max_title_length,
                config.max_abstract_length,
                suffix,
            ),
            'entity': 'cache/entity-%s.json' % suffix,
            'entity_embedding': 'cache/entity_embedding-%s.pkl' % suffix,
            'context_embedding': 'cache/context_embedding-%s.pkl' % suffix,
        }

    @staticmethod
    def _cache_is_valid(config: Config, paths):
        if not all(os.path.exists(path) for path in paths.values()):
            return False
        with open(paths['category'], 'r', encoding='utf-8') as f:
            category_dict = json.load(f)
        with open(paths['subCategory'], 'r', encoding='utf-8') as f:
            subCategory_dict = json.load(f)
        if category_dict != {Fake_MIND_Corpus.CATEGORY_PLACEHOLDER: 0}:
            return False
        if subCategory_dict != {Fake_MIND_Corpus.SUBCATEGORY_PLACEHOLDER: 0}:
            return False

        with open(paths['user_ID'], 'r', encoding='utf-8') as f:
            user_ID_dict = json.load(f)
        train_users = {row['user_id'] for row in Fake_MIND_Corpus._read_behaviors(config, 'train')}
        if not train_users.issubset(user_ID_dict):
            return False

        with open(paths['news_ID'], 'r', encoding='utf-8') as f:
            news_ID_dict = json.load(f)
        news_ids = {row['news_id'] for row in Fake_MIND_Corpus._read_news(config)}
        return news_ids.issubset(news_ID_dict)

    @staticmethod
    def preprocess(config: Config):
        paths = Fake_MIND_Corpus._cache_paths(config)
        if Fake_MIND_Corpus._cache_is_valid(config, paths):
            return

        os.makedirs('cache', exist_ok=True)
        user_ID_dict = {'<UNK>': 0}
        news_ID_dict = {'<PAD>': 0}
        category_dict = {}
        subCategory_dict = {}
        word_dict = {'<PAD>': 0, '<UNK>': 1}
        word_counter = collections.Counter()
        entity_dict = {'<PAD>': 0, '<UNK>': 1}

        for row in Fake_MIND_Corpus._read_behaviors(config, 'train'):
            user_ID = row['user_id']
            if user_ID not in user_ID_dict:
                user_ID_dict[user_ID] = len(user_ID_dict)

        for row in Fake_MIND_Corpus._read_news(config):
            news_ID = row['news_id']
            category = row['category']
            subCategory = row['subCategory']
            if news_ID not in news_ID_dict:
                news_ID_dict[news_ID] = len(news_ID_dict)
            if category not in category_dict:
                category_dict[category] = len(category_dict)
            if subCategory not in subCategory_dict:
                subCategory_dict[subCategory] = len(subCategory_dict)
            for text_key in ['title', 'abstract']:
                for word in Fake_MIND_Corpus._tokens(row[text_key], config):
                    word_counter['<NUM>' if is_number(word) else word] += 1

        for word, count in word_counter.most_common():
            if count >= config.word_threshold:
                word_dict[word] = len(word_dict)

        if config.word_embedding_dim == 300:
            glove = GloVe(name='840B', dim=300, cache='./glove', max_vectors=10000000000)
        else:
            glove = GloVe(name='6B', dim=config.word_embedding_dim, cache='./glove', max_vectors=10000000000)
        glove_mean_vector = torch.mean(glove.vectors, dim=0, keepdim=False)
        word_embedding_vectors = torch.zeros([len(word_dict), config.word_embedding_dim])
        for word, index in word_dict.items():
            if index == 0:
                continue
            if word in glove.stoi:
                word_embedding_vectors[index, :] = glove.vectors[glove.stoi[word]]
            else:
                random_vector = torch.zeros(config.word_embedding_dim)
                random_vector.normal_(mean=0, std=0.1)
                word_embedding_vectors[index, :] = random_vector + glove_mean_vector

        entity_embedding_vectors = torch.zeros([len(entity_dict), config.entity_embedding_dim])
        context_embedding_vectors = torch.zeros([len(entity_dict), config.context_embedding_dim])

        with open(paths['user_ID'], 'w', encoding='utf-8') as f:
            json.dump(user_ID_dict, f)
        with open(paths['news_ID'], 'w', encoding='utf-8') as f:
            json.dump(news_ID_dict, f)
        with open(paths['category'], 'w', encoding='utf-8') as f:
            json.dump(category_dict, f)
        with open(paths['subCategory'], 'w', encoding='utf-8') as f:
            json.dump(subCategory_dict, f)
        with open(paths['vocabulary'], 'w', encoding='utf-8') as f:
            json.dump(word_dict, f)
        with open(paths['entity'], 'w', encoding='utf-8') as f:
            json.dump(entity_dict, f)
        with open(paths['word_embedding'], 'wb') as f:
            pickle.dump(word_embedding_vectors, f)
        with open(paths['entity_embedding'], 'wb') as f:
            pickle.dump(entity_embedding_vectors, f)
        with open(paths['context_embedding'], 'wb') as f:
            pickle.dump(context_embedding_vectors, f)

    def __init__(self, config: Config):
        self.model = config.model
        Fake_MIND_Corpus.preprocess(config)
        paths = Fake_MIND_Corpus._cache_paths(config)

        with open(paths['user_ID'], 'r', encoding='utf-8') as f:
            self.user_ID_dict = json.load(f)
            config.user_num = len(self.user_ID_dict)
        with open(paths['news_ID'], 'r', encoding='utf-8') as f:
            self.news_ID_dict = json.load(f)
            self.news_num = len(self.news_ID_dict)
        with open(paths['category'], 'r', encoding='utf-8') as f:
            self.category_dict = json.load(f)
            config.category_num = len(self.category_dict)
        with open(paths['subCategory'], 'r', encoding='utf-8') as f:
            self.subCategory_dict = json.load(f)
            config.subCategory_num = len(self.subCategory_dict)
        with open(paths['vocabulary'], 'r', encoding='utf-8') as f:
            self.word_dict = json.load(f)
            config.vocabulary_size = len(self.word_dict)
        with open(paths['entity'], 'r', encoding='utf-8') as f:
            self.entity_dict = json.load(f)
            config.entity_size = len(self.entity_dict)

        self.negative_sample_num = config.negative_sample_num
        self.max_history_num = config.max_history_num
        self.max_title_length = config.max_title_length
        self.max_abstract_length = config.max_abstract_length
        self.news_category = np.zeros([self.news_num], dtype=np.int32)
        self.news_subCategory = np.zeros([self.news_num], dtype=np.int32)
        self.news_title_text = np.zeros([self.news_num, self.max_title_length], dtype=np.int32)
        self.news_title_mask = np.zeros([self.news_num, self.max_title_length], dtype=np.float32)
        self.news_title_entity = np.zeros([self.news_num, self.max_title_length], dtype=np.int32)
        self.news_abstract_text = np.zeros([self.news_num, self.max_abstract_length], dtype=np.int32)
        self.news_abstract_mask = np.zeros([self.news_num, self.max_abstract_length], dtype=np.float32)
        self.news_abstract_entity = np.zeros([self.news_num, self.max_abstract_length], dtype=np.int32)
        self.train_behaviors = []
        self.dev_behaviors = []
        self.dev_indices = []
        self.test_behaviors = []
        self.test_indices = []
        self.train_user_history_graph = np.zeros([0], dtype=np.float32)
        self.train_user_history_category_mask = np.zeros([0], dtype=np.float32)
        self.train_user_history_category_indices = np.zeros([0], dtype=np.int64)
        self.dev_user_history_graph = np.zeros([0], dtype=np.float32)
        self.dev_user_history_category_mask = np.zeros([0], dtype=np.float32)
        self.dev_user_history_category_indices = np.zeros([0], dtype=np.int64)
        self.test_user_history_graph = np.zeros([0], dtype=np.float32)
        self.test_user_history_category_mask = np.zeros([0], dtype=np.float32)
        self.test_user_history_category_indices = np.zeros([0], dtype=np.int64)
        self.title_word_num = 0
        self.abstract_word_num = 0

        self._load_news(config)
        self._load_train_behaviors(config)
        self._load_devtest_behaviors(config, 'dev')
        self._load_devtest_behaviors(config, 'test')

    def _load_news(self, config):
        for row in Fake_MIND_Corpus._read_news(config):
            news_ID = row['news_id']
            if news_ID not in self.news_ID_dict:
                continue
            index = self.news_ID_dict[news_ID]
            self.news_category[index] = self.category_dict.get(row['category'], 0)
            self.news_subCategory[index] = self.subCategory_dict.get(row['subCategory'], 0)
            self._fill_text(row['title'], config, self.news_title_text[index], self.news_title_mask[index], self.max_title_length)
            self._fill_text(row['abstract'], config, self.news_abstract_text[index], self.news_abstract_mask[index], self.max_abstract_length)
            self.title_word_num += len(Fake_MIND_Corpus._tokens(row['title'], config))
            self.abstract_word_num += len(Fake_MIND_Corpus._tokens(row['abstract'], config))
        self.news_title_mask[0][0] = 1
        self.news_abstract_mask[0][0] = 1

    def _fill_text(self, text, config, text_array, mask_array, max_length):
        for i, word in enumerate(Fake_MIND_Corpus._tokens(text, config)[:max_length]):
            text_array[i] = self.word_dict.get('<NUM>' if is_number(word) else word, 1)
            mask_array[i] = 1

    def _history(self, raw_history):
        history = [self.news_ID_dict[x] for x in raw_history.split() if x in self.news_ID_dict]
        padding_num = max(0, self.max_history_num - len(history))
        user_history = history[-self.max_history_num:] + [0] * padding_num
        user_history_mask = np.zeros([self.max_history_num], dtype=np.float32)
        user_history_mask[:min(len(history), self.max_history_num)] = 1.0
        return user_history, user_history_mask

    def _split_impressions(self, row):
        click_impressions = []
        non_click_impressions = []
        for impression in Fake_MIND_Corpus._impressions(row):
            if impression[-2:] == '-1':
                click_impressions.append(self.news_ID_dict[impression[:-2]])
            else:
                non_click_impressions.append(self.news_ID_dict[impression[:-2]])
        return click_impressions, non_click_impressions

    def _load_train_behaviors(self, config):
        for behavior_index, row in enumerate(Fake_MIND_Corpus._read_behaviors(config, 'train')):
            user_history, user_history_mask = self._history(row['history'])
            click_impressions, non_click_impressions = self._split_impressions(row)
            user_ID = self.user_ID_dict[row['user_id']]
            for click_impression in click_impressions:
                if len(non_click_impressions) > 0:
                    self.train_behaviors.append([user_ID, user_history, user_history_mask, click_impression, non_click_impressions, behavior_index])

    def _load_devtest_behaviors(self, config, split):
        target_behaviors = self.dev_behaviors if split == 'dev' else self.test_behaviors
        target_indices = self.dev_indices if split == 'dev' else self.test_indices
        for behavior_index, row in enumerate(Fake_MIND_Corpus._read_behaviors(config, split)):
            user_history, user_history_mask = self._history(row['history'])
            user_ID = self.user_ID_dict.get(row['user_id'], 0)
            for impression in Fake_MIND_Corpus._impressions(row):
                target_indices.append(behavior_index)
                target_behaviors.append([user_ID, user_history, user_history_mask, self.news_ID_dict[impression[:-2]], behavior_index])


class MIND_Train_Dataset(data.Dataset):
    def __init__(self, corpus: Fake_MIND_Corpus):
        self.model = corpus.model
        self.negative_sample_num = corpus.negative_sample_num
        self.news_category = corpus.news_category
        self.news_subCategory = corpus.news_subCategory
        self.news_title_text = corpus.news_title_text
        self.news_title_mask = corpus.news_title_mask
        self.news_title_entity = corpus.news_title_entity
        self.news_abstract_text = corpus.news_abstract_text
        self.news_abstract_mask = corpus.news_abstract_mask
        self.news_abstract_entity = corpus.news_abstract_entity
        self.user_history_graph = corpus.train_user_history_graph
        self.user_history_category_mask = corpus.train_user_history_category_mask
        self.user_history_category_indices = corpus.train_user_history_category_indices
        self.train_behaviors = corpus.train_behaviors
        self.train_samples = [[0 for _ in range(1 + self.negative_sample_num)] for __ in range(len(self.train_behaviors))]
        self.num = len(self.train_behaviors)

    def negative_sampling(self, rank=None):
        print('\n%sBegin negative sampling, training sample num : %d' % ('' if rank is None else ('rank ' + str(rank) + ' : '), self.num))
        start_time = time.time()
        for i, train_behavior in enumerate(self.train_behaviors):
            self.train_samples[i][0] = train_behavior[3]
            negative_samples = train_behavior[4]
            news_num = len(negative_samples)
            if news_num <= self.negative_sample_num:
                for j in range(self.negative_sample_num):
                    self.train_samples[i][j + 1] = negative_samples[j % news_num]
            else:
                used_negative_samples = set()
                for j in range(self.negative_sample_num):
                    while True:
                        k = randint(0, news_num)
                        if k not in used_negative_samples:
                            self.train_samples[i][j + 1] = negative_samples[k]
                            used_negative_samples.add(k)
                            break
        print('%sEnd negative sampling, used time : %.3fs' % ('' if rank is None else ('rank ' + str(rank) + ' : '), time.time() - start_time))

    def __getitem__(self, index):
        train_behavior = self.train_behaviors[index]
        history_index = torch.tensor(train_behavior[1])
        sample_index = torch.tensor(self.train_samples[index])
        return train_behavior[0], self.news_category[history_index], self.news_subCategory[history_index], self.news_title_text[history_index], self.news_title_mask[history_index], self.news_title_entity[history_index], self.news_abstract_text[history_index], self.news_abstract_mask[history_index], self.news_abstract_entity[history_index], train_behavior[2], self.user_history_graph, self.user_history_category_mask, self.user_history_category_indices, self.news_category[sample_index], self.news_subCategory[sample_index], self.news_title_text[sample_index], self.news_title_mask[sample_index], self.news_title_entity[sample_index], self.news_abstract_text[sample_index], self.news_abstract_mask[sample_index], self.news_abstract_entity[sample_index], history_index, sample_index

    def __len__(self):
        return self.num


class MIND_DevTest_Dataset(data.Dataset):
    def __init__(self, corpus: Fake_MIND_Corpus, mode: str):
        assert mode in ['dev', 'test'], "mode must be chosen from 'dev' or 'test'"
        self.model = corpus.model
        self.news_category = corpus.news_category
        self.news_subCategory = corpus.news_subCategory
        self.news_title_text = corpus.news_title_text
        self.news_title_mask = corpus.news_title_mask
        self.news_title_entity = corpus.news_title_entity
        self.news_abstract_text = corpus.news_abstract_text
        self.news_abstract_mask = corpus.news_abstract_mask
        self.news_abstract_entity = corpus.news_abstract_entity
        self.user_history_graph = corpus.dev_user_history_graph if mode == 'dev' else corpus.test_user_history_graph
        self.user_history_category_mask = corpus.dev_user_history_category_mask if mode == 'dev' else corpus.test_user_history_category_mask
        self.user_history_category_indices = corpus.dev_user_history_category_indices if mode == 'dev' else corpus.test_user_history_category_indices
        self.behaviors = corpus.dev_behaviors if mode == 'dev' else corpus.test_behaviors
        self.num = len(self.behaviors)

    def __getitem__(self, index):
        behavior = self.behaviors[index]
        history_index = torch.tensor(behavior[1])
        candidate_news_index = torch.tensor(behavior[3])
        return behavior[0], self.news_category[history_index], self.news_subCategory[history_index], self.news_title_text[history_index], self.news_title_mask[history_index], self.news_title_entity[history_index], self.news_abstract_text[history_index], self.news_abstract_mask[history_index], self.news_abstract_entity[history_index], behavior[2], self.user_history_graph, self.user_history_category_mask, self.user_history_category_indices, self.news_category[candidate_news_index], self.news_subCategory[candidate_news_index], self.news_title_text[candidate_news_index], self.news_title_mask[candidate_news_index], self.news_title_entity[candidate_news_index], self.news_abstract_text[candidate_news_index], self.news_abstract_mask[candidate_news_index], self.news_abstract_entity[candidate_news_index], history_index, candidate_news_index

    def __len__(self):
        return self.num

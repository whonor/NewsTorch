import os
import json
import pickle
import collections
import re
import random
import time

import nltk
import pandas as pd
import numpy as np
import torch
import torch.utils.data as data
from nltk.tokenize import word_tokenize, RegexpTokenizer
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from torchtext.vocab import GloVe
from tqdm import tqdm

from config import Config

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

pat = re.compile(r"[\w]+|[.,!?;|]")

# nltk.download('stopwords')
# nltk.download('vader_lexicon')
# nltk.download('punkt')

stop_words = set(stopwords.words('english'))
word_tokenizer = RegexpTokenizer(r'\w+')

def remove_stopword(sentence):
    return ' '.join([word for word in word_tokenizer.tokenize(sentence) if word not in stop_words])

class MIND_Corpus_SentiDebias:
    @staticmethod
    def preprocess(config: Config):
        user_ID_file = 'cache/user_ID-%s.json' % config.DATASET_ROOT
        news_ID_file = 'cache/news_ID-%s.json' % config.DATASET_ROOT
        category_file = 'cache/category-%s.json' % config.DATASET_ROOT
        subCategory_file = 'cache/subCategory-%s.json' % config.DATASET_ROOT
        sentiment_file = 'cache/sentiment-%s.pkl' % config.DATASET_ROOT
        vocabulary_file = 'cache/vocabulary-' + str(config.word_threshold) + '-' + config.tokenizer + '-' + str(config.max_title_length) + '-' + str(config.max_abstract_length) + '-' + config.DATASET_ROOT + '.json'
        word_embedding_file = 'cache/word_embedding-' + str(config.word_threshold) + '-' + str(config.word_embedding_dim) + '-' + config.tokenizer + '-' + str(config.max_title_length) + '-' + str(config.max_abstract_length) + '-' + config.DATASET_ROOT + '.pkl'
        entity_file = 'cache/entity-%s.json' % config.DATASET_ROOT
        entity_embedding_file = 'cache/entity_embedding-%s.pkl' % config.DATASET_ROOT
        context_embedding_file = 'cache/context_embedding-%s.pkl' % config.DATASET_ROOT
        user_history_graph_file = 'cache/user_history_graph-' + str(config.max_history_num) + ('' if config.no_self_connection else '-self') + ('' if config.no_adjacent_normalization else '-normalize-' + config.gcn_normalization_type) + '-' + config.DATASET_ROOT + '.pkl'
        
        preprocessed_data_files = [user_ID_file, news_ID_file, category_file, subCategory_file, vocabulary_file, word_embedding_file, entity_file, entity_embedding_file, context_embedding_file, user_history_graph_file, sentiment_file]

        if not all(list(map(os.path.exists, preprocessed_data_files))):
            user_ID_dict = {'<UNK>': 0}
            news_ID_dict = {'<PAD>': 0}
            category_dict = {}
            subCategory_dict = {}
            word_dict = {'<PAD>': 0, '<UNK>': 1}
            word_counter = collections.Counter()
            entity_dict = {'<PAD>': 0, '<UNK>': 1}
            news_category_dict = {}
            news_sentiment_dict = {0: 0.0} # PAD news has 0 sentiment

            print("Downloading NLTK data...")
            nltk.download('vader_lexicon', quiet=True)
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            sia = SentimentIntensityAnalyzer()

            # 1. user ID dictionary
            print("Processing user IDs...")
            with open(os.path.join(config.train_root, 'behaviors.tsv'), 'r', encoding='utf-8') as train_behaviors_f:
                for line in train_behaviors_f:
                    impression_ID, user_ID, time_str, history, impressions = line.split('	')
                    if user_ID not in user_ID_dict:
                        user_ID_dict[user_ID] = len(user_ID_dict)
            with open(user_ID_file, 'w', encoding='utf-8') as user_ID_f:
                json.dump(user_ID_dict, user_ID_f)

            # 2. news ID dictionary & categories & sentiment
            print("Processing news and sentiment...")
            for i, prefix in enumerate([config.train_root, config.dev_root, config.test_root]):
                with open(os.path.join(prefix, 'news.tsv'), 'r', encoding='utf-8') as news_f:
                    for line in news_f:
                        news_ID, category, subCategory, title, abstract, _, title_entities, abstract_entities = line.split('	')
                        if news_ID not in news_ID_dict:
                            news_id_idx = len(news_ID_dict)
                            news_ID_dict[news_ID] = news_id_idx
                            
                            if category not in category_dict:
                                category_dict[category] = len(category_dict)
                            if subCategory not in subCategory_dict:
                                subCategory_dict[subCategory] = len(subCategory_dict)
                            
                            # Calculate sentiment using VADER on Title + Abstract
                            text_for_sentiment = title + " " + abstract
                            sentiment_score = sia.polarity_scores(text_for_sentiment)['compound']
                            news_sentiment_dict[news_id_idx] = sentiment_score

                            words = pat.findall(title.lower()) if config.tokenizer == 'MIND' else word_tokenize(title.lower())
                            for word in words:
                                if is_number(word):
                                    word_counter['<NUM>'] += 1
                                else:
                                    if i == 0: # training set
                                        word_counter[word] += 1
                                    else:
                                        if word in word_counter:
                                            word_counter[word] += 1
                                            
                            words = pat.findall(abstract.lower()) if config.tokenizer == 'MIND' else word_tokenize(abstract.lower())
                            for word in words:
                                if is_number(word):
                                    word_counter['<NUM>'] += 1
                                else:
                                    if i == 0: # training set
                                        word_counter[word] += 1
                                    else:
                                        if word in word_counter:
                                            word_counter[word] += 1
                                            
                            for entity in json.loads(title_entities):
                                WikidataId = entity['WikidataId']
                                if WikidataId not in entity_dict:
                                    entity_dict[WikidataId] = len(entity_dict)
                            for entity in json.loads(abstract_entities):
                                WikidataId = entity['WikidataId']
                                if WikidataId not in entity_dict:
                                    entity_dict[WikidataId] = len(entity_dict)
                        
                        news_category_dict[news_ID] = category_dict[category]

            with open(news_ID_file, 'w', encoding='utf-8') as news_ID_f:
                json.dump(news_ID_dict, news_ID_f)
            with open(category_file, 'w', encoding='utf-8') as category_f:
                json.dump(category_dict, category_f)
            with open(subCategory_file, 'w', encoding='utf-8') as subCategory_f:
                json.dump(subCategory_dict, subCategory_f)
            with open(sentiment_file, 'wb') as sentiment_f:
                pickle.dump(news_sentiment_dict, sentiment_f)

            # 3. word dictionary
            word_counter_list = [[word, word_counter[word]] for word in word_counter]
            word_counter_list.sort(key=lambda x: x[1], reverse=True)
            filtered_word_counter_list = list(filter(lambda x: x[1] >= config.word_threshold, word_counter_list))
            for i, word in enumerate(filtered_word_counter_list):
                word_dict[word[0]] = i + 2
            with open(vocabulary_file, 'w', encoding='utf-8') as vocabulary_f:
                json.dump(word_dict, vocabulary_f)

            # 4. GloVe word embedding
            if config.word_embedding_dim == 300:
                glove = GloVe(name='840B', dim=300, cache='./glove', max_vectors=10000000000)
            else:
                glove = GloVe(name='6B', dim=config.word_embedding_dim, cache='./glove', max_vectors=10000000000)
            glove_stoi = glove.stoi
            glove_vectors = glove.vectors
            glove_mean_vector = torch.mean(glove_vectors, dim=0, keepdim=False)
            word_embedding_vectors = torch.zeros([len(word_dict), config.word_embedding_dim])
            for word in word_dict:
                index = word_dict[word]
                if index != 0:
                    if word in glove_stoi:
                        word_embedding_vectors[index, :] = glove_vectors[glove_stoi[word]]
                    else:
                        random_vector = torch.zeros(config.word_embedding_dim)
                        random_vector.normal_(mean=0, std=0.1)
                        word_embedding_vectors[index, :] = random_vector + glove_mean_vector
            with open(word_embedding_file, 'wb') as word_embedding_f:
                pickle.dump(word_embedding_vectors, word_embedding_f)

            # 5. Entity embeddings
            entity_embedding_vectors = torch.zeros([len(entity_dict), config.entity_embedding_dim])
            context_embedding_vectors = torch.zeros([len(entity_dict), config.context_embedding_dim])
            for prefix in [config.train_root, config.dev_root, config.test_root]:
                entity_path = os.path.join(prefix, 'entity_embedding.vec')
                if os.path.exists(entity_path):
                    with open(entity_path, 'r', encoding='utf-8') as entity_f:
                        for line in entity_f:
                            if len(line.strip()) > 0:
                                terms = line.strip().split('	')
                                WikidataId = terms[0]
                                if WikidataId in entity_dict:
                                    entity_embedding_vectors[entity_dict[WikidataId]] = torch.FloatTensor(list(map(float, terms[1:])))
                context_path = os.path.join(prefix, 'context_embedding.vec')
                if os.path.exists(context_path):
                    with open(context_path, 'r', encoding='utf-8') as context_f:
                        for line in context_f:
                            if len(line.strip()) > 0:
                                terms = line.strip().split('	')
                                WikidataId = terms[0]
                                if WikidataId in entity_dict:
                                    context_embedding_vectors[entity_dict[WikidataId]] = torch.FloatTensor(list(map(float, terms[1:])))
            with open(entity_file, 'w', encoding='utf-8') as entity_f:
                json.dump(entity_dict, entity_f)
            with open(entity_embedding_file, 'wb') as entity_embedding_f:
                pickle.dump(entity_embedding_vectors, entity_embedding_f)
            with open(context_embedding_file, 'wb') as context_embedding_f:
                pickle.dump(context_embedding_vectors, context_embedding_f)

            # 6. User history graph (for models like CNE-SUE, CNRCL)
            if config.model in ['CNE-SUE', 'CNRCL']:
                category_num = len(category_dict)
                graph_size = config.max_history_num + category_num
                prefix_mode = ['train', 'dev', 'test']
                user_history_graph_data = {}
                for prefix_index, prefix in enumerate([config.train_root, config.dev_root, config.test_root]):
                    mode = prefix_mode[prefix_index]
                    with open(os.path.join(prefix, 'behaviors.tsv'), 'r', encoding='utf-8') as behaviors_f:
                        lines = behaviors_f.readlines()
                    user_history_num = len(lines)
                    user_history_graph = np.zeros([user_history_num, graph_size, graph_size], dtype=np.float32)
                    user_history_category_mask = np.zeros([user_history_num, category_num + 1], dtype=np.float32)
                    user_history_category_indices = np.zeros([user_history_num, config.max_history_num], dtype=np.int64)
                    
                    for line_index, line in enumerate(lines):
                        impression_ID, user_ID, time_str, history, impressions = line.split('	')
                        if config.no_self_connection:
                            history_graph = np.zeros([graph_size, graph_size], dtype=np.float32)
                        else:
                            history_graph = np.identity(graph_size, dtype=np.float32)
                        history_category_mask = np.zeros(category_num + 1, dtype=np.float32)
                        history_category_indices = np.full([config.max_history_num], category_num, dtype=np.int64)
                        
                        if len(history.strip()) > 0:
                            history_news_IDs = history.split(' ')
                            offset = max(0, len(history_news_IDs) - config.max_history_num)
                            history_news_num = min(len(history_news_IDs), config.max_history_num)
                            for i in range(history_news_num):
                                news_id = history_news_IDs[i + offset]
                                category_index = news_category_dict.get(news_id, 0)
                                history_category_mask[category_index] = 1.0
                                history_category_indices[i] = category_index
                                history_graph[i, config.max_history_num + category_index] = 1
                                history_graph[config.max_history_num + category_index, i] = 1
                                for j in range(i + 1, history_news_num):
                                    other_news_id = history_news_IDs[j + offset]
                                    other_category_idx = news_category_dict.get(other_news_id, 0)
                                    if category_index == other_category_idx:
                                        history_graph[i, j] = 1
                                        history_graph[j, i] = 1
                                    else:
                                        history_graph[config.max_history_num + category_index, config.max_history_num + other_category_idx] = 1
                                        history_graph[config.max_history_num + other_category_idx, config.max_history_num + category_index] = 1
                        
                        user_history_graph[line_index] = history_graph
                        user_history_category_mask[line_index] = history_category_mask
                        user_history_category_indices[line_index] = history_category_indices
                    
                    user_history_graph_data[mode + '_user_history_graph'] = user_history_graph
                    user_history_graph_data[mode + '_user_history_category_mask'] = user_history_category_mask
                    user_history_graph_data[mode + '_user_history_category_indices'] = user_history_category_indices
                
                with open(user_history_graph_file, 'wb') as f:
                    pickle.dump(user_history_graph_data, f)

    def __init__(self, config: Config):
        self.config = config
        MIND_Corpus_SentiDebias.preprocess(config)
        
        with open('cache/user_ID-%s.json' % config.DATASET_ROOT, 'r', encoding='utf-8') as f:
            self.user_ID_dict = json.load(f)
            config.user_num = len(self.user_ID_dict)
        with open('cache/news_ID-%s.json' % config.DATASET_ROOT, 'r', encoding='utf-8') as f:
            self.news_ID_dict = json.load(f)
            self.news_num = len(self.news_ID_dict)
        with open('cache/category-%s.json' % config.DATASET_ROOT, 'r', encoding='utf-8') as f:
            self.category_dict = json.load(f)
            config.category_num = len(self.category_dict)
        with open('cache/subCategory-%s.json' % config.DATASET_ROOT, 'r', encoding='utf-8') as f:
            self.subCategory_dict = json.load(f)
            config.subCategory_num = len(self.subCategory_dict)
        with open('cache/sentiment-%s.pkl' % config.DATASET_ROOT, 'rb') as f:
            self.news_sentiment_dict = pickle.load(f)
        with open('cache/vocabulary-' + str(config.word_threshold) + '-' + config.tokenizer + '-' + str(config.max_title_length) + '-' + str(config.max_abstract_length) + '-' + config.DATASET_ROOT + '.json', 'r', encoding='utf-8') as f:
            self.word_dict = json.load(f)
            config.vocabulary_size = len(self.word_dict)
        with open('cache/entity-%s.json' % config.DATASET_ROOT, 'r', encoding='utf-8') as f:
            self.entity_dict = json.load(f)
            config.entity_size = len(self.entity_dict)

        if config.model in ['CNE-SUE', 'CNRCL']:
            user_history_graph_file = 'cache/user_history_graph-' + str(config.max_history_num) + ('' if config.no_self_connection else '-self') + ('' if config.no_adjacent_normalization else '-normalize-' + config.gcn_normalization_type) + '-' + config.DATASET_ROOT + '.pkl'
            with open(user_history_graph_file, 'rb') as f:
                user_history_data = pickle.load(f)
            self.train_user_history_graph = user_history_data['train_user_history_graph']
            self.train_user_history_category_mask = user_history_data['train_user_history_category_mask']
            self.train_user_history_category_indices = user_history_data['train_user_history_category_indices']
            self.dev_user_history_graph = user_history_data['dev_user_history_graph']
            self.dev_user_history_category_mask = user_history_data['dev_user_history_category_mask']
            self.dev_user_history_category_indices = user_history_data['dev_user_history_category_indices']
            self.test_user_history_graph = user_history_data['test_user_history_graph']
            self.test_user_history_category_mask = user_history_data['test_user_history_category_mask']
            self.test_user_history_category_indices = user_history_data['test_user_history_category_indices']
        else:
            self.train_user_history_graph = None
            self.train_user_history_category_mask = None
            self.train_user_history_category_indices = None
            self.dev_user_history_graph = None
            self.dev_user_history_category_mask = None
            self.dev_user_history_category_indices = None
            self.test_user_history_graph = None
            self.test_user_history_category_mask = None
            self.test_user_history_category_indices = None

        self.negative_sample_num = config.negative_sample_num
        self.max_history_num = config.max_history_num
        self.max_title_length = config.max_title_length
        self.max_abstract_length = config.max_abstract_length
        
        self.news_category = np.zeros([self.news_num], dtype=np.int32)
        self.news_subCategory = np.zeros([self.news_num], dtype=np.int32)
        self.news_sentiment = np.zeros([self.news_num], dtype=np.float32)
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

        news_ID_set = set(['<PAD>'])
        news_lines = []
        for prefix in [config.train_root, config.dev_root, config.test_root]:
            with open(os.path.join(prefix, 'news.tsv'), 'r', encoding='utf-8') as f:
                for line in f:
                    news_ID = line.split('	')[0]
                    if news_ID not in news_ID_set:
                        news_lines.append(line)
                        news_ID_set.add(news_ID)
        
        for line in news_lines:
            news_ID, category, subCategory, title, abstract, _, title_entities, abstract_entities = line.split('	')
            index = self.news_ID_dict[news_ID]
            self.news_category[index] = self.category_dict.get(category, 0)
            self.news_subCategory[index] = self.subCategory_dict.get(subCategory, 0)
            self.news_sentiment[index] = self.news_sentiment_dict.get(index, 0.0)
            
            words = pat.findall(title.lower()) if config.tokenizer == 'MIND' else word_tokenize(title.lower())
            offsets = [-1] * len(title)
            offset_index = 0
            for i, word in enumerate(words):
                if i == self.max_title_length: break
                self.news_title_text[index][i] = self.word_dict.get(word, 1) if not is_number(word) else self.word_dict.get('<NUM>', 1)
                self.news_title_mask[index][i] = 1
                while offset_index < len(title) and title[offset_index] in [' ', '	']: offset_index += 1
                for _ in range(len(word)):
                    if offset_index < len(offsets):
                        offsets[offset_index] = i
                        offset_index += 1
            for entity in json.loads(title_entities):
                WikidataId = entity['WikidataId']
                for offset in entity['OccurrenceOffsets']:
                    if offset < len(offsets) and offsets[offset] != -1 and WikidataId in self.entity_dict:
                        self.news_title_entity[index][offsets[offset]] = self.entity_dict[WikidataId]

            words = pat.findall(abstract.lower()) if config.tokenizer == 'MIND' else word_tokenize(abstract.lower())
            offsets = [-1] * len(abstract)
            offset_index = 0
            for i, word in enumerate(words):
                if i == self.max_abstract_length: break
                self.news_abstract_text[index][i] = self.word_dict.get(word, 1) if not is_number(word) else self.word_dict.get('<NUM>', 1)
                self.news_abstract_mask[index][i] = 1
                while offset_index < len(abstract) and abstract[offset_index] in [' ', '	']: offset_index += 1
                for _ in range(len(word)):
                    if offset_index < len(offsets):
                        offsets[offset_index] = i
                        offset_index += 1
            for entity in json.loads(abstract_entities):
                WikidataId = entity['WikidataId']
                for offset in entity['OccurrenceOffsets']:
                    if offset < len(offsets) and offsets[offset] != -1 and WikidataId in self.entity_dict:
                        self.news_abstract_entity[index][offsets[offset]] = self.entity_dict[WikidataId]
        
        self.news_title_mask[0][0] = 1
        self.news_abstract_mask[0][0] = 1

        # Process behaviors
        with open(os.path.join(config.train_root, 'behaviors.tsv'), 'r', encoding='utf-8') as f:
            for behavior_index, line in enumerate(f):
                _, user_ID, _, history, impressions = line.split('	')
                click_imps = []
                non_click_imps = []
                for imp in impressions.strip().split(' '):
                    if imp.endswith('-1'): click_imps.append(self.news_ID_dict[imp[:-2]])
                    else: non_click_imps.append(self.news_ID_dict[imp[:-2]])
                
                history_list = [self.news_ID_dict[x] for x in history.strip().split(' ')] if history.strip() else []
                padding_num = max(0, self.max_history_num - len(history_list))
                user_history = history_list[-self.max_history_num:] + [0] * padding_num
                user_history_mask = np.zeros(self.max_history_num, dtype=np.float32)
                user_history_mask[:min(len(history_list), self.max_history_num)] = 1.0
                
                for click_imp in click_imps:
                    if non_click_imps:
                        self.train_behaviors.append([self.user_ID_dict[user_ID], user_history, user_history_mask, click_imp, non_click_imps, behavior_index])

        for mode, root in [('dev', config.dev_root), ('test', config.test_root)]:
            behaviors = self.dev_behaviors if mode == 'dev' else self.test_behaviors
            indices = self.dev_indices if mode == 'dev' else self.test_indices
            with open(os.path.join(root, 'behaviors.tsv'), 'r', encoding='utf-8') as f:
                for dev_ID, line in enumerate(f):
                    _, user_ID, _, history, impressions = line.split('	')
                    history_list = [self.news_ID_dict[x] for x in history.strip().split(' ')] if history.strip() else []
                    padding_num = max(0, self.max_history_num - len(history_list))
                    user_history = history_list[-self.max_history_num:] + [0] * padding_num
                    user_history_mask = np.zeros(self.max_history_num, dtype=np.float32)
                    user_history_mask[:min(len(history_list), self.max_history_num)] = 1.0
                    
                    for imp in impressions.strip().split(' '):
                        indices.append(dev_ID)
                        nid = imp[:-2] if '-' in imp else imp
                        behaviors.append([self.user_ID_dict.get(user_ID, 0), user_history, user_history_mask, self.news_ID_dict[nid], dev_ID])


class MIND_Train_Dataset_SentiDebias(data.Dataset):
    def __init__(self, corpus: MIND_Corpus_SentiDebias):
        self.config = corpus.config
        self.negative_sample_num = corpus.negative_sample_num
        self.news_category = corpus.news_category
        self.news_subCategory = corpus.news_subCategory
        self.news_sentiment = corpus.news_sentiment
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
        for i, train_behavior in enumerate(self.train_behaviors):
            self.train_samples[i][0] = train_behavior[3]
            negative_samples = train_behavior[4]
            for j in range(self.negative_sample_num):
                self.train_samples[i][j + 1] = random.choice(negative_samples)

    def __getitem__(self, index):
        behavior = self.train_behaviors[index]
        h_idx = behavior[1]
        s_idx = self.train_samples[index]
        b_idx = behavior[5]
        
        # Placeholder for image embeddings if needed by trainer
        img_placeholder = np.zeros(2048, dtype=np.float32)
        
        return (behavior[0], self.news_category[h_idx], self.news_subCategory[h_idx], self.news_title_text[h_idx], self.news_title_mask[h_idx], self.news_title_entity[h_idx], self.news_abstract_text[h_idx], self.news_abstract_mask[h_idx], self.news_abstract_entity[h_idx], behavior[2],
                self.news_category[s_idx], self.news_subCategory[s_idx], self.news_title_text[s_idx], self.news_title_mask[s_idx], self.news_title_entity[s_idx], self.news_abstract_text[s_idx], self.news_abstract_mask[s_idx], self.news_abstract_entity[s_idx], self.news_sentiment[h_idx], self.news_sentiment[s_idx], h_idx, s_idx,
                img_placeholder, img_placeholder)

    def __len__(self):
        return self.num


class MIND_DevTest_Dataset_SentiDebias(data.Dataset):
    def __init__(self, corpus: MIND_Corpus_SentiDebias, mode: str):
        self.config = corpus.config
        self.news_category = corpus.news_category
        self.news_subCategory = corpus.news_subCategory
        self.news_sentiment = corpus.news_sentiment
        self.news_title_text = corpus.news_title_text
        self.news_title_mask = corpus.news_title_mask
        self.news_title_entity = corpus.news_title_entity
        self.news_abstract_text = corpus.news_abstract_text
        self.news_abstract_mask = corpus.news_abstract_mask
        self.news_abstract_entity = corpus.news_abstract_entity
        self.user_history_graph = getattr(corpus, mode + '_user_history_graph')
        self.user_history_category_mask = getattr(corpus, mode + '_user_history_category_mask')
        self.user_history_category_indices = getattr(corpus, mode + '_user_history_category_indices')
        self.behaviors = getattr(corpus, mode + '_behaviors')
        self.num = len(self.behaviors)

    def __getitem__(self, index):
        behavior = self.behaviors[index]
        h_idx = behavior[1]
        c_idx = behavior[3]
        b_idx = behavior[4]
        
        img_placeholder = np.zeros(2048, dtype=np.float32)
        
        return (behavior[0], self.news_category[h_idx], self.news_subCategory[h_idx], self.news_title_text[h_idx], self.news_title_mask[h_idx], self.news_title_entity[h_idx], self.news_abstract_text[h_idx], self.news_abstract_mask[h_idx], self.news_abstract_entity[h_idx], behavior[2],
                self.news_category[c_idx], self.news_subCategory[c_idx], self.news_title_text[c_idx], self.news_title_mask[c_idx], self.news_title_entity[c_idx], self.news_abstract_text[c_idx], self.news_abstract_mask[c_idx], self.news_abstract_entity[c_idx], self.news_sentiment[h_idx], self.news_sentiment[c_idx], h_idx, c_idx,
                img_placeholder, img_placeholder)

    def __len__(self):
        return self.num

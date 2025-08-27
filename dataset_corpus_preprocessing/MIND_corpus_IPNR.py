import os
import json
import pickle
import collections
import re
from nltk.tokenize import word_tokenize
from torchtext.vocab import GloVe
from config import Config
import torch
import numpy as np
import networkx as nx
from config import Config


def build_subgraph(graph: nx.Graph, vocab: list):
    all_ents = set([])
    all_ents = all_ents.union(vocab)

    all_ents = list(all_ents)
    sub_graph = graph.subgraph(all_ents)

    remove = [node for node, degree in dict(sub_graph.degree()).items() if degree < 1]
    # print("Remove_nodes: ")
    # print(remove)
    g = sub_graph.copy()
    g.remove_nodes_from(remove)
    print(g)
    # adj是图G的邻接矩阵的稀疏表示，左边节点对代表一条边，右边是边的值，adj是对称矩阵。
    adj = nx.to_scipy_sparse_array(g).tocoo()

    # row是adj中非零元素所在的行索引
    row = torch.from_numpy(adj.row.astype(np.int64)).to(torch.long)
    # col是adj中非零元素所在的列索引。
    col = torch.from_numpy(adj.col.astype(np.int64)).to(torch.long)

    # 将行和列进行拼接，shape变为[2, num_edges], 包含两个列表，第一个是row, 第二个是col
    edge_index = torch.stack([row, col], dim=0)

    return edge_index, adj


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

pat = re.compile(r"[\w]+|[.,!?;|]")

class MIND_Corpus_IPNR:
    @staticmethod
    def preprocess(config: Config):
        user_ID_file = 'cache/IPNR/user_ID-%s.json' % config.dataset_size
        news_ID_file = 'cache/IPNR/news_ID-%s.json' % config.dataset_size
        category_file = 'cache/IPNR/category-%s.json' % config.dataset_size
        subCategory_file = 'cache/IPNR/subCategory-%s.json' % config.dataset_size
        vocabulary_file = 'cache/IPNR/vocabulary-' + str(config.word_threshold) + '-' + config.tokenizer + '-' + str(config.max_title_length) + '-' + str(config.max_abstract_length) + '-' + config.dataset_size + '.json'
        word_embedding_file = 'cache/IPNR/word_embedding-' + str(config.word_threshold) + '-' + str(config.word_embedding_dim) + '-' + config.tokenizer + '-' + str(config.max_title_length) + '-' + str(config.max_abstract_length) + '-' + config.dataset_size + '.pkl'
        entity_file = 'cache/IPNR/entity-%s.json' % config.dataset_size
        entity_embedding_file = 'cache/IPNR/entity_embedding-%s.pkl' % config.dataset_size
        context_embedding_file = 'cache/IPNR/context_embedding-%s.pkl' % config.dataset_size
        user_history_graph_file = 'cache/IPNR/user_history_graph-' + str(config.max_history_num) + ('' if config.no_self_connection else '-self') + ('' if config.no_adjacent_normalization else '-normalize-' + config.gcn_normalization_type) + '-' + config.dataset_size + '.pkl'
        preprocessed_data_files = [user_ID_file, news_ID_file, category_file, subCategory_file, vocabulary_file, word_embedding_file, entity_file, entity_embedding_file, context_embedding_file, user_history_graph_file]
        content_file = 'cache/IPNR/content-%s.json' % config.dataset_size

        if not all(list(map(os.path.exists, preprocessed_data_files))):
            user_ID_dict = {'<UNK>': 0}
            news_ID_dict = {'<PAD>': 0}
            category_dict = {}
            subCategory_dict = {}
            word_dict = {'<PAD>': 0, '<UNK>': 1}
            word_counter = collections.Counter()
            entity_dict = {'<PAD>': 0, '<UNK>': 1}
            news_category_dict = {}

            content_dict = {}
            news_content_dict = {}

            # 1. user ID dictionay
            with open(os.path.join(config.train_root, 'behaviors.tsv'), 'r', encoding='utf-8') as train_behaviors_f:
                for line in train_behaviors_f:
                    impression_ID, user_ID, time, history, impressions = line.split('\t')
                    if user_ID not in user_ID_dict:
                        user_ID_dict[user_ID] = len(user_ID_dict)
                with open(user_ID_file, 'w', encoding='utf-8') as user_ID_f:
                    json.dump(user_ID_dict, user_ID_f)

            # 2. news ID dictionay & news category dictionay & news subCategory dictionay
            for i, prefix in enumerate([config.train_root, config.dev_root, config.test_root]):
                with open(os.path.join(prefix, 'news_.tsv'), 'r', encoding='utf-8') as news_f:
                    for line in news_f:
                        news_ID, category, subCategory, title, abstract, title_entities, abstract_entities, content = line.split('\t')
                        if news_ID not in news_ID_dict:
                            news_ID_dict[news_ID] = len(news_ID_dict)
                            if category not in category_dict:
                                category_dict[category] = len(category_dict)
                            if subCategory not in subCategory_dict:
                                subCategory_dict[subCategory] = len(subCategory_dict)

                            if content not in content_dict:
                                content_dict[news_ID] = len(content_dict)

                            words = pat.findall(title.lower()) if config.tokenizer == 'MIND' else word_tokenize(title.lower())
                            for word in words:
                                if is_number(word):
                                    word_counter['<NUM>'] += 1
                                else:
                                    if i == 0: # training set
                                        word_counter[word] += 1
                                    else:
                                        if word in word_counter: # already appeared in training set
                                            word_counter[word] += 1
                            words = pat.findall(abstract.lower()) if config.tokenizer == 'MIND' else word_tokenize(abstract.lower())
                            for word in words:
                                if is_number(word):
                                    word_counter['<NUM>'] += 1
                                else:
                                    if i == 0: # training set
                                        word_counter[word] += 1
                                    else:
                                        if word in word_counter: # already appeared in training set
                                            word_counter[word] += 1

                            for entity in json.loads(title_entities):
                                WikidataId = entity['WikidataId']
                                if WikidataId not in entity_dict:
                                    entity_dict[WikidataId] = len(entity_dict)
                            for entity in json.loads(abstract_entities):
                                WikidataId = entity['WikidataId']
                                if WikidataId not in entity_dict:
                                    entity_dict[WikidataId] = len(entity_dict)

                            words = pat.findall(content.lower()) if config.tokenizer == 'MIND' else word_tokenize(
                                content.lower())
                            for word in words:
                                if is_number(word):
                                    word_counter['<NUM>'] += 1
                                else:
                                    if i == 0:  # training set
                                        word_counter[word] += 1
                                    else:
                                        if word in word_counter:  # already appeared in training set
                                            word_counter[word] += 1

                        news_category_dict[news_ID] = category_dict[category]
                        news_content_dict[news_ID] = content.strip().split(' ')

            with open(news_ID_file, 'w', encoding='utf-8') as news_ID_f:
                json.dump(news_ID_dict, news_ID_f)
            with open(category_file, 'w', encoding='utf-8') as category_f:
                json.dump(category_dict, category_f)
            with open(subCategory_file, 'w', encoding='utf-8') as subCategory_f:
                json.dump(subCategory_dict, subCategory_f)

            with open(content_file, 'w', encoding='utf-8') as content_f:
                json.dump(content_dict, content_f)

            # 3. word dictionay
            word_counter_list = [[word, word_counter[word]] for word in word_counter]
            word_counter_list.sort(key=lambda x: x[1], reverse=True) # sort by word frequency
            filtered_word_counter_list = list(filter(lambda x: x[1] >= config.word_threshold, word_counter_list))
            for i, word in enumerate(filtered_word_counter_list):
                word_dict[word[0]] = i + 2
            with open(vocabulary_file, 'w', encoding='utf-8') as vocabulary_f:
                json.dump(word_dict, vocabulary_f)

            # 4. Glove word embedding
            if config.word_embedding_dim == 300:
                glove = GloVe(name='840B', dim=300, cache='../glove', max_vectors=10000000000)
            else:
                glove = GloVe(name='6B', dim=config.word_embedding_dim, cache='../glove', max_vectors=10000000000)
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

            # 5. knowledge-graph entity dictionary & eneity embedding & context embedding
            entity_embedding_vectors = torch.zeros([len(entity_dict), config.entity_embedding_dim])
            context_embedding_vectors = torch.zeros([len(entity_dict), config.context_embedding_dim])
            for prefix in [config.train_root, config.dev_root, config.test_root]:
                with open(os.path.join(prefix, 'entity_embedding.vec'), 'r', encoding='utf-8') as entity_f:
                    for line in entity_f:
                        if len(line.strip()) > 0:
                            terms = line.strip().split('\t')
                            assert len(terms) == config.entity_embedding_dim + 1, 'entity embedding dim does not match'
                            WikidataId = terms[0]
                            if WikidataId in entity_dict:
                                entity_embedding_vectors[entity_dict[WikidataId]] = torch.FloatTensor(list(map(float, terms[1:])))
            for prefix in [config.train_root, config.dev_root, config.test_root]:
                with open(os.path.join(prefix, 'context_embedding.vec'), 'r', encoding='utf-8') as context_f:
                    for line in context_f:
                        if len(line.strip()) > 0:
                            terms = line.strip().split('\t')
                            assert len(terms) == config.context_embedding_dim + 1, 'context embedding dim does not match'
                            WikidataId = terms[0]
                            if WikidataId in entity_dict:
                                context_embedding_vectors[entity_dict[WikidataId]] = torch.FloatTensor(list(map(float, terms[1:])))
            with open(entity_file, 'w', encoding='utf-8') as entity_f:
                json.dump(entity_dict, entity_f)
            with open(entity_embedding_file, 'wb') as entity_embedding_f:
                pickle.dump(entity_embedding_vectors, entity_embedding_f)
            with open(context_embedding_file, 'wb') as context_embedding_f:
                pickle.dump(context_embedding_vectors, context_embedding_f)

            content_vocab_num = config.num_concepts * config.max_history_num
            # 6. user history graph

            graph_size = content_vocab_num  # graph size of |V_{n}|+|V_{p}|
            prefix_mode = ['train', 'dev', 'test']
            user_history_graph_data = {}
            for prefix_index, prefix in enumerate([config.train_root, config.dev_root, config.test_root]):
                mode = prefix_mode[prefix_index]
                user_history_num = 0
                with open(os.path.join(prefix, 'behaviors.tsv'), 'r', encoding='utf-8') as behaviors_f:
                    for line in behaviors_f:
                        user_history_num += 1

                user_history_graph = np.zeros([user_history_num, graph_size, graph_size], dtype=np.float32)

                graph = pickle.load(open(config.DATA_NAME + "/"+mode+"/concepts_subgraph.pkl", 'rb'))   # build by 02_build_edge_index.py

                with open(os.path.join(prefix, 'behaviors.tsv'), 'r', encoding='utf-8') as behaviors_f:
                    for line_index, line in enumerate(behaviors_f):
                        impression_ID, user_ID, time, history, impressions = line.split('\t')
                        if config.no_self_connection:
                            history_graph = np.zeros([graph_size, graph_size], dtype=np.float32)
                        else:
                            history_graph = np.identity(graph_size, dtype=np.float16)

                        if len(history.strip()) > 0:
                            history_news_ID = history.split(' ')
                            offset = max(0, len(history_news_ID) - config.max_history_num)
                            history_news_num = min(len(history_news_ID), config.max_history_num)
                            history_news_contents = []
                            history_news_content_vocabs = []
                            for i in range(history_news_num):
                                content_ith_news = news_content_dict[history_news_ID[i + offset]]
                                history_news_contents.append(content_ith_news)
                            for m in history_news_contents:
                                for n in m:
                                    history_news_content_vocabs.append(n)
                            print(len(history_news_content_vocabs))
                            repeated_times = content_vocab_num - len(history_news_content_vocabs)
                            if repeated_times >= 0:
                                history_news_content_vocabs = history_news_content_vocabs + ['0']*repeated_times
                            else:
                                history_news_content_vocabs = history_news_content_vocabs[:content_vocab_num]

                            for v in range(len(history_news_content_vocabs)):
                                for w in range(len(history_news_content_vocabs[v+1:])):
                                    c1 = history_news_content_vocabs[v]
                                    c2 = history_news_content_vocabs[w]
                                    if graph.get_edge_data(c1, c2) is not None:
                                        history_graph[v, w] = 1

                            if not config.no_adjacent_normalization:
                                if config.gcn_normalization_type == 'asymmetric':
                                    # Asymmetric adjacent matrix normalization: D^{-\frac{1}{2}}A
                                    D_inv = np.zeros([graph_size, graph_size], dtype=np.float32)
                                    np.fill_diagonal(D_inv, 1 / history_graph.sum(axis=1, keepdims=False))
                                    history_graph = np.matmul(D_inv, history_graph)
                                else:
                                    # Symmetric adjacent matrix normalization: D^{-\frac{1}{2}}AD^{-\frac{1}{2}}
                                    D_inv_sqrt = np.zeros([graph_size, graph_size], dtype=np.float32)
                                    np.fill_diagonal(D_inv_sqrt,
                                                     np.sqrt(1 / history_graph.sum(axis=1, keepdims=False)))
                                    history_graph = np.matmul(np.matmul(D_inv_sqrt, history_graph), D_inv_sqrt)
                        user_history_graph[line_index] = history_graph
                    user_history_graph_data[mode + '_user_history_graph'] = user_history_graph

            with open(user_history_graph_file, 'wb') as user_history_graph_f:
                pickle.dump(user_history_graph_data, user_history_graph_f)


    def __init__(self, config: Config):
        # preprocess data
        MIND_Corpus_IPNR.preprocess(config)
        with open('cache/IPNR/user_ID-%s.json' % config.dataset_size, 'r', encoding='utf-8') as user_ID_f:
            self.user_ID_dict = json.load(user_ID_f)
            config.user_num = len(self.user_ID_dict)
        with open('cache/IPNR/news_ID-%s.json' % config.dataset_size, 'r', encoding='utf-8') as news_ID_f:
            self.news_ID_dict = json.load(news_ID_f)
            self.news_num = len(self.news_ID_dict)
        with open('cache/IPNR/category-%s.json' % config.dataset_size, 'r', encoding='utf-8') as category_f:
            self.category_dict = json.load(category_f)
            config.category_num = len(self.category_dict)
        with open('cache/IPNR/subCategory-%s.json' % config.dataset_size, 'r', encoding='utf-8') as subCategory_f:
            self.subCategory_dict = json.load(subCategory_f)
            config.subCategory_num = len(self.subCategory_dict)
        with open('cache/IPNR/vocabulary-' + str(config.word_threshold) + '-' + config.tokenizer + '-' + str(config.max_title_length) + '-' + str(config.max_abstract_length) + '-' + config.dataset_size + '.json', 'r', encoding='utf-8') as vocabulary_f:
            self.word_dict = json.load(vocabulary_f)
            config.vocabulary_size = len(self.word_dict)
        with open('cache/IPNR/entity-%s.json' % config.dataset_size, 'r', encoding='utf-8') as entity_f:
            self.entity_dict = json.load(entity_f)
            config.entity_size = len(self.entity_dict)
        with open('cache/IPNR/user_history_graph-' + str(config.max_history_num) + ('' if config.no_self_connection else '-self') + ('' if config.no_adjacent_normalization else '-normalize-' + config.gcn_normalization_type) + '-' + config.dataset_size + '.pkl', 'rb') as user_history_graph_f:
            user_history_data = pickle.load(user_history_graph_f)
            self.train_user_history_graph = user_history_data['train_user_history_graph']
            # self.train_user_history_category_mask = user_history_data['train_user_history_category_mask']
            # self.train_user_history_category_indices = user_history_data['train_user_history_category_indices']
            self.dev_user_history_graph = user_history_data['dev_user_history_graph']
            # self.dev_user_history_category_mask = user_history_data['dev_user_history_category_mask']
            # self.dev_user_history_category_indices = user_history_data['dev_user_history_category_indices']
            self.test_user_history_graph = user_history_data['test_user_history_graph']
            # self.test_user_history_category_mask = user_history_data['test_user_history_category_mask']
            # self.test_user_history_category_indices = user_history_data['test_user_history_category_indices']

        # meta data
        self.negative_sample_num = config.negative_sample_num                                           # negative sample number for training
        self.max_history_num = config.max_history_num                                                   # max history number for each training user
        self.max_title_length = config.max_title_length                                                 # max title length for each news text
        self.max_abstract_length = config.max_abstract_length                                           # max abstract length for each news text
        self.news_category = np.zeros([self.news_num], dtype=np.int32)                                  # [news_num]
        self.news_subCategory = np.zeros([self.news_num], dtype=np.int32)                               # [news_num]
        self.news_title_text = np.zeros([self.news_num, self.max_title_length], dtype=np.int32)         # [news_num, max_title_length]
        self.news_title_mask = np.zeros([self.news_num, self.max_title_length], dtype=bool)             # [news_num, max_title_length]
        self.news_title_entity = np.zeros([self.news_num, self.max_title_length], dtype=np.int32)       # [news_num, max_title_length]
        self.news_abstract_text = np.zeros([self.news_num, self.max_abstract_length], dtype=np.int32)   # [news_num, max_abstract_length]
        self.news_abstract_mask = np.zeros([self.news_num, self.max_abstract_length], dtype=bool)       # [news_num, max_abstract_length]
        self.news_abstract_entity = np.zeros([self.news_num, self.max_abstract_length], dtype=np.int32) # [news_num, max_abstract_length]
        self.train_behaviors = []                                                                       # [user_ID, [history], [history_mask], click impression, [non-click impressions], behavior_index]
        self.dev_behaviors = []                                                                         # [user_ID, [history], [history_mask], candidate_news_ID, behavior_index]
        self.dev_indices = []                                                                           # index for dev
        self.test_behaviors = []                                                                        # [user_ID, [history], [history_mask], candidate_news_ID, behavior_index]
        self.test_indices = []                                                                          # index for test
        self.title_word_num = 0
        self.abstract_word_num = 0

        self.max_concept_num = config.concept_num_per_news
        self.news_concept_text = np.zeros([self.news_num, self.max_concept_num], dtype=np.int32)
        self.news_concept_mask = np.zeros([self.news_num, self.max_concept_num], dtype=bool)
        self.concept_word_num = 0

        # generate news meta data
        news_ID_set = set(['<PAD>'])
        news_lines = []
        with open(os.path.join(config.train_root, 'news_.tsv'), 'r', encoding='utf-8') as train_news_f:
            for line in train_news_f:
                news_ID, category, subCategory, title, abstract, title_entities, abstract_entities, content = line.split('\t')
                if news_ID not in news_ID_set:
                    news_lines.append(line)
                    news_ID_set.add(news_ID)
        with open(os.path.join(config.dev_root, 'news_.tsv'), 'r', encoding='utf-8') as dev_news_f:
            for line in dev_news_f:
                news_ID, category, subCategory, title, abstract, title_entities, abstract_entities, content = line.split('\t')
                if news_ID not in news_ID_set:
                    news_lines.append(line)
                    news_ID_set.add(news_ID)
        with open(os.path.join(config.test_root, 'news_.tsv'), 'r', encoding='utf-8') as test_news_f:
            for line in test_news_f:
                news_ID, category, subCategory, title, abstract, title_entities, abstract_entities, content = line.split('\t')
                if news_ID not in news_ID_set:
                    news_lines.append(line)
                    news_ID_set.add(news_ID)
        assert self.news_num == len(news_ID_set), 'news num mismatch %d v.s. %d' % (self.news_num, len(news_ID_set))
        for line in news_lines:
            news_ID, category, subCategory, title, abstract, title_entities, abstract_entities, content = line.split('\t')
            index = self.news_ID_dict[news_ID]
            self.news_category[index] = self.category_dict[category] if category in self.category_dict else 0
            self.news_subCategory[index] = self.subCategory_dict[subCategory] if subCategory in self.subCategory_dict else 0
            words = pat.findall(title.lower()) if config.tokenizer == 'MIND' else word_tokenize(title.lower())
            offsets = [-1 for _ in range(len(title))]
            offset_index = 0
            for i, word in enumerate(words):
                if i == self.max_title_length:
                    break
                if is_number(word):
                    self.news_title_text[index][i] = self.word_dict['<NUM>']
                elif word in self.word_dict:
                    self.news_title_text[index][i] = self.word_dict[word]
                else:
                    self.news_title_text[index][i] = 1
                self.news_title_mask[index][i] = 1
                while title[offset_index] in [' ', '\t']:
                    offset_index += 1
                for j in range(len(word)):
                    offsets[offset_index] = i
                    offset_index += 1
            for entity in json.loads(title_entities):
                WikidataId = entity['WikidataId']
                for offset in entity['OccurrenceOffsets']:
                    if offsets[offset] != -1 and WikidataId in self.entity_dict:
                        self.news_title_entity[index][offsets[offset]] = self.entity_dict[WikidataId]
            self.title_word_num += len(words)
            words = pat.findall(abstract.lower()) if config.tokenizer == 'MIND' else word_tokenize(abstract.lower())
            offsets = [-1 for _ in range(len(abstract))]
            offset_index = 0
            for i, word in enumerate(words):
                if i == self.max_abstract_length:
                    break
                if is_number(word):
                    self.news_abstract_text[index][i] = self.word_dict['<NUM>']
                elif word in self.word_dict:
                    self.news_abstract_text[index][i] = self.word_dict[word]
                else:
                    self.news_abstract_text[index][i] = 1
                self.news_abstract_mask[index][i] = 1
                while abstract[offset_index] in [' ', '\t']:
                    offset_index += 1
                for j in range(len(word)):
                    offsets[offset_index] = i
                    offset_index += 1
            for entity in json.loads(abstract_entities):
                WikidataId = entity['WikidataId']
                for offset in entity['OccurrenceOffsets']:
                    if offsets[offset] != -1 and WikidataId in self.entity_dict:
                        self.news_abstract_entity[index][offsets[offset]] = self.entity_dict[WikidataId]
            self.abstract_word_num += len(words)

            # content
            words = pat.findall(content.lower()) if config.tokenizer == 'MIND' else word_tokenize(content.lower())
            offsets = [-1 for _ in range(len(content))]
            offset_index = 0
            for i, word in enumerate(words):
                if i == self.max_concept_num:
                    break
                if is_number(word):
                    self.news_concept_text[index][i] = self.word_dict['<NUM>']
                elif word in self.word_dict:
                    self.news_concept_text[index][i] = self.word_dict[word]
                else:
                    self.news_concept_text[index][i] = 1
                self.news_concept_mask[index][i] = 1
                while content[offset_index] in [' ', '\t']:
                    offset_index += 1
                for j in range(len(word)):
                    offsets[offset_index] = i
                    offset_index += 1
            self.concept_word_num += len(words)
        self.news_concept_mask[0][0] = 1    # for <PAD> news
        self.news_title_mask[0][0] = 1  # for <PAD> news
        self.news_abstract_mask[0][0] = 1  # for <PAD> news

        # generate behavior meta data
        with open(os.path.join(config.train_root, 'behaviors.tsv'), 'r', encoding='utf-8') as train_behaviors_f:
            for behavior_index, line in enumerate(train_behaviors_f):
                impression_ID, user_ID, time, history, impressions = line.split('\t')
                click_impressions = []
                non_click_impressions = []
                for impression in impressions.strip().split(' '):
                    if impression[-2:] == '-1':
                        click_impressions.append(self.news_ID_dict[impression[:-2]])
                    else:
                        non_click_impressions.append(self.news_ID_dict[impression[:-2]])
                if len(history) != 0:
                    history = list(map(lambda x: self.news_ID_dict[x], history.strip().split(' ')))
                    padding_num = max(0, self.max_history_num - len(history))
                    user_history = history[-self.max_history_num:] + [0] * padding_num
                    user_history_mask = np.zeros([self.max_history_num], dtype=bool)
                    user_history_mask[:min(len(history), self.max_history_num)] = 1
                    for click_impression in click_impressions:
                        self.train_behaviors.append([self.user_ID_dict[user_ID], user_history, user_history_mask, click_impression, non_click_impressions, behavior_index])
                else:
                    for click_impression in click_impressions:
                        self.train_behaviors.append([self.user_ID_dict[user_ID], [0 for _ in range(self.max_history_num)], np.zeros([self.max_history_num], dtype=bool), click_impression, non_click_impressions, behavior_index])
        with open(os.path.join(config.dev_root, 'behaviors.tsv'), 'r', encoding='utf-8') as dev_behaviors_f:
            for dev_ID, line in enumerate(dev_behaviors_f):
                impression_ID, user_ID, time, history, impressions = line.split('\t')
                if len(history) != 0:
                    history = list(map(lambda x: self.news_ID_dict[x], history.strip().split(' ')))
                    padding_num = max(0, self.max_history_num - len(history))
                    user_history = history[-self.max_history_num:] + [0] * padding_num
                    user_history_mask = np.zeros([self.max_history_num], dtype=bool)
                    user_history_mask[:min(len(history), self.max_history_num)] = 1
                    for impression in impressions.strip().split(' '):
                        self.dev_indices.append(dev_ID)
                        self.dev_behaviors.append([self.user_ID_dict[user_ID] if user_ID in self.user_ID_dict else 0, user_history, user_history_mask, self.news_ID_dict[impression[:-2]], dev_ID])
                else:
                    for impression in impressions.strip().split(' '):
                        self.dev_indices.append(dev_ID)
                        self.dev_behaviors.append([self.user_ID_dict[user_ID] if user_ID in self.user_ID_dict else 0, [0 for _ in range(self.max_history_num)], np.zeros([self.max_history_num], dtype=bool), self.news_ID_dict[impression[:-2]], dev_ID])
        with open(os.path.join(config.test_root, 'behaviors.tsv'), 'r', encoding='utf-8') as test_behaviors_f:
            for test_ID, line in enumerate(test_behaviors_f):
                impression_ID, user_ID, time, history, impressions = line.split('\t')
                if len(history) != 0:
                    history = list(map(lambda x: self.news_ID_dict[x], history.strip().split(' ')))
                    padding_num = max(0, self.max_history_num - len(history))
                    user_history = history[-self.max_history_num:] + [0] * padding_num
                    user_history_mask = np.zeros([self.max_history_num], dtype=bool)
                    user_history_mask[:min(len(history), self.max_history_num)] = 1
                    for impression in impressions.strip().split(' '):
                        self.test_indices.append(test_ID)
                        if config.dataset_size != 'submission':
                            self.test_behaviors.append([self.user_ID_dict[user_ID] if user_ID in self.user_ID_dict else 0, user_history, user_history_mask, self.news_ID_dict[impression[:-2]], test_ID])
                        else:
                            self.test_behaviors.append([self.user_ID_dict[user_ID] if user_ID in self.user_ID_dict else 0, user_history, user_history_mask, self.news_ID_dict[impression], test_ID])
                else:
                    for impression in impressions.strip().split(' '):
                        self.test_indices.append(test_ID)
                        if config.dataset_size != 'submission':
                            self.test_behaviors.append([self.user_ID_dict[user_ID] if user_ID in self.user_ID_dict else 0, [0 for _ in range(self.max_history_num)], np.zeros([self.max_history_num], dtype=bool), self.news_ID_dict[impression[:-2]], test_ID])
                        else:
                            self.test_behaviors.append([self.user_ID_dict[user_ID] if user_ID in self.user_ID_dict else 0, [0 for _ in range(self.max_history_num)], np.zeros([self.max_history_num], dtype=bool), self.news_ID_dict[impression], test_ID])



import time
from config import Config
import torch.utils.data as data
from numpy.random import randint
from torch.utils.data import DataLoader


class MIND_Train_Dataset_IPNR(data.Dataset):
    def __init__(self, corpus: MIND_Corpus_IPNR):
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
        # self.user_history_category_mask = corpus.train_user_history_category_mask
        # self.user_history_category_indices = corpus.train_user_history_category_indices
        self.train_behaviors = corpus.train_behaviors
        self.train_samples = [[0 for _ in range(1 + self.negative_sample_num)] for __ in
                              range(len(self.train_behaviors))]
        self.num = len(self.train_behaviors)

        self.news_concept_text = corpus.news_concept_text
        self.news_concept_mask = corpus.news_concept_mask

    def negative_sampling(self, rank=None):
        print('\n%sBegin negative sampling, training sample num : %d' % (
        '' if rank is None else ('rank ' + str(rank) + ' : '), self.num))
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
        end_time = time.time()
        print('%sEnd negative sampling, used time : %.3fs' % (
        '' if rank is None else ('rank ' + str(rank) + ' : '), end_time - start_time))

    def __getitem__(self, index):
        train_behavior = self.train_behaviors[index]
        history_index = train_behavior[1]
        sample_index = self.train_samples[index]
        behavior_index = train_behavior[5]
        return train_behavior[0], self.news_category[history_index], self.news_subCategory[history_index], \
               self.news_title_text[history_index], self.news_title_mask[history_index], self.news_title_entity[
                   history_index], self.news_abstract_text[history_index], self.news_abstract_mask[history_index], \
               self.news_abstract_entity[history_index], train_behavior[2], self.user_history_graph[behavior_index], \
               self.news_concept_text[history_index], self.news_concept_mask[history_index], \
               self.news_category[sample_index], self.news_subCategory[sample_index], self.news_title_text[
                   sample_index], self.news_title_mask[sample_index], self.news_title_entity[sample_index], \
               self.news_abstract_text[sample_index], self.news_abstract_mask[sample_index], self.news_abstract_entity[
                   sample_index], self.news_concept_text[sample_index], self.news_concept_mask[sample_index]

    def __len__(self):
        return self.num


class MIND_DevTest_Dataset_IPNR(data.Dataset):
    def __init__(self, corpus: MIND_Corpus_IPNR, mode: str):
        assert mode in ['dev', 'test'], 'mode must be chosen from \'dev\' or \'test\''
        self.news_category = corpus.news_category
        self.news_subCategory = corpus.news_subCategory
        self.news_title_text = corpus.news_title_text
        self.news_title_mask = corpus.news_title_mask
        self.news_title_entity = corpus.news_title_entity
        self.news_abstract_text = corpus.news_abstract_text
        self.news_abstract_mask = corpus.news_abstract_mask
        self.news_abstract_entity = corpus.news_abstract_entity
        self.user_history_graph = corpus.dev_user_history_graph if mode == 'dev' else corpus.test_user_history_graph
        # self.user_history_category_mask = corpus.dev_user_history_category_mask if mode == 'dev' else corpus.test_user_history_category_mask
        # self.user_history_category_indices = corpus.dev_user_history_category_indices if mode == 'dev' else corpus.test_user_history_category_indices
        self.behaviors = corpus.dev_behaviors if mode == 'dev' else corpus.test_behaviors
        self.num = len(self.behaviors)

        self.news_concept_text = corpus.news_concept_text
        self.news_concept_mask = corpus.news_concept_mask

    def __getitem__(self, index):
        behavior = self.behaviors[index]
        history_index = behavior[1]
        candidate_news_index = behavior[3]
        behavior_index = behavior[4]
        return behavior[0], self.news_category[history_index], self.news_subCategory[history_index], \
               self.news_title_text[history_index], self.news_title_mask[history_index], self.news_title_entity[
                   history_index], self.news_abstract_text[history_index], self.news_abstract_mask[history_index], \
               self.news_abstract_entity[history_index], behavior[2], self.user_history_graph[behavior_index], \
               self.news_concept_text[history_index], self.news_concept_mask[history_index], \
               self.news_category[candidate_news_index], self.news_subCategory[candidate_news_index], \
               self.news_title_text[candidate_news_index], self.news_title_mask[candidate_news_index], \
               self.news_title_entity[candidate_news_index], self.news_abstract_text[candidate_news_index], \
               self.news_abstract_mask[candidate_news_index], self.news_abstract_entity[candidate_news_index], \
               self.news_concept_text[candidate_news_index], self.news_concept_mask[candidate_news_index]

    def __len__(self):
        return self.num
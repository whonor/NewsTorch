"""Unified Adressa-1week corpus.

The returned training and evaluation tuples intentionally match the extended
EB-NeRD contract.  This lets every NewsTorch model share one Adressa adapter;
modalities absent from Adressa (images and calibrated sentiment) are explicit
zero-valued ablations rather than missing fields.
"""

import collections
import json
import os
import pickle
import re

import numpy as np
import torch

from dataset_corpus_preprocessing.EBNeRD_corpus_main import (
    Ebnerd_DevTest_Dataset,
    Ebnerd_Train_Dataset,
)
from dataset_download_prepare.Adressa_dataset_prepare import is_prepared


TOKEN_PATTERN = re.compile(r"[\w]+|[.,!?;|]")


def _tokens(text):
    return TOKEN_PATTERN.findall(str(text or "").lower())


def _is_number(value):
    try:
        float(value)
        return True
    except ValueError:
        return False


def _read_jsonl(path):
    with open(path, "r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON in {path}:{line_number}: {error}") from error


def adressa_knowledge_nodes(record):
    """Build linked knowledge nodes from metadata present in Adressa-1week."""
    nodes = []
    seen = set()
    fields = (
        ("entities", "entity"),
        ("keywords", "keyword"),
        ("category", "category"),
        ("subcategory", "subcategory"),
    )
    for field, prefix in fields:
        values = record.get(field, [])
        if not isinstance(values, (list, tuple, set)):
            values = [values]
        for value in values:
            value = str(value or "").strip()
            if not value:
                continue
            node = f"{prefix}:{value.casefold()}"
            if node not in seen:
                seen.add(node)
                nodes.append(node)
    return nodes


def _embedding_file(config, cache_dir):
    return os.path.join(
        cache_dir,
        "word_embedding-{}-{}-{}-{}-{}-{}.pkl".format(
            config.word_threshold,
            config.word_embedding_dim,
            config.tokenizer,
            config.max_title_length,
            config.max_abstract_length,
            config.dataset_size,
        ),
    )


class _HistoryGraphStore:
    """Compute a CNE-SUE/CNRCL history graph only when it is requested."""

    def __init__(self, corpus, behaviors, component, index_field):
        self.corpus = corpus
        self.component = component
        self.histories = {}
        for behavior in behaviors:
            self.histories.setdefault(int(behavior[index_field]), behavior[1])

    def __len__(self):
        return len(self.histories)

    def __getitem__(self, index):
        history = self.histories[int(index)]
        config = self.corpus.config
        category_num = config.category_num
        history_num = config.max_history_num
        graph_size = history_num + category_num
        graph = (
            np.zeros((graph_size, graph_size), dtype=np.float32)
            if config.no_self_connection
            else np.eye(graph_size, dtype=np.float32)
        )
        category_mask = np.zeros(category_num + 1, dtype=np.float32)
        category_indices = np.zeros(history_num, dtype=np.int64)
        real_positions = [position for position, news_index in enumerate(history) if news_index != 0]
        for position in real_positions:
            category = int(self.corpus.news_category[history[position]])
            category_indices[position] = category
            category_mask[category] = 1.0
            category_node = history_num + category
            graph[position, category_node] = 1.0
            graph[category_node, position] = 1.0
        for offset, left in enumerate(real_positions):
            left_category = int(category_indices[left])
            for right in real_positions[offset + 1:]:
                right_category = int(category_indices[right])
                if left_category == right_category:
                    graph[left, right] = graph[right, left] = 1.0
                else:
                    left_node = history_num + left_category
                    right_node = history_num + right_category
                    graph[left_node, right_node] = graph[right_node, left_node] = 1.0
        if not config.no_adjacent_normalization:
            degree = np.clip(graph.sum(axis=1), 1e-12, None)
            if config.gcn_normalization_type == "asymmetric":
                graph = (1.0 / degree)[:, None] * graph
            else:
                scale = np.sqrt(1.0 / degree)
                graph = scale[:, None] * graph * scale[None, :]
        if self.component == "graph":
            return graph.astype(np.float32, copy=False)
        if self.component == "mask":
            return category_mask
        return category_indices


class Adressa_Corpus:
    def __init__(self, config):
        self.config = config
        dataset_root = os.path.join(config.root, config.DATASET_ROOT)
        if not is_prepared(dataset_root):
            raise FileNotFoundError(
                f"Adressa-1week is not prepared under {dataset_root}. Run "
                "`python dataset_download_prepare/Adressa_dataset_prepare.py "
                f"--dataset-root {dataset_root}` after placing the licensed files in its raw directory."
            )
        self._load_records(config)
        self._build_dictionaries(config)
        self._build_news_arrays(config)
        self._build_behaviors(config)
        del self.split_behavior_records
        self._build_popularity()
        self._install_lazy_graphs()
        self._write_model_caches(config)

    def _load_records(self, config):
        records = {}
        for split in ("train", "dev", "test"):
            path = os.path.join(config.root, config.DATASET_ROOT, split, "news.jsonl")
            for record in _read_jsonl(path):
                records.setdefault(str(record["nid"]), record)
        self.news_records = records
        self.split_behavior_records = {
            split: list(_read_jsonl(os.path.join(
                config.root, config.DATASET_ROOT, split, "behaviors.jsonl"
            )))
            for split in ("train", "dev", "test")
        }

    def _build_dictionaries(self, config):
        self.news_ID_dict = {"<PAD>": 0}
        for news_id in sorted(self.news_records):
            self.news_ID_dict[news_id] = len(self.news_ID_dict)
        self.news_num = len(self.news_ID_dict)
        config.news_num = self.news_num

        self.user_ID_dict = {"<UNK>": 0}
        for split in ("train", "dev", "test"):
            for behavior in self.split_behavior_records[split]:
                user_id = str(behavior["uid"])
                if user_id not in self.user_ID_dict:
                    self.user_ID_dict[user_id] = len(self.user_ID_dict)
        config.user_num = len(self.user_ID_dict)

        self.category_dict = {"<PAD>": 0, "unknown": 1}
        self.subCategory_dict = {"<PAD>": 0, "unknown": 1}
        entity_values = {"<PAD>", "<UNK>"}
        counter = collections.Counter()
        for record in self.news_records.values():
            category = str(record.get("category") or "unknown")
            subcategory = str(record.get("subcategory") or "unknown")
            if category not in self.category_dict:
                self.category_dict[category] = len(self.category_dict)
            if subcategory not in self.subCategory_dict:
                self.subCategory_dict[subcategory] = len(self.subCategory_dict)
            counter.update(_tokens(record.get("title")))
            counter.update(_tokens(record.get("abstract")))
            entity_values.update(adressa_knowledge_nodes(record))
        config.category_num = len(self.category_dict)
        config.subCategory_num = len(self.subCategory_dict)

        self.word_dict = {"<PAD>": 0, "<UNK>": 1, "<NUM>": 2}
        for word, count in sorted(counter.items(), key=lambda item: (-item[1], item[0])):
            if count >= config.word_threshold and word not in self.word_dict:
                self.word_dict[word] = len(self.word_dict)
        config.vocabulary_size = len(self.word_dict)
        self.entity_dict = {"<PAD>": 0, "<UNK>": 1}
        for entity in sorted(entity_values - {"<PAD>", "<UNK>"}):
            self.entity_dict[entity] = len(self.entity_dict)
        config.entity_size = len(self.entity_dict)
        self.sentiment_dict = {"Neutral": 0}
        self.sentiment_label_dict = {"Neutral": 0}
        config.num_sent_classes = 3

    def _encode_text(self, value, length):
        text = np.zeros(length, dtype=np.int32)
        mask = np.zeros(length, dtype=np.float32)
        for offset, token in enumerate(_tokens(value)[:length]):
            text[offset] = self.word_dict["<NUM>"] if _is_number(token) else self.word_dict.get(token, 1)
            mask[offset] = 1.0
        return text, mask

    def _build_news_arrays(self, config):
        n = self.news_num
        title_length = config.max_title_length
        abstract_length = config.max_abstract_length
        self.news_category = np.zeros(n, dtype=np.int64)
        self.news_subCategory = np.zeros(n, dtype=np.int64)
        self.news_sentiment = np.zeros(n, dtype=np.float32)
        self.news_title_text = np.zeros((n, title_length), dtype=np.int32)
        self.news_title_mask = np.zeros((n, title_length), dtype=np.float32)
        self.news_title_entity = np.zeros((n, title_length), dtype=np.int32)
        self.news_abstract_text = np.zeros((n, abstract_length), dtype=np.int32)
        self.news_abstract_mask = np.zeros((n, abstract_length), dtype=np.float32)
        self.news_abstract_entity = np.zeros((n, abstract_length), dtype=np.int32)
        self.news_title_word_pop = np.zeros((n, title_length), dtype=np.int64)
        self.news_title_entity_pop = np.zeros((n, title_length), dtype=np.int64)
        self.news_concept_text = np.zeros((n, config.concept_num_per_news), dtype=np.int32)
        self.news_concept_mask = np.zeros((n, config.concept_num_per_news), dtype=np.float32)

        for news_id, record in self.news_records.items():
            index = self.news_ID_dict[news_id]
            category = str(record.get("category") or "unknown")
            subcategory = str(record.get("subcategory") or "unknown")
            self.news_category[index] = self.category_dict.get(category, 1)
            self.news_subCategory[index] = self.subCategory_dict.get(subcategory, 1)
            title, title_mask = self._encode_text(record.get("title"), title_length)
            abstract, abstract_mask = self._encode_text(record.get("abstract"), abstract_length)
            self.news_title_text[index] = title
            self.news_title_mask[index] = title_mask
            self.news_abstract_text[index] = abstract
            self.news_abstract_mask[index] = abstract_mask
            for offset, entity in enumerate(
                adressa_knowledge_nodes(record)[:title_length]
            ):
                self.news_title_entity[index, offset] = self.entity_dict.get(entity, 1)
            concept_tokens = list(title[title_mask.astype(bool)])[:config.concept_num_per_news]
            self.news_concept_text[index, :len(concept_tokens)] = concept_tokens
            self.news_concept_mask[index, :len(concept_tokens)] = 1.0
        self.news_title_mask[0, 0] = 1.0
        self.news_abstract_mask[0, 0] = 1.0
        if config.model.lower() == "mmrec":
            self.news_image_embeddings = np.zeros(
                (n, config.image_embedding_dim), dtype=np.float32
            )

    def _history(self, values, max_history):
        indices = [self.news_ID_dict.get(str(value), 0) for value in values][-max_history:]
        mask = np.zeros(max_history, dtype=np.float32)
        mask[:len(indices)] = 1.0
        return indices + [0] * (max_history - len(indices)), mask

    @staticmethod
    def _read_times(values, max_history):
        values = [max(0.0, float(value or 0.0)) for value in values][-max_history:]
        return values + [0.0] * (max_history - len(values))

    def _build_behaviors(self, config):
        self.train_behaviors = []
        self.dev_behaviors = []
        self.test_behaviors = []
        self.dev_indices = []
        self.test_indices = []
        for split in ("train", "dev", "test"):
            target = getattr(self, f"{split}_behaviors")
            for row_index, row in enumerate(self.split_behavior_records[split]):
                history, history_mask = self._history(row.get("history", []), config.max_history_num)
                read_times = self._read_times(row.get("history_read_time", []), config.max_history_num)
                candidates = [self.news_ID_dict[str(value)] for value in row["candidates"]]
                labels = [int(value) for value in row["labels"]]
                user_index = self.user_ID_dict.get(str(row["uid"]), 0)
                if split == "train":
                    negatives = [candidate for candidate, label in zip(candidates, labels) if label == 0]
                    if not negatives:
                        continue
                    candidate_times = row.get("candidate_read_time", [])
                    for position, (candidate, label) in enumerate(zip(candidates, labels)):
                        if label == 1:
                            behavior_index = len(target)
                            candidate_time = (
                                float(candidate_times[position])
                                if position < len(candidate_times)
                                else float(row.get("next_read_time", 0.0))
                            )
                            target.append([
                                user_index, history, history_mask, candidate, negatives,
                                behavior_index, read_times, candidate_time, 0.0,
                            ])
                else:
                    behavior_index = row_index
                    indices = self.dev_indices if split == "dev" else self.test_indices
                    candidate_times = row.get("candidate_read_time", [])
                    for position, candidate in enumerate(candidates):
                        indices.append(row_index)
                        candidate_time = float(candidate_times[position]) if position < len(candidate_times) else 0.0
                        target.append([
                            user_index, history, history_mask, candidate, behavior_index,
                            read_times, candidate_time, 0.0,
                        ])
        self.negative_sample_num = config.negative_sample_num
        self.max_history_num = config.max_history_num
        self.max_title_length = config.max_title_length
        self.max_abstract_length = config.max_abstract_length

    def _build_popularity(self):
        word_count = collections.Counter()
        entity_count = collections.Counter()
        for behavior in self.train_behaviors:
            positive = behavior[3]
            word_count.update(int(value) for value in self.news_title_text[positive] if value)
            entity_count.update(int(value) for value in self.news_title_entity[positive] if value)
        word_max = max(word_count.values(), default=1)
        entity_max = max(entity_count.values(), default=1)
        for news_index in range(self.news_num):
            for offset, word in enumerate(self.news_title_text[news_index]):
                self.news_title_word_pop[news_index, offset] = min(199, int(199 * word_count[word] / word_max)) if word else 0
            for offset, entity in enumerate(self.news_title_entity[news_index]):
                self.news_title_entity_pop[news_index, offset] = min(199, int(199 * entity_count[entity] / entity_max)) if entity else 0

    def _install_lazy_graphs(self):
        for split in ("train", "dev", "test"):
            behaviors = getattr(self, f"{split}_behaviors")
            index_field = 5 if split == "train" else 4
            setattr(self, f"{split}_user_history_graph", _HistoryGraphStore(self, behaviors, "graph", index_field))
            setattr(self, f"{split}_user_history_category_mask", _HistoryGraphStore(self, behaviors, "mask", index_field))
            setattr(self, f"{split}_user_history_category_indices", _HistoryGraphStore(self, behaviors, "indices", index_field))

    def _write_model_caches(self, config):
        cache_dir = os.path.join("cache", "adressa")
        os.makedirs(cache_dir, exist_ok=True)
        dictionaries = {
            f"user_ID-{config.dataset_size}.json": self.user_ID_dict,
            f"news_ID-{config.dataset_size}.json": self.news_ID_dict,
            f"category-{config.dataset_size}.json": self.category_dict,
            f"subCategory-{config.dataset_size}.json": self.subCategory_dict,
            f"entity-{config.dataset_size}.json": self.entity_dict,
        }
        for name, value in dictionaries.items():
            with open(os.path.join(cache_dir, name), "w", encoding="utf-8") as output:
                json.dump(value, output, ensure_ascii=False)

        embedding_path = _embedding_file(config, cache_dir)
        if not os.path.exists(embedding_path):
            generator = torch.Generator().manual_seed(config.seed)
            embeddings = torch.empty(len(self.word_dict), config.word_embedding_dim)
            embeddings.normal_(mean=0.0, std=0.1, generator=generator)
            embeddings[0].zero_()
            with open(embedding_path, "wb") as output:
                pickle.dump(embeddings, output)
        with open(embedding_path, "rb") as source:
            word_embeddings = pickle.load(source)

        entity_embeddings = torch.zeros(len(self.entity_dict), config.entity_embedding_dim)
        context_embeddings = torch.zeros(len(self.entity_dict), config.context_embedding_dim)
        for directory in ("cache", cache_dir):
            with open(os.path.join(directory, f"entity_embedding-{config.dataset_size}.pkl"), "wb") as output:
                pickle.dump(entity_embeddings, output)
            with open(os.path.join(directory, f"context_embedding-{config.dataset_size}.pkl"), "wb") as output:
                pickle.dump(context_embeddings, output)

        # IPNR currently owns a separate embedding cache and concept prototype.
        ipnr_dir = os.path.join("cache", "IPNR")
        os.makedirs(ipnr_dir, exist_ok=True)
        ipnr_embedding_path = _embedding_file(config, ipnr_dir)
        if not os.path.exists(ipnr_embedding_path):
            with open(ipnr_embedding_path, "wb") as output:
                pickle.dump(word_embeddings, output)
        concept_path = os.path.join(config.root, config.DATASET_ROOT, "all_concept_word_embedding.npy")
        if not os.path.exists(concept_path):
            concept = np.zeros((config.num_concepts, config.word_embedding_dim), dtype=np.float32)
            if len(word_embeddings) > 1:
                concept[:] = torch.as_tensor(word_embeddings[1:]).mean(dim=0).numpy()
            np.save(concept_path, concept)


class Adressa_Train_Dataset(Ebnerd_Train_Dataset):
    def __getitem__(self, index):
        item = list(super().__getitem__(index))
        item[23] = np.asarray(item[23], dtype=np.int64)
        item[24] = np.asarray(item[24], dtype=np.int64)
        return tuple(item)


class Adressa_DevTest_Dataset(Ebnerd_DevTest_Dataset):
    def __getitem__(self, index):
        item = list(super().__getitem__(index))
        item[23] = np.asarray(item[23], dtype=np.int64)
        item[24] = np.asarray(item[24], dtype=np.int64)
        return tuple(item)


def _ipnr_graph(config):
    size = config.max_history_num * config.num_concepts
    return np.eye(size, dtype=np.float32)


class Adressa_Train_Dataset_IPNR(Adressa_Train_Dataset):
    def __init__(self, corpus):
        super().__init__(corpus)
        self.corpus = corpus

    def __getitem__(self, index):
        behavior = self.train_behaviors[index]
        history = behavior[1]
        candidates = self.train_samples[index]
        c = self.corpus
        return (
            behavior[0], c.news_category[history], c.news_subCategory[history],
            c.news_title_text[history], c.news_title_mask[history], c.news_title_entity[history],
            c.news_abstract_text[history], c.news_abstract_mask[history], c.news_abstract_entity[history],
            behavior[2], _ipnr_graph(c.config), c.news_concept_text[history], c.news_concept_mask[history],
            c.news_category[candidates], c.news_subCategory[candidates], c.news_title_text[candidates],
            c.news_title_mask[candidates], c.news_title_entity[candidates], c.news_abstract_text[candidates],
            c.news_abstract_mask[candidates], c.news_abstract_entity[candidates],
            c.news_concept_text[candidates], c.news_concept_mask[candidates],
        )


class Adressa_DevTest_Dataset_IPNR(Adressa_DevTest_Dataset):
    def __init__(self, corpus, mode):
        super().__init__(corpus, mode)
        self.corpus = corpus

    def __getitem__(self, index):
        behavior = self.behaviors[index]
        history = behavior[1]
        candidate = behavior[3]
        c = self.corpus
        return (
            behavior[0], c.news_category[history], c.news_subCategory[history],
            c.news_title_text[history], c.news_title_mask[history], c.news_title_entity[history],
            c.news_abstract_text[history], c.news_abstract_mask[history], c.news_abstract_entity[history],
            behavior[2], _ipnr_graph(c.config), c.news_concept_text[history], c.news_concept_mask[history],
            c.news_category[candidate], c.news_subCategory[candidate], c.news_title_text[candidate],
            c.news_title_mask[candidate], c.news_title_entity[candidate], c.news_abstract_text[candidate],
            c.news_abstract_mask[candidate], c.news_abstract_entity[candidate],
            c.news_concept_text[candidate], c.news_concept_mask[candidate],
        )

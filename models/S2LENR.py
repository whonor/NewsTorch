import json
import os
import re
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Config
from models.modules import newsEncoders
from models.modules.click_predictor import DotProduct
from models.modules.layers import Attention, MultiHeadAttention


_TOKEN_PATTERN = re.compile(r"[\w]+|[.,!?;|]")


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "default"


def _is_number(value: str) -> bool:
    try:
        float(value)
        return True
    except ValueError:
        return False


def _clean_text(value) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() == "nan" else text


class S2LENRGeneratedNewsStore:
    """Sparse generated-news cache keyed by NewsTorch user index."""

    def __init__(self, config: Config, corpus):
        self.config = config
        self.corpus = corpus
        self.max_generated_num = int(getattr(config, "s2lenr_generated_news_num", 5))
        self.max_title_length = config.max_title_length
        self.max_abstract_length = config.max_abstract_length
        self.word_dict = getattr(corpus, "word_dict", {})
        self.category_dict = getattr(corpus, "category_dict", {})
        self.subcategory_dict = getattr(corpus, "subCategory_dict", {})
        self.user_id_dict = getattr(corpus, "user_ID_dict", {})
        self.records: Dict[int, Dict[str, np.ndarray]] = {}
        self.path = self._resolve_path(config)
        self._load()

    @staticmethod
    def _resolve_path(config: Config) -> str:
        explicit_path = getattr(config, "s2lenr_generated_news_path", "")
        if explicit_path:
            return explicit_path
        llm_name = _safe_name(str(getattr(config, "s2lenr_llm_model", "Qwen/Qwen3-32B")))
        dataset_root = _safe_name(str(getattr(config, "DATASET_ROOT", "dataset")))
        return os.path.join("cache", "s2lenr", dataset_root, llm_name, "generated_news.jsonl")

    def _load(self):
        if self.max_generated_num <= 0:
            return
        if not os.path.exists(self.path):
            message = (
                f"S2LENR generated-news cache not found at {self.path}. "
                "Run scripts/generate_s2lenr_qwen_news.py first, or set "
                "s2lenr_require_generated_cache: false to train the ablated path."
            )
            if bool(getattr(self.config, "s2lenr_require_generated_cache", False)):
                raise FileNotFoundError(message)
            print("Warning: " + message)
            return

        loaded = 0
        with open(self.path, "r", encoding="utf-8") as cache_f:
            for line in cache_f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                user_index = self._resolve_user_index(record)
                if user_index is None:
                    continue
                articles = self._record_articles(record)
                encoded = self._encode_articles(articles)
                if encoded is None:
                    continue
                self.records[user_index] = encoded
                loaded += 1
        print(f"S2LENR loaded generated news for {loaded} users from {self.path}")

    def _resolve_user_index(self, record) -> Optional[int]:
        user_index = record.get("user_index")
        if user_index is not None:
            try:
                user_index = int(user_index)
            except (TypeError, ValueError):
                user_index = None
            if user_index is not None and user_index >= 0:
                return user_index

        raw_user_id = record.get("user_id_raw", record.get("user_id", record.get("uid")))
        raw_user_id = _clean_text(raw_user_id)
        if raw_user_id in self.user_id_dict:
            return int(self.user_id_dict[raw_user_id])
        return None

    @staticmethod
    def _record_articles(record):
        for key in ("generated_news", "generated", "articles", "news"):
            articles = record.get(key)
            if articles:
                return articles
        return []

    def _encode_articles(self, articles):
        if not isinstance(articles, list):
            return None
        selected = articles[: self.max_generated_num]
        if len(selected) == 0:
            return None

        title_text = np.zeros((self.max_generated_num, self.max_title_length), dtype=np.int64)
        title_mask = np.zeros((self.max_generated_num, self.max_title_length), dtype=np.float32)
        title_entity = np.zeros((self.max_generated_num, self.max_title_length), dtype=np.int64)
        abstract_text = np.zeros((self.max_generated_num, self.max_abstract_length), dtype=np.int64)
        abstract_mask = np.zeros((self.max_generated_num, self.max_abstract_length), dtype=np.float32)
        abstract_entity = np.zeros((self.max_generated_num, self.max_abstract_length), dtype=np.int64)
        category = np.zeros(self.max_generated_num, dtype=np.int64)
        subcategory = np.zeros(self.max_generated_num, dtype=np.int64)
        item_mask = np.zeros(self.max_generated_num, dtype=np.float32)

        for index, article in enumerate(selected):
            title, abstract, topic = self._article_fields(article)
            if not title and abstract:
                title = abstract
            if not title:
                continue
            self._fill_tokens(title, title_text[index], title_mask[index], self.max_title_length)
            self._fill_tokens(abstract, abstract_text[index], abstract_mask[index], self.max_abstract_length)
            category[index] = self.category_dict.get(topic, 0)
            subcategory[index] = self.subcategory_dict.get(topic, 0)
            item_mask[index] = 1.0

        if not item_mask.any():
            return None

        title_mask[item_mask == 0, 0] = 1.0
        abstract_mask[item_mask == 0, 0] = 1.0
        return {
            "title_text": title_text,
            "title_mask": title_mask,
            "title_entity": title_entity,
            "abstract_text": abstract_text,
            "abstract_mask": abstract_mask,
            "abstract_entity": abstract_entity,
            "category": category,
            "subcategory": subcategory,
            "item_mask": item_mask,
        }

    @staticmethod
    def _article_fields(article):
        if isinstance(article, str):
            return article, "", ""
        if not isinstance(article, dict):
            return "", "", ""
        title = _clean_text(article.get("title", article.get("Title")))
        abstract = _clean_text(article.get("abstract", article.get("Abstract")))
        topic = _clean_text(article.get("topic", article.get("Topic", article.get("category", ""))))
        return title, abstract, topic

    def _fill_tokens(self, text: str, out_text: np.ndarray, out_mask: np.ndarray, max_length: int):
        words = _TOKEN_PATTERN.findall(text.lower()) if self.config.tokenizer == "MIND" else text.lower().split()
        num_index = self.word_dict.get("<NUM>", self.word_dict.get("<UNK>", 1))
        unk_index = self.word_dict.get("<UNK>", 1)
        for index, word in enumerate(words[:max_length]):
            out_text[index] = num_index if _is_number(word) else self.word_dict.get(word, unk_index)
            out_mask[index] = 1.0
        if out_mask.sum() == 0:
            out_mask[0] = 1.0

    def lookup(self, user_ids: torch.Tensor, device: torch.device):
        batch_size = user_ids.size(0)
        shape_title = (batch_size, self.max_generated_num, self.max_title_length)
        shape_abstract = (batch_size, self.max_generated_num, self.max_abstract_length)
        title_text = torch.zeros(shape_title, dtype=torch.long, device=device)
        title_mask = torch.zeros(shape_title, dtype=torch.float32, device=device)
        title_entity = torch.zeros(shape_title, dtype=torch.long, device=device)
        abstract_text = torch.zeros(shape_abstract, dtype=torch.long, device=device)
        abstract_mask = torch.zeros(shape_abstract, dtype=torch.float32, device=device)
        abstract_entity = torch.zeros(shape_abstract, dtype=torch.long, device=device)
        category = torch.zeros((batch_size, self.max_generated_num), dtype=torch.long, device=device)
        subcategory = torch.zeros((batch_size, self.max_generated_num), dtype=torch.long, device=device)
        item_mask = torch.zeros((batch_size, self.max_generated_num), dtype=torch.float32, device=device)

        if self.max_generated_num == 0:
            return category, subcategory, title_text, title_mask, title_entity, abstract_text, abstract_mask, abstract_entity, item_mask

        title_mask[:, :, 0] = 1.0
        abstract_mask[:, :, 0] = 1.0
        for batch_index, user_index in enumerate(user_ids.detach().cpu().tolist()):
            record = self.records.get(int(user_index))
            if record is None:
                continue
            title_text[batch_index] = torch.as_tensor(record["title_text"], dtype=torch.long, device=device)
            title_mask[batch_index] = torch.as_tensor(record["title_mask"], dtype=torch.float32, device=device)
            title_entity[batch_index] = torch.as_tensor(record["title_entity"], dtype=torch.long, device=device)
            abstract_text[batch_index] = torch.as_tensor(record["abstract_text"], dtype=torch.long, device=device)
            abstract_mask[batch_index] = torch.as_tensor(record["abstract_mask"], dtype=torch.float32, device=device)
            abstract_entity[batch_index] = torch.as_tensor(record["abstract_entity"], dtype=torch.long, device=device)
            category[batch_index] = torch.as_tensor(record["category"], dtype=torch.long, device=device)
            subcategory[batch_index] = torch.as_tensor(record["subcategory"], dtype=torch.long, device=device)
            item_mask[batch_index] = torch.as_tensor(record["item_mask"], dtype=torch.float32, device=device)

        return category, subcategory, title_text, title_mask, title_entity, abstract_text, abstract_mask, abstract_entity, item_mask


class S2LENR(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.model_name = config.model
        self.batch_size = config.batch_size
        self.news_encoder = newsEncoders.MHSA(config)
        self.news_embedding_dim = self.news_encoder.news_embedding_dim
        self.generated_news_num = int(getattr(config, "s2lenr_generated_news_num", 5))
        self.prototype_num = int(getattr(config, "s2lenr_prototype_num", 1000))
        self.sequence_length = config.max_history_num + self.generated_news_num
        self.ssl_temperature = float(getattr(config, "s2lenr_ssl_temperature", 1.0))
        self.use_exact_ratio_loss = bool(getattr(config, "s2lenr_ssl_exact_ratio", False))

        self.user_self_attention = MultiHeadAttention(
            config.head_num,
            self.news_embedding_dim,
            self.sequence_length,
            self.sequence_length,
            config.head_dim,
            config.head_dim,
        )
        self.user_affine = nn.Linear(config.head_num * config.head_dim, self.news_embedding_dim, bias=True)
        self.user_attention = Attention(self.news_embedding_dim, config.attention_dim)
        self.dropout = nn.Dropout(p=config.dropout_rate)

        self.prototypes = nn.Parameter(torch.empty(self.prototype_num, self.news_embedding_dim))
        self.prototype_attention = nn.Linear(self.news_embedding_dim * 2, 1, bias=False)
        self.prototype_transform = nn.Linear(self.news_embedding_dim, self.news_embedding_dim, bias=False)
        self.filter_norm = nn.LayerNorm(self.news_embedding_dim)
        self.filter_gate = nn.Linear(self.news_embedding_dim, self.news_embedding_dim, bias=False)
        self.click_predictor = DotProduct()
        self.generated_news_store: Optional[S2LENRGeneratedNewsStore] = None

    def initialize(self):
        self.news_encoder.initialize()
        self.user_self_attention.initialize()
        nn.init.xavier_uniform_(self.user_affine.weight, gain=nn.init.calculate_gain("relu"))
        nn.init.zeros_(self.user_affine.bias)
        self.user_attention.initialize()
        if self.prototype_num > 0:
            nn.init.xavier_uniform_(self.prototypes)
        nn.init.xavier_uniform_(self.prototype_attention.weight)
        nn.init.xavier_uniform_(self.prototype_transform.weight, gain=nn.init.calculate_gain("relu"))
        nn.init.xavier_uniform_(self.filter_gate.weight)

    def set_corpus(self, corpus):
        self.generated_news_store = S2LENRGeneratedNewsStore(self.config, corpus)

    def _lookup_generated_news(self, user_id: torch.Tensor):
        if self.generated_news_store is None:
            raise RuntimeError("S2LENR requires set_corpus(corpus) before training or evaluation.")
        return self.generated_news_store.lookup(user_id, user_id.device)

    def _encode_user_sequence(self, sequence_embedding: torch.Tensor, sequence_mask: torch.Tensor) -> torch.Tensor:
        safe_mask = sequence_mask.float()
        empty_rows = safe_mask.sum(dim=1) == 0
        if empty_rows.any():
            safe_mask = safe_mask.clone()
            safe_mask[empty_rows, 0] = 1.0
        contextual = self.user_self_attention(sequence_embedding, sequence_embedding, sequence_embedding, safe_mask)
        contextual = F.relu(self.dropout(self.user_affine(contextual)), inplace=True)
        return self.user_attention(contextual, safe_mask)

    def _refined_user_prototypes(self, generated_representation: torch.Tensor, generated_mask: torch.Tensor) -> torch.Tensor:
        batch_size = generated_representation.size(0)
        if self.generated_news_num <= 0 or self.prototype_num <= 0:
            return generated_representation.new_zeros(batch_size, self.generated_news_num, self.news_embedding_dim)

        flat_generated = generated_representation.reshape(-1, self.news_embedding_dim)
        flat_mask = generated_mask.reshape(-1).bool()
        if not flat_mask.any():
            return generated_representation.new_zeros(batch_size, self.generated_news_num, self.news_embedding_dim)

        valid_generated = flat_generated[flat_mask]
        weight = self.prototype_attention.weight.squeeze(0)
        prototype_scores = self.prototypes.matmul(weight[: self.news_embedding_dim])
        generated_scores = valid_generated.matmul(weight[self.news_embedding_dim :])
        beta = F.softmax(prototype_scores.unsqueeze(1) + generated_scores.unsqueeze(0), dim=1)
        refined_prototypes = F.relu(self.prototypes + self.prototype_transform(beta.matmul(valid_generated)), inplace=False)

        assignment_scores = generated_representation.matmul(refined_prototypes.t())
        assignment = assignment_scores.argmax(dim=2)
        user_prototypes = refined_prototypes[assignment]
        return user_prototypes * generated_mask.unsqueeze(-1)

    def _ssl_loss(self, user_positive: torch.Tensor, user_negative: torch.Tensor, anchor_news: torch.Tensor) -> torch.Tensor:
        temperature = max(self.ssl_temperature, 1e-6)
        positive_score = F.cosine_similarity(user_positive, anchor_news, dim=1) / temperature
        negative_score = F.cosine_similarity(user_negative, anchor_news, dim=1) / temperature
        if self.use_exact_ratio_loss:
            return (negative_score - positive_score).mean()
        logits = torch.stack([positive_score, negative_score], dim=1)
        labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
        return F.cross_entropy(logits, labels)

    def forward(
        self,
        user_ID,
        user_category,
        user_subCategory,
        user_title_text,
        user_title_mask,
        user_title_entity,
        user_content_text,
        user_content_mask,
        user_content_entity,
        user_history_mask,
        user_history_graph,
        user_history_category_mask,
        user_history_category_indices,
        news_category,
        news_subCategory,
        news_title_text,
        news_title_mask,
        news_title_entity,
        news_content_text,
        news_content_mask,
        news_content_entity,
    ):
        news_representation = self.news_encoder(
            news_title_text,
            news_title_mask,
            news_title_entity,
            news_content_text,
            news_content_mask,
            news_content_entity,
            news_category,
            news_subCategory,
            None,
        )
        if news_representation.dim() == 2:
            news_representation = news_representation.unsqueeze(1)

        history_representation = self.news_encoder(
            user_title_text,
            user_title_mask,
            user_title_entity,
            user_content_text,
            user_content_mask,
            user_content_entity,
            user_category,
            user_subCategory,
            None,
        )

        (
            generated_category,
            generated_subcategory,
            generated_title_text,
            generated_title_mask,
            generated_title_entity,
            generated_content_text,
            generated_content_mask,
            generated_content_entity,
            generated_mask,
        ) = self._lookup_generated_news(user_ID)

        if self.generated_news_num > 0:
            generated_representation = self.news_encoder(
                generated_title_text,
                generated_title_mask,
                generated_title_entity,
                generated_content_text,
                generated_content_mask,
                generated_content_entity,
                generated_category,
                generated_subcategory,
                None,
            )
            user_prototypes = self._refined_user_prototypes(generated_representation, generated_mask)
        else:
            user_prototypes = history_representation.new_zeros(user_ID.size(0), 0, self.news_embedding_dim)
        sequence = torch.cat([history_representation, user_prototypes], dim=1)
        sequence_mask = torch.cat([user_history_mask.float(), generated_mask], dim=1)

        gate = torch.sigmoid(self.filter_gate(self.filter_norm(sequence)))
        positive_sequence = gate * sequence
        user_positive = self._encode_user_sequence(positive_sequence, sequence_mask)

        logits = self.click_predictor(
            user_positive.unsqueeze(dim=1),
            news_representation.permute(0, 2, 1),
        ).squeeze(dim=1)

        if not self.training:
            return logits

        negative_sequence = (1.0 - gate) * sequence
        user_negative = self._encode_user_sequence(negative_sequence, sequence_mask)
        ssl_loss = self._ssl_loss(user_positive, user_negative, news_representation[:, 0, :])
        return logits, ssl_loss

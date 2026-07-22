import json
import os
import re
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Config
from models.modules import newsEncoders
from models.modules.click_predictor import DotProduct


_TOKEN_PATTERN = re.compile(r"[\w]+|[.,!?;|]")


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "default"


def _clean_text(value) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() == "nan" else text


def _is_number(value: str) -> bool:
    try:
        float(value)
        return True
    except ValueError:
        return False


class PNRLLMEnrichmentStore:
    """Loads sparse, offline PNR-LLM enrichments keyed by corpus news index."""

    def __init__(self, config: Config, corpus):
        self.config = config
        self.max_title_length = int(getattr(config, "pnr_llm_max_title_length", 40))
        self.max_entity_length = int(getattr(config, "pnr_llm_max_entity_length", 20))
        self.word_dict = getattr(corpus, "word_dict", {})
        self.news_id_dict = getattr(corpus, "news_ID_dict", {})
        self.path = self._resolve_path(config)
        self.entity_dict: Dict[str, int] = {"<PAD>": 0}
        self.records: Dict[int, Dict[str, np.ndarray]] = {}
        self.news_rows: Dict[int, int] = {}
        self.stacked_records: Dict[str, np.ndarray] = {}
        self._load()

    @staticmethod
    def _resolve_path(config: Config) -> str:
        explicit_path = getattr(config, "pnr_llm_enrichment_path", "")
        if explicit_path:
            return explicit_path
        dataset_root = _safe_name(str(getattr(config, "DATASET_ROOT", "dataset")))
        llm_name = _safe_name(str(getattr(config, "pnr_llm_llm_model", "Qwen/Qwen3-0.6B")))
        return os.path.join("cache", "pnr_llm", dataset_root, llm_name, "enriched_news.jsonl")

    def _load(self):
        if not os.path.exists(self.path):
            message = (
                f"PNR-LLM enrichment cache not found at {self.path}. "
                "Run scripts/generate_pnr_llm_enrichment.py before training the full model."
            )
            if bool(getattr(self.config, "pnr_llm_require_enrichment_cache", False)):
                raise FileNotFoundError(message)
            print("Warning: " + message + " Using original titles without enriched entities.")
            return

        parsed_records = []
        entity_keys = set()
        with open(self.path, "r", encoding="utf-8") as cache_f:
            for line in cache_f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                news_index = self._resolve_news_index(record)
                title = self._record_title(record)
                entities = self._record_entities(record)
                if news_index is None or not title:
                    continue
                parsed_records.append((news_index, title, entities))
                entity_keys.update(key for key, _ in entities if key)

        for entity_key in sorted(entity_keys):
            self.entity_dict[entity_key] = len(self.entity_dict)

        for news_index, title, entities in parsed_records:
            self.records[news_index] = self._encode_record(title, entities)
        self._stack_records()
        print(
            f"PNR-LLM loaded enrichments for {len(self.records)} news articles "
            f"with {len(self.entity_dict) - 1} entities from {self.path}"
        )

    def _stack_records(self):
        ordered_records = sorted(self.records.items())
        self.news_rows = {
            news_index: row_index
            for row_index, (news_index, _) in enumerate(ordered_records, start=1)
        }
        shapes = {
            "title_text": (self.max_title_length,),
            "title_mask": (self.max_title_length,),
            "entity_ids": (self.max_entity_length,),
            "entity_mask": (self.max_entity_length,),
        }
        dtypes = {
            "title_text": np.int64,
            "title_mask": np.float32,
            "entity_ids": np.int64,
            "entity_mask": np.float32,
        }
        for key, shape in shapes.items():
            stacked = np.zeros((len(ordered_records) + 1, *shape), dtype=dtypes[key])
            for row_index, (_, encoded) in enumerate(ordered_records, start=1):
                stacked[row_index] = encoded[key]
            self.stacked_records[key] = stacked

    def _resolve_news_index(self, record) -> Optional[int]:
        news_index = record.get("news_index")
        if news_index is not None:
            try:
                news_index = int(news_index)
            except (TypeError, ValueError):
                news_index = None
            if news_index is not None and news_index >= 0:
                return news_index

        raw_news_id = _clean_text(
            record.get("news_id", record.get("news_id_raw", record.get("nid")))
        )
        if raw_news_id in self.news_id_dict:
            return int(self.news_id_dict[raw_news_id])
        return None

    @staticmethod
    def _record_title(record) -> str:
        for key in ("enriched_title", "hierarchical_title", "final_title", "generated_title"):
            title = _clean_text(record.get(key))
            if title:
                return title
        return ""

    @classmethod
    def _record_entities(cls, record) -> Sequence[Tuple[str, str]]:
        values = record.get(
            "enriched_entities",
            record.get("entities", record.get("related_entities", [])),
        )
        if isinstance(values, str):
            values = [item.strip() for item in re.split(r"[,;\n]", values) if item.strip()]
        if not isinstance(values, list):
            return []

        entities = []
        for entity in values:
            if isinstance(entity, dict):
                name = _clean_text(
                    entity.get("name", entity.get("label", entity.get("entity", entity.get("title"))))
                )
                canonical_id = _clean_text(
                    entity.get("wikidata_id", entity.get("canonical_id", entity.get("id")))
                )
            else:
                name = _clean_text(entity)
                canonical_id = ""
            if not name:
                continue
            key = canonical_id.upper() if canonical_id else re.sub(r"\s+", " ", name.lower())
            entities.append((key, name))
        return entities

    def _encode_record(self, title: str, entities: Sequence[Tuple[str, str]]):
        title_text = np.zeros(self.max_title_length, dtype=np.int64)
        title_mask = np.zeros(self.max_title_length, dtype=np.float32)
        entity_ids = np.zeros(self.max_entity_length, dtype=np.int64)
        entity_mask = np.zeros(self.max_entity_length, dtype=np.float32)

        words = (
            _TOKEN_PATTERN.findall(title.lower())
            if self.config.tokenizer == "MIND"
            else title.lower().split()
        )
        unk_index = self.word_dict.get("<UNK>", 1)
        num_index = self.word_dict.get("<NUM>", unk_index)
        for index, word in enumerate(words[: self.max_title_length]):
            title_text[index] = num_index if _is_number(word) else self.word_dict.get(word, unk_index)
            title_mask[index] = 1.0

        for index, (entity_key, _) in enumerate(entities[: self.max_entity_length]):
            entity_ids[index] = self.entity_dict[entity_key]
            entity_mask[index] = 1.0

        return {
            "title_text": title_text,
            "title_mask": title_mask,
            "entity_ids": entity_ids,
            "entity_mask": entity_mask,
        }

    def lookup(
        self,
        news_indices: torch.Tensor,
        fallback_title_text: torch.Tensor,
        fallback_title_mask: torch.Tensor,
    ):
        leading_shape = tuple(news_indices.shape)
        device = news_indices.device
        flat_size = int(news_indices.numel())
        title_text = torch.zeros(
            (flat_size, self.max_title_length), dtype=torch.long, device=device
        )
        title_mask = torch.zeros(
            (flat_size, self.max_title_length), dtype=torch.float32, device=device
        )
        entity_ids = torch.zeros(
            (flat_size, self.max_entity_length), dtype=torch.long, device=device
        )
        entity_mask = torch.zeros(
            (flat_size, self.max_entity_length), dtype=torch.float32, device=device
        )

        fallback_text = fallback_title_text.reshape(flat_size, -1)
        fallback_mask = fallback_title_mask.reshape(flat_size, -1)
        fallback_length = min(fallback_text.size(1), self.max_title_length)
        title_text[:, :fallback_length] = fallback_text[:, :fallback_length]
        title_mask[:, :fallback_length] = fallback_mask[:, :fallback_length].float()

        rows = np.fromiter(
            (
                self.news_rows.get(int(news_index), 0)
                for news_index in news_indices.detach().cpu().reshape(-1).tolist()
            ),
            dtype=np.int64,
            count=flat_size,
        )
        valid_flat_indices = np.flatnonzero(rows)
        if valid_flat_indices.size:
            valid_device_indices = torch.as_tensor(
                valid_flat_indices, dtype=torch.long, device=device
            )
            valid_rows = rows[valid_flat_indices]
            title_text[valid_device_indices] = torch.as_tensor(
                self.stacked_records["title_text"][valid_rows],
                dtype=torch.long,
                device=device,
            )
            title_mask[valid_device_indices] = torch.as_tensor(
                self.stacked_records["title_mask"][valid_rows],
                dtype=torch.float32,
                device=device,
            )
            entity_ids[valid_device_indices] = torch.as_tensor(
                self.stacked_records["entity_ids"][valid_rows],
                dtype=torch.long,
                device=device,
            )
            entity_mask[valid_device_indices] = torch.as_tensor(
                self.stacked_records["entity_mask"][valid_rows],
                dtype=torch.float32,
                device=device,
            )

        return (
            title_text.view(*leading_shape, self.max_title_length),
            title_mask.view(*leading_shape, self.max_title_length),
            entity_ids.view(*leading_shape, self.max_entity_length),
            entity_mask.view(*leading_shape, self.max_entity_length),
        )


class MaskedAdditiveAttention(nn.Module):
    def __init__(self, feature_dim: int, attention_dim: int):
        super().__init__()
        self.affine1 = nn.Linear(feature_dim, attention_dim)
        self.affine2 = nn.Linear(attention_dim, 1, bias=False)

    def initialize(self):
        nn.init.xavier_uniform_(self.affine1.weight, gain=nn.init.calculate_gain("tanh"))
        nn.init.zeros_(self.affine1.bias)
        nn.init.xavier_uniform_(self.affine2.weight)

    def forward(self, features: torch.Tensor, mask: Optional[torch.Tensor] = None):
        scores = self.affine2(torch.tanh(self.affine1(features))).squeeze(-1)
        if mask is None:
            weights = F.softmax(scores, dim=-1)
            return torch.sum(features * weights.unsqueeze(-1), dim=-2)

        mask = mask.bool()
        has_value = mask.any(dim=-1, keepdim=True)
        safe_mask = mask | ~has_value
        weights = F.softmax(scores.masked_fill(~safe_mask, -1e9), dim=-1)
        output = torch.sum(features * weights.unsqueeze(-1), dim=-2)
        return output * has_value.to(output.dtype)


class PNRLLMNewsEncoder(newsEncoders.NewsEncoder):
    def __init__(self, config: Config):
        super().__init__(config)
        self.max_title_length = int(getattr(config, "pnr_llm_max_title_length", 40))
        self.max_entity_length = int(getattr(config, "pnr_llm_max_entity_length", 20))
        self.cnn_kernel_num = int(getattr(config, "cnn_kernel_num", 400))
        self.entity_embedding_dim = int(getattr(config, "entity_embedding_dim", 100))
        self.category_embedding_dim = config.category_embedding_dim
        self.entity_attention_heads = int(getattr(config, "pnr_llm_entity_attention_heads", 4))
        if self.entity_embedding_dim % self.entity_attention_heads != 0:
            raise ValueError("PNR-LLM entity_embedding_dim must be divisible by entity attention heads.")

        window_size = int(getattr(config, "cnn_window_size", 3))
        self.title_conv = nn.Conv1d(
            config.word_embedding_dim,
            self.cnn_kernel_num,
            kernel_size=window_size,
            padding=(window_size - 1) // 2,
        )
        self.title_attention = MaskedAdditiveAttention(
            self.cnn_kernel_num, config.attention_dim
        )
        self.entity_embedding = nn.Embedding(1, self.entity_embedding_dim, padding_idx=0)
        self.entity_self_attention = nn.MultiheadAttention(
            self.entity_embedding_dim,
            self.entity_attention_heads,
            dropout=config.dropout_rate,
            batch_first=True,
        )
        self.entity_attention = MaskedAdditiveAttention(
            self.entity_embedding_dim, config.attention_dim
        )
        self.entity_affine = nn.Linear(self.entity_embedding_dim, self.cnn_kernel_num)
        self.category_affine = nn.Linear(config.category_embedding_dim, self.cnn_kernel_num)
        self.view_attention = MaskedAdditiveAttention(
            self.cnn_kernel_num, config.attention_dim
        )
        self.output_dropout = nn.Dropout(config.dropout_rate)
        self.news_embedding_dim = self.cnn_kernel_num

    def set_entity_vocabulary_size(self, vocabulary_size: int):
        device = self.word_embedding.weight.device
        self.entity_embedding = nn.Embedding(
            max(int(vocabulary_size), 1), self.entity_embedding_dim, padding_idx=0
        ).to(device)
        nn.init.uniform_(self.entity_embedding.weight, -0.1, 0.1)
        with torch.no_grad():
            self.entity_embedding.weight[0].zero_()

    def initialize(self):
        super().initialize()
        nn.init.xavier_uniform_(self.title_conv.weight, gain=nn.init.calculate_gain("relu"))
        nn.init.zeros_(self.title_conv.bias)
        self.title_attention.initialize()
        self.entity_attention.initialize()
        self.view_attention.initialize()
        nn.init.xavier_uniform_(self.entity_affine.weight, gain=nn.init.calculate_gain("relu"))
        nn.init.zeros_(self.entity_affine.bias)
        nn.init.xavier_uniform_(self.category_affine.weight, gain=nn.init.calculate_gain("relu"))
        nn.init.zeros_(self.category_affine.bias)

    def forward(
        self,
        title_text: torch.Tensor,
        title_mask: torch.Tensor,
        entity_ids: torch.Tensor,
        entity_mask: torch.Tensor,
        category: torch.Tensor,
    ):
        batch_size, news_num, _ = title_text.shape
        flat_news_num = batch_size * news_num

        flat_title_mask = title_mask.reshape(flat_news_num, self.max_title_length)
        title_embedding = self.dropout(
            self.word_embedding(title_text).reshape(
                flat_news_num, self.max_title_length, self.word_embedding_dim
            )
        )
        title_context = F.relu(
            self.title_conv(title_embedding.transpose(1, 2)).transpose(1, 2)
        )
        title_representation = self.title_attention(
            self.output_dropout(title_context), flat_title_mask
        )

        flat_entity_mask = entity_mask.reshape(flat_news_num, self.max_entity_length).bool()
        entity_embedding = self.entity_embedding(entity_ids).reshape(
            flat_news_num, self.max_entity_length, self.entity_embedding_dim
        )
        safe_entity_mask = flat_entity_mask.clone()
        empty_entities = ~safe_entity_mask.any(dim=1)
        if empty_entities.any():
            safe_entity_mask[empty_entities, 0] = True
        entity_context, _ = self.entity_self_attention(
            entity_embedding,
            entity_embedding,
            entity_embedding,
            key_padding_mask=~safe_entity_mask,
            need_weights=False,
        )
        entity_context = entity_context * flat_entity_mask.unsqueeze(-1)
        entity_representation = F.relu(
            self.entity_affine(
                self.entity_attention(entity_context, flat_entity_mask)
            )
        )

        category_representation = F.relu(
            self.category_affine(self.category_embedding(category).reshape(
                flat_news_num, self.category_embedding_dim
            ))
        )
        features = torch.stack(
            [title_representation, category_representation, entity_representation], dim=1
        )
        view_mask = torch.stack(
            [
                flat_title_mask.bool().any(dim=1),
                torch.ones(flat_news_num, dtype=torch.bool, device=title_text.device),
                flat_entity_mask.any(dim=1),
            ],
            dim=1,
        )
        return self.view_attention(features, view_mask).view(
            batch_size, news_num, self.news_embedding_dim
        )


class PNR_LLM(nn.Module):
    """PNR-LLM reproduction from Kieu et al., WWW Companion 2025."""

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.model_name = config.model
        self.batch_size = config.batch_size
        self.news_encoder = PNRLLMNewsEncoder(config)
        self.user_attention = MaskedAdditiveAttention(
            self.news_encoder.news_embedding_dim, config.attention_dim
        )
        self.click_predictor = DotProduct()
        self.enrichment_store: Optional[PNRLLMEnrichmentStore] = None

    def initialize(self):
        self.news_encoder.initialize()
        self.user_attention.initialize()

    def set_corpus(self, corpus):
        self.enrichment_store = PNRLLMEnrichmentStore(self.config, corpus)
        self.news_encoder.set_entity_vocabulary_size(
            len(self.enrichment_store.entity_dict)
        )

    def _encode_news(
        self,
        news_indices: torch.Tensor,
        fallback_title_text: torch.Tensor,
        fallback_title_mask: torch.Tensor,
        category: torch.Tensor,
    ):
        if self.enrichment_store is None:
            raise RuntimeError("PNR-LLM requires set_corpus(corpus) before training or evaluation.")
        title_text, title_mask, entity_ids, entity_mask = self.enrichment_store.lookup(
            news_indices, fallback_title_text, fallback_title_mask
        )
        return self.news_encoder(
            title_text, title_mask, entity_ids, entity_mask, category
        )

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
        history_news_indices=None,
        candidate_news_indices=None,
    ):
        if history_news_indices is None or candidate_news_indices is None:
            raise ValueError(
                "PNR-LLM requires history and candidate news indices from the unified dataset."
            )
        if candidate_news_indices.dim() == 1:
            candidate_news_indices = candidate_news_indices.unsqueeze(1)
        if news_category.dim() == 1:
            news_category = news_category.unsqueeze(1)
            news_title_text = news_title_text.unsqueeze(1)
            news_title_mask = news_title_mask.unsqueeze(1)

        candidate_representation = self._encode_news(
            candidate_news_indices, news_title_text, news_title_mask, news_category
        )
        history_representation = self._encode_news(
            history_news_indices, user_title_text, user_title_mask, user_category
        )
        user_representation = self.user_attention(
            history_representation, user_history_mask
        )
        return self.click_predictor(
            user_representation.unsqueeze(1), candidate_representation.transpose(1, 2)
        ).squeeze(dim=1)

import os
import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Config
from ..layers import Attention, MultiHeadAttention, QAttention


def _word_embedding_path(config: Config) -> str:
    cache_dir = "cache" if config.dataset_name == "MIND" else os.path.join(
        "cache", config.dataset_name.lower()
    )
    filename = "word_embedding-{}-{}-{}-{}-{}-{}.pkl".format(
        config.word_threshold,
        config.word_embedding_dim,
        config.tokenizer,
        config.max_title_length,
        config.max_abstract_length,
        config.dataset_size,
    )
    return os.path.join(cache_dir, filename)


def _native_semantic_path(config: Config) -> str:
    configured_path = getattr(config, "lkpnr_semantic_embedding_path", "")
    if configured_path:
        return configured_path
    return os.path.join(
        "cache",
        config.dataset_name.lower(),
        f"lkpnr_item_embedding-{config.dataset_size}.npy",
    )


class NewsEncoder(nn.Module):
    """LKPNR news encoder shared by the legacy MIND and native data paths.

    The original implementation indexes four MIND-specific files by internal
    article ID.  EB-NeRD and Adressa have different IDs and knowledge schemas,
    so their native path derives the semantic view from pretrained word
    embeddings and consumes linked-node IDs supplied by their corpus adapters.
    """

    LEGACY_ASSETS = {
        "item": os.path.join("KGraph_LKPNR", "item_emb.npy"),
        "ids": os.path.join("KGraph_LKPNR", "ID_news-large.pkl"),
        "links": os.path.join("KGraph_LKPNR", "link_entity_dic.pkl"),
        "entities": os.path.join("KGraph_LKPNR", "all_entity_emb_dic.pkl"),
    }

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.word_embedding_dim = config.word_embedding_dim
        self.word_embedding = nn.Embedding(
            num_embeddings=config.vocabulary_size,
            embedding_dim=self.word_embedding_dim,
            padding_idx=0,
        )
        embedding_path = _word_embedding_path(config)
        with open(embedding_path, "rb") as word_embedding_file:
            weights = torch.as_tensor(pickle.load(word_embedding_file))
        expected_shape = (config.vocabulary_size, config.word_embedding_dim)
        if tuple(weights.shape) != expected_shape:
            raise ValueError(
                f"LKPNR word embedding shape mismatch at {embedding_path}: "
                f"expected {expected_shape}, got {tuple(weights.shape)}"
            )
        self.word_embedding.weight.data.copy_(weights)

        self.category_embedding = nn.Embedding(
            config.category_num, config.category_embedding_dim
        )
        self.subCategory_embedding = nn.Embedding(
            config.subCategory_num, config.subCategory_embedding_dim
        )
        self.dropout = nn.Dropout(p=config.dropout_rate)
        self.auxiliary_loss = None

        requested_mode = getattr(config, "lkpnr_knowledge_mode", "auto").lower()
        if requested_mode not in {"auto", "legacy", "native"}:
            raise ValueError(
                "lkpnr_knowledge_mode must be one of: auto, legacy, native"
            )
        legacy_available = all(os.path.exists(path) for path in self.LEGACY_ASSETS.values())
        if requested_mode == "auto":
            self.knowledge_mode = (
                "legacy" if config.dataset_name == "MIND" and legacy_available else "native"
            )
        else:
            self.knowledge_mode = requested_mode

        if self.knowledge_mode == "legacy":
            missing = [
                path for path in self.LEGACY_ASSETS.values() if not os.path.exists(path)
            ]
            if missing:
                raise FileNotFoundError(
                    "LKPNR legacy knowledge mode is missing required assets: "
                    + ", ".join(missing)
                )
            self.item_emb_dic = np.load(
                self.LEGACY_ASSETS["item"], allow_pickle=True
            ).item()
            with open(self.LEGACY_ASSETS["ids"], "rb") as source:
                self.ID_news = pickle.load(source)
            with open(self.LEGACY_ASSETS["links"], "rb") as source:
                self.link_entity_dic = pickle.load(source)
            with open(self.LEGACY_ASSETS["entities"], "rb") as source:
                self.all_entity_emb_dic = pickle.load(source)
            self.semantic_input_dim = config.pretrain_emb_d
            self.semantic_source = "legacy_llm_cache"
            self.entity_embedding = None
        else:
            entity_size = max(2, int(getattr(config, "entity_size", 2)))
            self.entity_embedding = nn.Embedding(
                entity_size, config.entity_embedding_dim, padding_idx=0
            )
            semantic_path = _native_semantic_path(config)
            self.semantic_cache = None
            if os.path.exists(semantic_path):
                semantic_cache = np.load(semantic_path, mmap_mode="r")
                if semantic_cache.ndim != 2:
                    raise ValueError(
                        f"LKPNR semantic cache must be a 2-D array: {semantic_path}"
                    )
                news_num = int(getattr(config, "news_num", 0))
                if news_num and semantic_cache.shape[0] < news_num:
                    raise ValueError(
                        f"LKPNR semantic cache has {semantic_cache.shape[0]} rows, "
                        f"but the corpus has {news_num} news items: {semantic_path}"
                    )
                self.semantic_cache = semantic_cache
                self.semantic_input_dim = int(semantic_cache.shape[1])
                self.semantic_source = "llm_cache"
            else:
                if getattr(config, "lkpnr_require_semantic_cache", False):
                    raise FileNotFoundError(
                        "LKPNR semantic cache is required but missing: "
                        f"{semantic_path}. Run scripts/generate_lkpnr_embeddings.py first."
                    )
                self.semantic_input_dim = config.word_embedding_dim
                self.semantic_source = "pretrained_text_pooling"

    def initialize(self):
        nn.init.uniform_(self.category_embedding.weight, -0.1, 0.1)
        nn.init.uniform_(self.subCategory_embedding.weight, -0.1, 0.1)
        if self.entity_embedding is not None:
            nn.init.uniform_(self.entity_embedding.weight, -0.1, 0.1)
            nn.init.zeros_(self.entity_embedding.weight[0])

    def feature_fusion(
        self,
        news_representation,
        category,
        subCategory,
        semantic_representation,
        linked_entity_representation,
    ):
        category_representation = self.dropout(self.category_embedding(category))
        subcategory_representation = self.dropout(
            self.subCategory_embedding(subCategory)
        )
        return torch.cat(
            [
                news_representation,
                category_representation,
                subcategory_representation,
                semantic_representation,
                linked_entity_representation,
            ],
            dim=2,
        )

    @staticmethod
    def _masked_mean(feature: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask = mask.to(feature.dtype).unsqueeze(dim=-1)
        return (feature * mask).sum(dim=-2) / mask.sum(dim=-2).clamp_min(1.0)

    def native_semantic_embedding(
        self,
        title_word_embedding,
        title_mask,
        content_text,
        content_mask,
    ):
        title_mean = self._masked_mean(title_word_embedding, title_mask)
        if content_text is None or content_mask is None:
            return title_mean
        content_embedding = self.word_embedding(content_text)
        content_mean = self._masked_mean(content_embedding, content_mask)
        title_available = title_mask.sum(dim=-1, keepdim=True).gt(0).to(title_mean.dtype)
        content_available = content_mask.sum(dim=-1, keepdim=True).gt(0).to(title_mean.dtype)
        return (
            title_mean * title_available + content_mean * content_available
        ) / (title_available + content_available).clamp_min(1.0)

    def cached_semantic_embedding(self, news_indices, device):
        indices = news_indices.detach().cpu().numpy()
        if indices.size and (
            indices.min() < 0 or indices.max() >= self.semantic_cache.shape[0]
        ):
            raise IndexError(
                "LKPNR article index is outside the generated semantic cache"
            )
        values = np.array(self.semantic_cache[indices], dtype=np.float32, copy=True)
        return torch.from_numpy(values).to(device=device)

    def _legacy_news_id(self, index: int):
        if isinstance(self.ID_news, dict):
            return self.ID_news.get(index, self.ID_news.get(str(index), "<PAD>"))
        if 0 <= index < len(self.ID_news):
            return self.ID_news[index]
        return "<PAD>"

    def legacy_semantic_embedding(self, news_indices, device):
        indices = news_indices.detach().cpu().numpy()
        result = np.zeros((*indices.shape, self.config.pretrain_emb_d), dtype=np.float32)
        for position in np.ndindex(indices.shape):
            news_id = self._legacy_news_id(int(indices[position]))
            vector = self.item_emb_dic.get(news_id)
            if vector is None:
                continue
            vector = np.asarray(vector, dtype=np.float32)
            if vector.shape != (self.config.pretrain_emb_d,):
                raise ValueError(
                    f"LKPNR item embedding for {news_id!r} has shape {vector.shape}; "
                    f"expected {(self.config.pretrain_emb_d,)}"
                )
            result[position] = vector
        return torch.from_numpy(result).to(device=device)

    def native_linked_entities(self, title_entity, content_entity):
        entity_ids = title_entity
        if content_entity is not None:
            entity_ids = torch.cat([entity_ids, content_entity], dim=-1)
        max_length = self.config.max_linked_entity_length
        entity_ids = entity_ids[..., :max_length]
        if entity_ids.size(-1) < max_length:
            entity_ids = F.pad(entity_ids, (0, max_length - entity_ids.size(-1)))
        mask = entity_ids.ne(0)
        return self.entity_embedding(entity_ids), mask

    def legacy_linked_entities(self, news_indices, device):
        indices = news_indices.detach().cpu().numpy()
        shape = (*indices.shape, self.config.max_linked_entity_length)
        mask = np.zeros(shape, dtype=bool)
        embeddings = np.zeros(
            (*shape, self.config.entity_embedding_dim), dtype=np.float32
        )
        for position in np.ndindex(indices.shape):
            news_id = self._legacy_news_id(int(indices[position]))
            if news_id == "<PAD>":
                continue
            linked_entities = self.link_entity_dic.get(news_id, [])[
                : self.config.max_linked_entity_length
            ]
            for offset, entity_id in enumerate(linked_entities):
                vector = self.all_entity_emb_dic.get(entity_id)
                if vector is None:
                    continue
                vector = np.asarray(vector, dtype=np.float32)
                if vector.shape != (self.config.entity_embedding_dim,):
                    continue
                embeddings[position + (offset,)] = vector
                mask[position + (offset,)] = True
        return (
            torch.from_numpy(embeddings).to(device=device),
            torch.from_numpy(mask).to(device=device),
        )


class MHSA(NewsEncoder):
    def __init__(self, config: Config):
        super().__init__(config)
        if config.entity_att_head_num != 3:
            raise ValueError("LKPNR currently requires entity_att_head_num: 3")
        self.max_sentence_length = config.max_title_length
        self.feature_dim = config.head_num * config.head_dim
        self.multiheadAttention = MultiHeadAttention(
            config.head_num,
            config.word_embedding_dim,
            config.max_title_length,
            config.max_title_length,
            config.head_dim,
            config.head_dim,
        )
        self.attention = Attention(self.feature_dim, config.attention_dim)
        self.entity_attention = QAttention(
            config.entity_embedding_dim * 3, config.entity_attention_dim
        )
        self.news_embedding_dim = (
            self.feature_dim
            + config.category_embedding_dim
            + config.subCategory_embedding_dim
            + config.pretrain_rep_d
            + config.entity_hidden_dim
        )

        self.dense1 = nn.Linear(
            self.semantic_input_dim, config.pretrain_hidden_dim, bias=True
        )
        self.dense2 = nn.Linear(
            config.pretrain_hidden_dim, config.pretrain_rep_d, bias=True
        )
        self.dense3 = nn.Linear(
            config.entity_embedding_dim * config.entity_att_head_num,
            config.entity_hidden_dim * 2,
            bias=True,
        )
        self.dense4 = nn.Linear(
            config.entity_hidden_dim * 2, config.entity_hidden_dim, bias=True
        )
        self.denseQ_pre1 = nn.Linear(
            self.semantic_input_dim, config.pretrain_hidden_dim, bias=True
        )
        self.denseQ_pre2 = nn.Linear(
            config.pretrain_hidden_dim, config.pretrain_rep_d, bias=True
        )
        self.denseHead1 = nn.Linear(
            config.pretrain_rep_d, config.entity_embedding_dim, bias=True
        )
        self.denseHead2 = nn.Linear(
            config.pretrain_rep_d, config.entity_embedding_dim, bias=True
        )
        self.denseHead3 = nn.Linear(
            config.pretrain_rep_d, config.entity_embedding_dim, bias=True
        )

    def initialize(self):
        super().initialize()
        self.multiheadAttention.initialize()
        self.attention.initialize()
        self.entity_attention.initialize()
        for layer in (
            self.dense1,
            self.dense2,
            self.dense3,
            self.dense4,
            self.denseQ_pre1,
            self.denseQ_pre2,
            self.denseHead1,
            self.denseHead2,
            self.denseHead3,
        ):
            nn.init.xavier_uniform_(layer.weight, gain=nn.init.calculate_gain("relu"))
            nn.init.zeros_(layer.bias)

    def forward(
        self,
        title_text,
        title_mask,
        title_entity,
        content_text,
        content_mask,
        content_entity,
        category,
        subCategory,
        user_embedding=None,
        sample_index=None,
    ):
        del user_embedding
        batch_size, news_num, _ = title_text.shape
        batch_news_num = batch_size * news_num
        title_mask_flat = title_mask.reshape(
            batch_news_num, self.max_sentence_length
        )

        title_word_embedding = self.word_embedding(title_text)
        word_feature = self.dropout(title_word_embedding).reshape(
            batch_news_num,
            self.max_sentence_length,
            self.word_embedding_dim,
        )
        contextual_feature = self.dropout(
            self.multiheadAttention(
                word_feature,
                word_feature,
                word_feature,
                title_mask_flat,
            )
        )
        news_representation = self.attention(
            contextual_feature, mask=title_mask_flat
        ).reshape(batch_size, news_num, self.feature_dim)

        if self.knowledge_mode == "legacy":
            if sample_index is None:
                raise ValueError("LKPNR legacy mode requires article indices")
            semantic_embedding = self.legacy_semantic_embedding(
                sample_index, title_text.device
            )
            linked_entity_embedding, linked_entity_mask = self.legacy_linked_entities(
                sample_index, title_text.device
            )
        else:
            if self.semantic_cache is not None:
                if sample_index is None:
                    raise ValueError(
                        "LKPNR LLM semantic cache requires article indices"
                    )
                semantic_embedding = self.cached_semantic_embedding(
                    sample_index, title_text.device
                )
            else:
                semantic_embedding = self.native_semantic_embedding(
                    title_word_embedding,
                    title_mask,
                    content_text,
                    content_mask,
                )
            linked_entity_embedding, linked_entity_mask = self.native_linked_entities(
                title_entity, content_entity
            )

        semantic_embedding = F.normalize(semantic_embedding, dim=2)
        semantic_hidden = F.relu(
            self.dropout(self.dense1(semantic_embedding))
        )
        semantic_representation = F.relu(
            self.dropout(self.dense2(semantic_hidden))
        )

        entity_query_hidden = F.relu(
            self.dropout(self.denseQ_pre1(semantic_embedding))
        )
        entity_query = F.relu(
            self.dropout(self.denseQ_pre2(entity_query_hidden))
        )
        query_heads = [
            F.relu(self.dropout(layer(entity_query)))
            .reshape(batch_news_num, self.config.entity_embedding_dim)
            .unsqueeze(dim=1)
            .expand(-1, self.config.max_linked_entity_length, -1)
            for layer in (self.denseHead1, self.denseHead2, self.denseHead3)
        ]
        linked_entity_mask_flat = linked_entity_mask.reshape(
            batch_news_num, self.config.max_linked_entity_length
        )
        linked_entity_embedding_flat = linked_entity_embedding.reshape(
            batch_news_num,
            self.config.max_linked_entity_length,
            self.config.entity_embedding_dim,
        )
        linked_entity_representation = self.entity_attention(
            linked_entity_embedding_flat,
            query_heads,
            mask=linked_entity_mask_flat,
        ).reshape(
            batch_size,
            news_num,
            self.config.entity_embedding_dim * self.config.entity_att_head_num,
        )
        linked_entity_representation = F.normalize(
            linked_entity_representation, dim=2
        )
        linked_entity_hidden = F.relu(
            self.dropout(self.dense3(linked_entity_representation))
        )
        linked_entity_representation = F.relu(
            self.dropout(self.dense4(linked_entity_hidden))
        )
        has_linked_entity = linked_entity_mask.any(dim=-1, keepdim=True).to(
            linked_entity_representation.dtype
        )
        linked_entity_representation = (
            linked_entity_representation * has_linked_entity
        )

        return self.feature_fusion(
            news_representation,
            category,
            subCategory,
            semantic_representation,
            linked_entity_representation,
        )

"""Strong-backbone model composition for the NewsTorch extensibility study.

The class in this module is shared by every B0--B7 configuration.  Variants
select registered news, user, and enhancement modules at runtime; no variant
has a dedicated model class.
"""

from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Tuple, Type

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_softmax, scatter_sum

from config import Config
from models.CNRCL import CNRCLNewsEncoder
from models.modules import newsEncoders
from models.modules.layers import (
    Attention,
    GCN,
    MultiHeadAttention,
    ScaledDotProduct_CandidateAttention,
)


NewsEncoderType = Type[nn.Module]

_TEXT_NEWS_ENCODERS: Dict[str, NewsEncoderType] = {
    "cne": newsEncoders.CNE,
    "cnrcl": CNRCLNewsEncoder,
    "naml": newsEncoders.NAML,
    "nrms": newsEncoders.MHSA,
}
_USER_ENCODERS = ("cnrcl", "nrms", "sein", "sue")
_ENHANCEMENTS = ("none", "llm-augmentation")


def _normalise_name(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Component names must be non-empty strings")
    return value.strip().lower().replace("_", "-")


def available_sota_components() -> Dict[str, Tuple[str, ...]]:
    """Return the stable component registry used by B0--B7."""

    return {
        "news": tuple(sorted(tuple(_TEXT_NEWS_ENCODERS) + ("once-qwen3",))),
        "user": tuple(sorted(_USER_ENCODERS)),
        "enhancement": tuple(sorted(_ENHANCEMENTS)),
        "training": ("cnrcl", "standard"),
        "special": ("sein-temporal-graph",),
    }


class TextNewsModule(nn.Module):
    """Adapt a registered NewsTorch news encoder to a shared output width."""

    def __init__(self, encoder_class: NewsEncoderType, config: Config, output_dim: int):
        super().__init__()
        self.encoder = encoder_class(config)
        self.source_dim = int(self.encoder.news_embedding_dim)
        self.news_embedding_dim = int(output_dim)
        self.projection = (
            nn.Identity()
            if self.source_dim == self.news_embedding_dim
            else nn.Linear(self.source_dim, self.news_embedding_dim)
        )
        self.auxiliary_loss = None

    def initialize(self) -> None:
        self.encoder.initialize()
        if isinstance(self.projection, nn.Linear):
            nn.init.xavier_uniform_(self.projection.weight)
            nn.init.zeros_(self.projection.bias)

    def encode(self, features, news_index=None):
        del news_index
        representation = self.encoder(*features, None)
        return self.projection(representation)

    def set_corpus(self, corpus) -> None:
        setter = getattr(self.encoder, "set_corpus", None)
        if callable(setter):
            setter(corpus)


class FrozenONCEQwenNewsModule(nn.Module):
    """Use one frozen, seed-shared ONCE/Qwen3 embedding per news item.

    The compact embedding file is built from NewsTorch's existing ONCE Qwen3
    hidden-state cache.  It is deliberately memory mapped and excluded from
    checkpoints.  Only the dimension adapter is trained.
    """

    def __init__(self, config: Config, output_dim: int):
        super().__init__()
        self.news_embedding_dim = int(output_dim)
        self.source_dim = int(getattr(config, "sota_llm_embedding_dim", 1024))
        self.embedding_path = str(getattr(config, "sota_llm_embedding_path", ""))
        self.projection = nn.Linear(self.source_dim, self.news_embedding_dim)
        self.dropout = nn.Dropout(config.dropout_rate)
        self._embeddings = None
        self.synthetic_embedding = None
        if self.embedding_path == ":synthetic:":
            self.synthetic_embedding = nn.Embedding(
                int(config.news_num), self.source_dim, padding_idx=0
            )
            self.synthetic_embedding.weight.requires_grad_(False)
        self.auxiliary_loss = None

    def initialize(self) -> None:
        nn.init.xavier_uniform_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)
        if self.synthetic_embedding is not None:
            with torch.no_grad():
                nn.init.normal_(self.synthetic_embedding.weight, std=0.02)
                self.synthetic_embedding.weight[0].zero_()

    def set_corpus(self, corpus) -> None:
        if self.synthetic_embedding is not None:
            return
        if not self.embedding_path:
            raise ValueError("sota_llm_embedding_path is required for ONCE-Qwen3")
        path = Path(self.embedding_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError("Frozen ONCE/Qwen3 embeddings not found: %s" % path)
        embeddings = np.load(str(path), mmap_mode="r", allow_pickle=False)
        expected_news = int(corpus.news_num)
        expected_shape = (expected_news, self.source_dim)
        if tuple(embeddings.shape) != expected_shape:
            raise ValueError(
                "Frozen ONCE/Qwen3 embeddings have shape %r; expected %r"
                % (tuple(embeddings.shape), expected_shape)
            )
        self._embeddings = embeddings

    @staticmethod
    def _indices(value, device) -> torch.Tensor:
        if torch.is_tensor(value):
            return value.to(device=device, dtype=torch.long)
        if isinstance(value, (list, tuple)) and value and torch.is_tensor(value[0]):
            return torch.stack(list(value), dim=1).to(device=device, dtype=torch.long)
        return torch.as_tensor(value, device=device, dtype=torch.long)

    def encode(self, features, news_index=None):
        del features
        if news_index is None:
            raise ValueError("ONCE-Qwen3 news composition requires news indices")
        device = self.projection.weight.device
        indices = self._indices(news_index, device)
        leading_shape = tuple(indices.shape)
        if self.synthetic_embedding is not None:
            source = self.synthetic_embedding(indices)
        else:
            if self._embeddings is None:
                raise RuntimeError("set_corpus(corpus) must load ONCE/Qwen3 embeddings")
            flat = indices.detach().cpu().reshape(-1).numpy()
            values = np.array(self._embeddings[flat], dtype=np.float32, copy=True)
            source = torch.from_numpy(values).to(device=device)
            source = source.view(*leading_shape, self.source_dim)
        valid = indices.ne(0).unsqueeze(-1).to(dtype=source.dtype)
        return self.dropout(self.projection(source)) * valid


class NewsRepresentationStack(nn.Module):
    """News representation with optional frozen-LLM augmentation."""

    def __init__(self, config: Config):
        super().__init__()
        news_name = _normalise_name(config.news_encoder_type)
        enhancement = _normalise_name(getattr(config, "enhancement_type", "none"))
        output_dim = int(getattr(config, "sota_composition_dim", 900))
        if news_name == "once-qwen3":
            self.base = FrozenONCEQwenNewsModule(config, output_dim)
        else:
            try:
                encoder_class = _TEXT_NEWS_ENCODERS[news_name]
            except KeyError as error:
                raise ValueError("Unknown news module %r" % news_name) from error
            self.base = TextNewsModule(encoder_class, config, output_dim)
        if enhancement not in _ENHANCEMENTS:
            raise ValueError("Unknown enhancement module %r" % enhancement)
        self.augmentation = (
            FrozenONCEQwenNewsModule(config, output_dim)
            if enhancement == "llm-augmentation"
            else None
        )
        # A residual gate preserves the text backbone at initialization.  The
        # previous unconstrained concatenation projection erased a trained
        # representation scale before the LLM signal had learned to align.
        self.fusion = (
            nn.Linear(output_dim, output_dim, bias=False)
            if self.augmentation is not None
            else None
        )
        self.fusion_gate = (
            nn.Parameter(torch.tensor(-4.0))
            if self.augmentation is not None
            else None
        )
        self.news_embedding_dim = output_dim
        self.news_encoder_type = news_name
        self.enhancement_type = enhancement
        self.auxiliary_loss = None

    def initialize(self) -> None:
        self.base.initialize()
        if self.augmentation is not None:
            self.augmentation.initialize()
            nn.init.xavier_uniform_(self.fusion.weight)

    def set_corpus(self, corpus) -> None:
        self.base.set_corpus(corpus)
        if self.augmentation is not None:
            self.augmentation.set_corpus(corpus)

    def encode(self, features, news_index=None):
        base = self.base.encode(features, news_index)
        if self.augmentation is None:
            return base
        llm = self.augmentation.encode(features, news_index)
        return base + torch.sigmoid(self.fusion_gate) * self.fusion(llm)


class EncodedCNRCLUserModule(nn.Module):
    """CNRCL structural user modeling over already encoded news vectors."""

    def __init__(self, config: Config, news_dim: int):
        super().__init__()
        self.news_embedding_dim = news_dim
        self.attention_dim = max(config.attention_dim, news_dim // 4)
        self.proxy_node_embedding = nn.Parameter(
            torch.zeros(config.category_num, news_dim)
        )
        self.gcn = GCN(
            in_dim=news_dim,
            out_dim=news_dim,
            hidden_dim=news_dim,
            num_layers=config.gcn_layer_num,
            dropout=config.dropout_rate / 2,
            residual=not getattr(config, "no_gcn_residual", False),
            layer_norm=False,
        )
        self.intra_cluster_key = nn.Linear(news_dim, self.attention_dim, bias=False)
        self.intra_cluster_query = nn.Linear(news_dim, self.attention_dim, bias=True)
        self.cluster_affine = nn.Linear(news_dim, news_dim)
        self.inter_cluster_attention = ScaledDotProduct_CandidateAttention(
            news_dim, news_dim, self.attention_dim
        )
        self.dropout = nn.Dropout(config.dropout_rate)
        self.proxy_dropout = nn.Dropout(config.dropout_rate)
        self.category_num = config.category_num + 1
        self.max_history_num = config.max_history_num
        self.attention_scalar = math.sqrt(float(self.attention_dim))
        self.auxiliary_loss = None

    def initialize(self) -> None:
        self.gcn.initialize()
        nn.init.zeros_(self.proxy_node_embedding)
        nn.init.xavier_uniform_(self.intra_cluster_key.weight)
        nn.init.xavier_uniform_(self.intra_cluster_query.weight)
        nn.init.zeros_(self.intra_cluster_query.bias)
        nn.init.xavier_uniform_(
            self.cluster_affine.weight, gain=nn.init.calculate_gain("relu")
        )
        nn.init.zeros_(self.cluster_affine.bias)
        self.inter_cluster_attention.initialize()

    def forward(
        self,
        history_embedding,
        candidate_embedding,
        history_mask,
        history_graph,
        history_category_mask,
        history_category_indices,
        **unused,
    ):
        del history_mask, unused
        batch_size, news_num = candidate_embedding.shape[:2]
        batch_news_num = batch_size * news_num
        category_mask = history_category_mask.clone()
        category_mask[:, -1] = 1
        category_mask = category_mask.unsqueeze(1).expand(-1, news_num, -1).contiguous()
        category_indices = history_category_indices.unsqueeze(1).expand(
            -1, news_num, -1
        )
        graph_input = torch.cat(
            [
                history_embedding,
                self.proxy_dropout(self.proxy_node_embedding)
                .unsqueeze(0)
                .expand(batch_size, -1, -1),
            ],
            dim=1,
        )
        graph_feature = self.gcn(graph_input, history_graph) + graph_input
        graph_feature = graph_feature[:, : self.max_history_num]
        graph_feature = graph_feature.unsqueeze(1).expand(-1, news_num, -1, -1)
        keys = self.intra_cluster_key(graph_feature).reshape(
            batch_news_num, self.max_history_num, self.attention_dim
        )
        queries = self.intra_cluster_query(candidate_embedding).reshape(
            batch_news_num, self.attention_dim, 1
        )
        scores = torch.bmm(keys, queries).reshape(
            batch_size, news_num, self.max_history_num
        ) / self.attention_scalar
        weights = scatter_softmax(scores, category_indices, dim=2).unsqueeze(3)
        clusters = scatter_sum(
            weights * graph_feature,
            category_indices,
            dim=2,
            dim_size=self.category_num,
        )
        clusters = self.dropout(
            F.relu(self.cluster_affine(clusters), inplace=False) + clusters
        )
        return self.inter_cluster_attention(
            clusters.reshape(batch_news_num, self.category_num, self.news_embedding_dim),
            candidate_embedding.reshape(batch_news_num, self.news_embedding_dim),
            mask=category_mask.reshape(batch_news_num, self.category_num),
        ).reshape(batch_size, news_num, self.news_embedding_dim)


class EncodedNRMSUserModule(nn.Module):
    """NRMS self-attentive user modeling over shared news vectors."""

    def __init__(self, config: Config, news_dim: int):
        super().__init__()
        self.multihead_attention = MultiHeadAttention(
            config.head_num,
            news_dim,
            config.max_history_num,
            config.max_history_num,
            config.head_dim,
            config.head_dim,
        )
        self.affine = nn.Linear(config.head_num * config.head_dim, news_dim)
        self.attention = Attention(news_dim, config.attention_dim)
        self.dropout_rate = config.dropout_rate
        self.auxiliary_loss = None

    def initialize(self) -> None:
        self.multihead_attention.initialize()
        nn.init.xavier_uniform_(self.affine.weight, gain=nn.init.calculate_gain("relu"))
        nn.init.zeros_(self.affine.bias)
        self.attention.initialize()

    def forward(self, history_embedding, candidate_embedding, history_mask, **unused):
        del unused
        hidden = self.multihead_attention(
            history_embedding, history_embedding, history_embedding, history_mask
        )
        hidden = F.relu(
            F.dropout(
                self.affine(hidden), p=self.dropout_rate, training=self.training
            ),
            inplace=False,
        )
        user = self.attention(hidden, history_mask)
        return user.unsqueeze(1).expand(-1, candidate_embedding.size(1), -1)


class EncodedSEINUserModule(nn.Module):
    """SEIN temporal/global-graph user modeling over shared news vectors."""

    def __init__(self, config: Config, news_dim: int):
        super().__init__()
        self.config = config
        self.news_dim = news_dim
        self.hidden_dim = int(getattr(config, "sein_hidden_dim", 256))
        self.stage_num = int(getattr(config, "sein_stage_num", 5))
        self.gcn_layer_num = int(getattr(config, "sein_gcn_layer_num", 2))
        self.temperature = float(getattr(config, "sein_temperature", 0.1))
        self.lambda_cl = float(getattr(config, "sein_consistency_loss_weight", 0.01))
        self.lambda_sl = float(getattr(config, "sein_smoothness_loss_weight", 0.01))
        self.user_embedding = nn.Embedding(config.user_num, self.hidden_dim)
        self.news_id_embedding = nn.Embedding(
            config.news_num, self.hidden_dim, padding_idx=0
        )
        self.semantic_projection = nn.Linear(news_dim, self.hidden_dim)
        self.output_projection = nn.Linear(self.hidden_dim, news_dim)
        self.stage_lstm = nn.LSTM(self.hidden_dim, self.hidden_dim, batch_first=True)
        self.item_lstm = nn.LSTM(self.hidden_dim, self.hidden_dim, batch_first=True)
        self.position_embedding = nn.Embedding(config.max_history_num, self.hidden_dim)
        self.history_query = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.history_key = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.stage_query = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.stage_key = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.user_gate = nn.Linear(self.hidden_dim * 4, self.hidden_dim)
        attention_heads = int(getattr(config, "sein_attention_heads", 4))
        if self.hidden_dim % attention_heads:
            raise ValueError("sein_hidden_dim must be divisible by sein_attention_heads")
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=attention_heads,
            dim_feedforward=self.hidden_dim * 4,
            dropout=config.dropout_rate,
            activation="relu",
            batch_first=True,
        )
        self.long_range_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=int(getattr(config, "sein_attention_layers", 1)),
        )
        self.dropout = nn.Dropout(config.dropout_rate)
        self.register_buffer("global_graph", torch.empty(0), persistent=False)
        self.register_buffer(
            "seen_user_mask", torch.empty(0, dtype=torch.bool), persistent=False
        )
        self.register_buffer(
            "seen_news_mask", torch.empty(0, dtype=torch.bool), persistent=False
        )
        self.auxiliary_loss = None

    def initialize(self) -> None:
        nn.init.normal_(self.user_embedding.weight, std=0.02)
        nn.init.normal_(self.news_id_embedding.weight, std=0.02)
        nn.init.zeros_(self.news_id_embedding.weight[0])
        nn.init.normal_(self.position_embedding.weight, std=0.02)
        for module in (
            self.semantic_projection,
            self.output_projection,
            self.history_query,
            self.history_key,
            self.stage_query,
            self.stage_key,
            self.user_gate,
        ):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        for module in (self.stage_lstm, self.item_lstm):
            for name, parameter in module.named_parameters():
                if "weight" in name:
                    nn.init.xavier_uniform_(parameter)
                else:
                    nn.init.zeros_(parameter)

    def set_corpus(self, corpus) -> None:
        edges = set()
        seen_users = torch.zeros(self.config.user_num, dtype=torch.bool)
        seen_news = torch.zeros(self.config.news_num, dtype=torch.bool)
        user_offset = self.config.user_num
        for behavior in corpus.train_behaviors:
            user_id = int(behavior[0])
            if 0 <= user_id < self.config.user_num:
                seen_users[user_id] = True
            items = [
                int(item)
                for item, mask in zip(behavior[1], behavior[2])
                if mask > 0 and int(item) > 0
            ]
            items.append(int(behavior[3]))
            for news_id in items:
                if 0 < news_id < self.config.news_num:
                    seen_news[news_id] = True
                    if 0 <= user_id < self.config.user_num:
                        edges.add((user_id, user_offset + news_id))
                        edges.add((user_offset + news_id, user_id))
        node_num = self.config.user_num + self.config.news_num
        self.seen_user_mask = seen_users
        self.seen_news_mask = seen_news
        if not edges:
            self.global_graph = torch.sparse_coo_tensor(
                torch.zeros((2, 0), dtype=torch.long),
                torch.zeros(0),
                (node_num, node_num),
            ).coalesce()
            return
        indices = torch.tensor(sorted(edges), dtype=torch.long).t()
        degrees = torch.bincount(indices[0], minlength=node_num).float().clamp_min(1)
        values = degrees[indices[0]].rsqrt() * degrees[indices[1]].rsqrt()
        self.global_graph = torch.sparse_coo_tensor(
            indices, values, (node_num, node_num)
        ).coalesce()

    def _global_embeddings(self):
        users = self.user_embedding.weight
        news = self.news_id_embedding.weight
        if self.seen_user_mask.numel() == self.config.user_num:
            users = users * self.seen_user_mask.to(users.device).unsqueeze(-1)
        if self.seen_news_mask.numel() == self.config.news_num:
            news = news * self.seen_news_mask.to(news.device).unsqueeze(-1)
        all_embeddings = torch.cat([users, news], dim=0)
        graph = self.global_graph
        if graph.numel() == 0:
            return users, news
        graph = graph.to(all_embeddings.device)
        layers = [all_embeddings]
        propagated = all_embeddings
        for _ in range(self.gcn_layer_num):
            propagated = torch.sparse.mm(graph, propagated)
            layers.append(propagated)
        combined = torch.stack(layers).sum(0)
        return combined[: self.config.user_num], combined[self.config.user_num :]

    def _stages(self, history, mask):
        history_len = history.size(1)
        positions = torch.arange(history_len, device=history.device)
        stage_ids = torch.clamp(
            positions * self.stage_num // max(history_len, 1), max=self.stage_num - 1
        )
        one_hot = F.one_hot(stage_ids, num_classes=self.stage_num).float()
        stage_mask = mask.unsqueeze(-1).float() * one_hot.unsqueeze(0)
        counts = stage_mask.sum(1).clamp_min(1)
        stages = torch.einsum("bhd,bht->btd", history, stage_mask) / counts.unsqueeze(-1)
        return stages, stage_mask.sum(1) > 0

    @staticmethod
    def _last_valid(sequence, valid):
        positions = torch.arange(valid.size(1), device=valid.device).unsqueeze(0)
        last = positions.expand_as(valid).masked_fill(~valid, -1).max(1).values
        index = last.clamp_min(0).view(-1, 1, 1).expand(-1, 1, sequence.size(-1))
        return sequence.gather(1, index).squeeze(1)

    def _long_range(self, history, mask):
        safe_mask = mask.bool()
        empty = ~safe_mask.any(1)
        if empty.any():
            safe_mask = safe_mask.clone()
            safe_mask[empty, 0] = True
        positions = self.position_embedding(
            torch.arange(history.size(1), device=history.device)
        ).unsqueeze(0)
        encoded = self.long_range_encoder(
            history + positions, src_key_padding_mask=~safe_mask
        )
        encoded = encoded * safe_mask.unsqueeze(-1)
        return encoded.sum(1) / safe_mask.sum(1, keepdim=True).clamp_min(1)

    def _candidate_attention(self, query, keys, values, mask):
        scores = (query.unsqueeze(2) * keys.unsqueeze(1)).sum(-1)
        scores = scores / math.sqrt(float(self.hidden_dim))
        scores = scores.masked_fill(~mask.unsqueeze(1), -1e9)
        empty = ~mask.any(1)
        if empty.any():
            scores = scores.clone()
            scores[empty] = 0
        attended = torch.matmul(F.softmax(scores, dim=-1), values)
        if empty.any():
            attended = attended.masked_fill(empty.view(-1, 1, 1), 0)
        return attended

    def _auxiliary(self, temporal_user, global_user, stages, stage_valid):
        temporal_user = F.normalize(temporal_user, dim=-1)
        global_user = F.normalize(global_user, dim=-1)
        contrast = torch.matmul(temporal_user, global_user.t()) / self.temperature
        labels = torch.arange(contrast.size(0), device=contrast.device)
        consistency = F.cross_entropy(contrast, labels)
        adjacent = stage_valid[:, 1:] & stage_valid[:, :-1]
        if adjacent.any():
            differences = (stages[:, 1:] - stages[:, :-1]).pow(2).sum(-1)
            smoothness = differences[adjacent].mean()
        else:
            smoothness = stages.new_tensor(0.0)
        return self.lambda_cl * consistency + self.lambda_sl * smoothness

    @staticmethod
    def _index_tensor(value, device):
        if value is None:
            return None
        if torch.is_tensor(value):
            return value.to(device=device, dtype=torch.long)
        if isinstance(value, (list, tuple)) and value and torch.is_tensor(value[0]):
            return torch.stack(list(value), dim=1).to(device=device, dtype=torch.long)
        return torch.as_tensor(value, device=device, dtype=torch.long)

    def forward(
        self,
        history_embedding,
        candidate_embedding,
        history_mask,
        user_id,
        history_index,
        news_index,
        **unused,
    ):
        del unused
        device = history_embedding.device
        history_index = self._index_tensor(history_index, device)
        news_index = self._index_tensor(news_index, device)
        if history_index is None or news_index is None:
            raise ValueError("SEIN composition requires history and candidate news indices")
        if news_index.dim() == 1:
            news_index = news_index.unsqueeze(1)
        global_users, global_news = self._global_embeddings()
        user_global = global_users[user_id]
        history = self.semantic_projection(history_embedding)
        history = self.dropout(
            history + self.news_id_embedding(history_index) + global_news[history_index]
        )
        stages, stage_valid = self._stages(history, history_mask)
        stage_sequence, _ = self.stage_lstm(stages)
        temporal_user = self._last_valid(stage_sequence, stage_valid)
        base_user = user_global + temporal_user + self._long_range(history, history_mask)
        candidate = self.semantic_projection(candidate_embedding)
        candidate = self.dropout(
            candidate + self.news_id_embedding(news_index) + global_news[news_index]
        )
        flat = candidate.reshape(-1, 1, self.hidden_dim)
        candidate_sequence, _ = self.item_lstm(flat)
        candidate = candidate_sequence.reshape_as(candidate) + global_news[news_index]
        history_match = self._candidate_attention(
            self.history_query(candidate),
            self.history_key(history),
            history,
            history_mask.bool(),
        )
        stage_match = self._candidate_attention(
            self.stage_query(candidate),
            self.stage_key(stage_sequence),
            stage_sequence,
            stage_valid,
        )
        expanded_user = base_user.unsqueeze(1).expand_as(candidate)
        gate = torch.sigmoid(
            self.user_gate(
                torch.cat(
                    [expanded_user, history_match, stage_match, candidate], dim=-1
                )
            )
        )
        candidate_user = gate * history_match + (1 - gate) * (
            expanded_user + stage_match
        )
        self.auxiliary_loss = self._auxiliary(
            temporal_user, user_global, stage_sequence, stage_valid
        )
        return self.output_projection(candidate_user)


_USER_MODULE_CLASSES = {
    "cnrcl": EncodedCNRCLUserModule,
    "nrms": EncodedNRMSUserModule,
    "sein": EncodedSEINUserModule,
    # EncodedCNRCLUserModule is the representation-level form of SUE: both
    # use the same proxy-category GCN and candidate-aware two-level attention.
    "sue": EncodedCNRCLUserModule,
}


class SOTAComposition(nn.Module):
    """One configuration-driven implementation for all strong variants B0--B7."""

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.model_name = config.model
        self.batch_size = config.batch_size
        self.composition_id = str(config.composition_id)
        self.news_encoder = NewsRepresentationStack(config)
        user_name = _normalise_name(config.user_encoder_type)
        try:
            user_class = _USER_MODULE_CLASSES[user_name]
        except KeyError as error:
            raise ValueError("Unknown user module %r" % user_name) from error
        self.user_encoder = user_class(
            config, self.news_encoder.news_embedding_dim
        )
        self.user_encoder_type = user_name
        self.training_strategy = _normalise_name(
            getattr(config, "training_strategy", "cnrcl")
        )
        if self.training_strategy not in available_sota_components()["training"]:
            raise ValueError(
                "Unknown training strategy %r" % self.training_strategy
            )

    @property
    def composition(self):
        return {
            "id": self.composition_id,
            "news_encoder": self.news_encoder.news_encoder_type,
            "user_encoder": self.user_encoder_type,
            "enhancement": self.news_encoder.enhancement_type,
            "training": self.training_strategy,
        }

    def initialize(self) -> None:
        self.news_encoder.initialize()
        self.user_encoder.initialize()

    def set_corpus(self, corpus) -> None:
        self.news_encoder.set_corpus(corpus)
        setter = getattr(self.user_encoder, "set_corpus", None)
        if callable(setter):
            setter(corpus)

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
        history_index=None,
        news_index=None,
        *extra_args,
    ):
        del extra_args
        candidate_values = [
            news_category,
            news_subCategory,
            news_title_text,
            news_title_mask,
            news_title_entity,
            news_content_text,
            news_content_mask,
            news_content_entity,
        ]
        # Dev/test supplies one candidate as [batch, ...]; training supplies
        # [batch, candidate_num, ...]. Use the title tensor to distinguish the
        # two contracts because category tensors have one fewer dimension.
        if news_title_text.dim() == 2:
            candidate_values = [value.unsqueeze(1) for value in candidate_values]
        if news_index is not None and torch.is_tensor(news_index) and news_index.dim() == 1:
            news_index = news_index.unsqueeze(1)
        history_features = (
            user_title_text,
            user_title_mask,
            user_title_entity,
            user_content_text,
            user_content_mask,
            user_content_entity,
            user_category,
            user_subCategory,
        )
        candidate_features = (
            candidate_values[2],
            candidate_values[3],
            candidate_values[4],
            candidate_values[5],
            candidate_values[6],
            candidate_values[7],
            candidate_values[0],
            candidate_values[1],
        )
        history_representation = self.news_encoder.encode(
            history_features, history_index
        )
        candidate_representation = self.news_encoder.encode(
            candidate_features, news_index
        )
        history_representation = history_representation * user_history_mask.unsqueeze(
            -1
        ).to(dtype=history_representation.dtype)
        user_representation = self.user_encoder(
            history_embedding=history_representation,
            candidate_embedding=candidate_representation,
            history_mask=user_history_mask,
            history_graph=user_history_graph,
            history_category_mask=user_history_category_mask,
            history_category_indices=user_history_category_indices,
            user_id=user_ID,
            history_index=history_index,
            news_index=news_index,
        )
        logits_unsummed = user_representation * candidate_representation
        logits = logits_unsummed.sum(dim=-1)
        logits_main = logits[:, : 1 + self.config.negative_sample_num]
        if self.training:
            auxiliary_loss = getattr(self.user_encoder, "auxiliary_loss", None)
            if auxiliary_loss is None:
                auxiliary_loss = logits.new_tensor(0.0)
            return logits_main, logits_unsummed, auxiliary_loss
        return logits_main.squeeze(1) if logits_main.size(1) == 1 else logits_main

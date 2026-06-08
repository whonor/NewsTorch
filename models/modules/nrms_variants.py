import hashlib
import json
import os
import re

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from config import Config
from models.modules.layers import Attention
from models.modules.newsEncoders import NewsEncoder
from utils import _transformers_compat


def _initialize_recurrent(module):
    for name, parameter in module.named_parameters():
        if "weight_ih" in name:
            nn.init.xavier_uniform_(parameter)
        elif "weight_hh" in name:
            nn.init.orthogonal_(parameter)
        elif "bias" in name:
            nn.init.zeros_(parameter)


def _flatten_news_inputs(title_text, title_mask, category, subcategory):
    if title_text.dim() == 2:
        title_text = title_text.unsqueeze(1)
        title_mask = title_mask.unsqueeze(1)
        category = category.unsqueeze(1)
        subcategory = subcategory.unsqueeze(1)
        squeeze_news_dim = True
    elif title_text.dim() == 3:
        squeeze_news_dim = False
    else:
        raise ValueError(
            "title_text must have shape [batch, length] or "
            f"[batch, news, length], got {tuple(title_text.shape)}"
        )

    batch_size, news_num, sequence_length = title_text.shape
    return (
        title_text.reshape(batch_size * news_num, sequence_length),
        title_mask.reshape(batch_size * news_num, sequence_length),
        category,
        subcategory,
        batch_size,
        news_num,
        squeeze_news_dim,
    )


def _restore_news_dim(news_representation, squeeze_news_dim):
    return news_representation.squeeze(1) if squeeze_news_dim else news_representation


def _masked_attention(attention, features, mask):
    valid_rows = mask.bool().any(dim=1)
    safe_mask = mask.clone()
    if (~valid_rows).any():
        safe_mask[~valid_rows, 0] = 1
    representation = attention(features, mask=safe_mask)
    return representation * valid_rows.unsqueeze(1).to(representation.dtype)


class CNNNewsEncoder(NewsEncoder):
    """NRMS-style title encoder using a temporal convolution."""

    def __init__(self, config: Config):
        super().__init__(config)
        self.text_embedding_dim = config.nrms_variant_text_dim
        self.kernel_size = config.nrms_cnn_kernel_size
        if self.kernel_size % 2 == 0:
            raise ValueError("nrms_cnn_kernel_size must be odd to preserve title length")
        self.conv = nn.Conv1d(
            config.word_embedding_dim,
            self.text_embedding_dim,
            kernel_size=self.kernel_size,
            padding=self.kernel_size // 2,
        )
        self.attention = Attention(self.text_embedding_dim, config.attention_dim)
        self.news_embedding_dim = (
            self.text_embedding_dim
            + config.category_embedding_dim
            + config.subCategory_embedding_dim
        )

    def initialize(self):
        super().initialize()
        nn.init.xavier_uniform_(self.conv.weight, gain=nn.init.calculate_gain("relu"))
        nn.init.zeros_(self.conv.bias)
        self.attention.initialize()

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
        user_embedding,
    ):
        (
            flat_title,
            flat_mask,
            category,
            subCategory,
            batch_size,
            news_num,
            squeeze_news_dim,
        ) = _flatten_news_inputs(title_text, title_mask, category, subCategory)

        word_vectors = self.dropout_(self.word_embedding(flat_title.long()))
        features = self.conv(word_vectors.transpose(1, 2)).transpose(1, 2)
        features = self.dropout_(F.relu(features))
        title_representation = _masked_attention(
            self.attention, features, flat_mask
        ).reshape(batch_size, news_num, self.text_embedding_dim)
        news_representation = self.feature_fusion(
            title_representation, category, subCategory
        )
        return _restore_news_dim(news_representation, squeeze_news_dim)


class RNNNewsEncoder(NewsEncoder):
    """NRMS-style title encoder using a bidirectional GRU."""

    def __init__(self, config: Config):
        super().__init__(config)
        self.hidden_dim = config.nrms_rnn_hidden_dim
        self.text_embedding_dim = self.hidden_dim * 2
        if self.text_embedding_dim != config.nrms_variant_text_dim:
            raise ValueError(
                "nrms_variant_text_dim must equal 2 * nrms_rnn_hidden_dim"
            )
        self.gru = nn.GRU(
            config.word_embedding_dim,
            self.hidden_dim,
            batch_first=True,
            bidirectional=True,
        )
        self.attention = Attention(self.text_embedding_dim, config.attention_dim)
        self.news_embedding_dim = (
            self.text_embedding_dim
            + config.category_embedding_dim
            + config.subCategory_embedding_dim
        )

    def initialize(self):
        super().initialize()
        _initialize_recurrent(self.gru)
        self.attention.initialize()

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
        user_embedding,
    ):
        (
            flat_title,
            flat_mask,
            category,
            subCategory,
            batch_size,
            news_num,
            squeeze_news_dim,
        ) = _flatten_news_inputs(title_text, title_mask, category, subCategory)

        word_vectors = self.dropout_(self.word_embedding(flat_title.long()))
        lengths = flat_mask.sum(dim=1).long()
        packed = pack_padded_sequence(
            word_vectors,
            lengths.clamp(min=1).cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        packed_features, _ = self.gru(packed)
        features, _ = pad_packed_sequence(
            packed_features,
            batch_first=True,
            total_length=flat_title.size(1),
        )
        features = self.dropout_(features)
        title_representation = _masked_attention(
            self.attention, features, flat_mask
        ).reshape(batch_size, news_num, self.text_embedding_dim)
        news_representation = self.feature_fusion(
            title_representation, category, subCategory
        )
        return _restore_news_dim(news_representation, squeeze_news_dim)


class _BaseUserEncoder(nn.Module):
    def __init__(self, news_encoder, config: Config):
        super().__init__()
        self.news_encoder = news_encoder
        self.news_embedding_dim = news_encoder.news_embedding_dim
        self.dropout = nn.Dropout(config.dropout_rate)
        self.auxiliary_loss = None

    def _encode_history(
        self,
        user_title_text,
        user_title_mask,
        user_title_entity,
        user_content_text,
        user_content_mask,
        user_content_entity,
        user_category,
        user_subCategory,
        user_history_mask,
        user_embedding,
    ):
        history = self.news_encoder(
            user_title_text,
            user_title_mask,
            user_title_entity,
            user_content_text,
            user_content_mask,
            user_content_entity,
            user_category,
            user_subCategory,
            user_embedding,
        )
        return history * user_history_mask.unsqueeze(2).to(history.dtype)


class CNNUserEncoder(_BaseUserEncoder):
    """Encode clicked-news sequences with a temporal convolution."""

    def __init__(self, news_encoder, config: Config):
        super().__init__(news_encoder, config)
        self.attention = Attention(
            self.news_embedding_dim, config.attention_dim
        )
        kernel_size = config.nrms_user_cnn_kernel_size
        if kernel_size % 2 == 0:
            raise ValueError(
                "nrms_user_cnn_kernel_size must be odd to preserve history length"
            )
        self.conv = nn.Conv1d(
            self.news_embedding_dim,
            self.news_embedding_dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )

    def initialize(self):
        nn.init.xavier_uniform_(self.conv.weight, gain=nn.init.calculate_gain("relu"))
        nn.init.zeros_(self.conv.bias)
        self.attention.initialize()

    def forward(
        self,
        user_title_text,
        user_title_mask,
        user_title_entity,
        user_content_text,
        user_content_mask,
        user_content_entity,
        user_category,
        user_subCategory,
        user_history_mask,
        user_history_graph,
        user_history_category_mask,
        user_history_category_indices,
        user_embedding,
        candidate_news_representation,
    ):
        history = self._encode_history(
            user_title_text,
            user_title_mask,
            user_title_entity,
            user_content_text,
            user_content_mask,
            user_content_entity,
            user_category,
            user_subCategory,
            user_history_mask,
            user_embedding,
        )
        features = self.conv(history.transpose(1, 2)).transpose(1, 2)
        features = self.dropout(F.relu(features))
        return _masked_attention(self.attention, features, user_history_mask)


class RNNUserEncoder(_BaseUserEncoder):
    """Encode clicked-news sequences with a GRU."""

    def __init__(self, news_encoder, config: Config):
        super().__init__(news_encoder, config)
        self.gru = nn.GRU(
            self.news_embedding_dim,
            self.news_embedding_dim,
            batch_first=True,
        )

    def initialize(self):
        _initialize_recurrent(self.gru)

    def forward(
        self,
        user_title_text,
        user_title_mask,
        user_title_entity,
        user_content_text,
        user_content_mask,
        user_content_entity,
        user_category,
        user_subCategory,
        user_history_mask,
        user_history_graph,
        user_history_category_mask,
        user_history_category_indices,
        user_embedding,
        candidate_news_representation,
    ):
        history = self._encode_history(
            user_title_text,
            user_title_mask,
            user_title_entity,
            user_content_text,
            user_content_mask,
            user_content_entity,
            user_category,
            user_subCategory,
            user_history_mask,
            user_embedding,
        )
        lengths = user_history_mask.sum(dim=1).long()
        non_empty = lengths > 0
        representation = history.new_zeros(
            history.size(0), self.news_embedding_dim
        )
        if non_empty.any():
            indices = non_empty.nonzero(as_tuple=False).squeeze(1)
            packed = pack_padded_sequence(
                history.index_select(0, indices),
                lengths.index_select(0, indices).cpu(),
                batch_first=True,
                enforce_sorted=False,
            )
            _, hidden = self.gru(packed)
            representation = representation.index_copy(
                0, indices, hidden[-1]
            )
        return representation


def _default_vocabulary_path(config):
    filename = (
        f"vocabulary-{config.word_threshold}-{config.tokenizer}-"
        f"{config.max_title_length}-{config.max_abstract_length}-"
        f"{config.dataset_size}.json"
    )
    if config.dataset_name == "ebnerd":
        return os.path.join("cache", "ebnerd", filename)
    return os.path.join("cache", filename)


def _safe_model_name(model_name):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model_name)


class PLMNewsEncoder(nn.Module):
    """
    Contextualize the repository's word-level IDs with a pretrained language model.

    The corpus uses a custom vocabulary, so its IDs cannot be sent directly to a
    Hugging Face model. Each corpus word is first aligned to the mean of its PLM
    subword embeddings. The resulting sequence is then passed through the PLM by
    using ``inputs_embeds``.
    """

    def __init__(self, config: Config):
        super().__init__()
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "transformers is required for the NRMS-PLM variant"
            ) from exc

        self.config = config
        self.model_name_or_path = config.plm_model_name_or_path
        self.freeze_plm = bool(config.plm_freeze)
        self.freeze_vocabulary_mapping = bool(
            config.plm_freeze_vocabulary_mapping
        )
        hf_kwargs = {
            "local_files_only": bool(config.plm_local_files_only),
            "trust_remote_code": bool(config.plm_trust_remote_code),
        }
        token = getattr(config, "plm_hf_token", None)
        if not token:
            token = os.getenv(getattr(config, "plm_hf_token_env", "HF_TOKEN"))
        if token:
            hf_kwargs["token"] = token

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name_or_path, **hf_kwargs
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = (
                self.tokenizer.eos_token or self.tokenizer.unk_token
            )
        if self.tokenizer.pad_token_id is None:
            raise ValueError(
                "The PLM tokenizer needs a pad, eos, or unk token"
            )
        self.transformer = AutoModel.from_pretrained(
            self.model_name_or_path, **hf_kwargs
        )
        self.plm_hidden_dim = self.transformer.config.hidden_size

        vocabulary_path = getattr(
            config, "plm_vocabulary_path", _default_vocabulary_path(config)
        )
        vocabulary_mapping = self._load_or_build_vocabulary_mapping(
            vocabulary_path
        )
        self.word_embedding = nn.Embedding.from_pretrained(
            vocabulary_mapping,
            freeze=self.freeze_vocabulary_mapping,
            padding_idx=0,
        )

        for parameter in self.transformer.get_input_embeddings().parameters():
            parameter.requires_grad = False
        if self.freeze_plm:
            for parameter in self.transformer.parameters():
                parameter.requires_grad = False
        elif bool(getattr(config, "plm_gradient_checkpointing", False)):
            self.transformer.gradient_checkpointing_enable()

        self.text_embedding_dim = config.nrms_variant_text_dim
        self.projection = nn.Linear(self.plm_hidden_dim, self.text_embedding_dim)
        self.attention = Attention(self.text_embedding_dim, config.attention_dim)
        self.category_embedding = nn.Embedding(
            config.category_num, config.category_embedding_dim
        )
        self.subCategory_embedding = nn.Embedding(
            config.subCategory_num, config.subCategory_embedding_dim
        )
        self.dropout = nn.Dropout(config.dropout_rate)
        self.news_embedding_dim = (
            self.text_embedding_dim
            + config.category_embedding_dim
            + config.subCategory_embedding_dim
        )
        self.auxiliary_loss = None

    def _mapping_cache_path(self, vocabulary):
        configured_path = getattr(self.config, "plm_mapping_cache_path", "")
        if configured_path:
            return configured_path
        signature = hashlib.sha1(
            json.dumps(vocabulary, sort_keys=True).encode("utf-8")
        ).hexdigest()[:12]
        filename = (
            f"{_safe_model_name(self.model_name_or_path)}-"
            f"{self.config.dataset_name}-{self.config.dataset_size}-"
            f"{signature}.pt"
        )
        return os.path.join("cache", "nrms_plm", filename)

    def _load_or_build_vocabulary_mapping(self, vocabulary_path):
        if not os.path.exists(vocabulary_path):
            raise FileNotFoundError(
                "NRMS-PLM needs the corpus vocabulary cache before model "
                f"construction: {vocabulary_path}"
            )
        with open(vocabulary_path, "r", encoding="utf-8") as vocabulary_file:
            vocabulary = json.load(vocabulary_file)

        cache_path = self._mapping_cache_path(vocabulary)
        if os.path.exists(cache_path):
            mapping = torch.load(cache_path, map_location="cpu")
            expected_shape = (len(vocabulary), self.plm_hidden_dim)
            if tuple(mapping.shape) == expected_shape:
                return mapping

        tokens = ["<UNK>"] * len(vocabulary)
        for token, index in vocabulary.items():
            tokens[index] = token
        unknown_token = getattr(self.tokenizer, "unk_token", None) or "<UNK>"
        normalized_tokens = []
        for token in tokens:
            if token == "<NUM>":
                normalized_tokens.append("0")
            elif token == "<UNK>":
                normalized_tokens.append(unknown_token)
            else:
                normalized_tokens.append(token)

        input_embedding = self.transformer.get_input_embeddings()
        mapping_batches = []
        batch_size = int(getattr(self.config, "plm_mapping_batch_size", 512))
        with torch.no_grad():
            for start in range(0, len(normalized_tokens), batch_size):
                encoded = self.tokenizer(
                    normalized_tokens[start:start + batch_size],
                    add_special_tokens=False,
                    padding=True,
                    truncation=True,
                    max_length=int(
                        getattr(self.config, "plm_mapping_max_subwords", 8)
                    ),
                    return_tensors="pt",
                )
                input_ids = encoded["input_ids"]
                attention_mask = encoded["attention_mask"].unsqueeze(2)
                subword_embeddings = input_embedding(input_ids)
                denominator = attention_mask.sum(dim=1).clamp(min=1)
                mapping_batches.append(
                    (subword_embeddings * attention_mask).sum(dim=1)
                    / denominator
                )
        mapping = torch.cat(mapping_batches, dim=0).cpu()
        mapping[0].zero_()
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        torch.save(mapping, cache_path)
        return mapping

    def initialize(self):
        nn.init.xavier_uniform_(
            self.projection.weight, gain=nn.init.calculate_gain("tanh")
        )
        nn.init.zeros_(self.projection.bias)
        self.attention.initialize()
        nn.init.uniform_(self.category_embedding.weight, -0.1, 0.1)
        nn.init.uniform_(self.subCategory_embedding.weight, -0.1, 0.1)

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
        user_embedding,
    ):
        (
            flat_title,
            flat_mask,
            category,
            subCategory,
            batch_size,
            news_num,
            squeeze_news_dim,
        ) = _flatten_news_inputs(title_text, title_mask, category, subCategory)

        inputs_embeds = self.word_embedding(flat_title.long())
        valid_rows = flat_mask.bool().any(dim=1)
        safe_mask = flat_mask.clone()
        if (~valid_rows).any():
            safe_mask[~valid_rows, 0] = 1
        transformer_kwargs = {
            "inputs_embeds": inputs_embeds,
            "attention_mask": safe_mask.long(),
            "return_dict": True,
        }
        if self.freeze_plm and self.freeze_vocabulary_mapping:
            with torch.no_grad():
                contextualized = self.transformer(
                    **transformer_kwargs
                ).last_hidden_state
        else:
            contextualized = self.transformer(
                **transformer_kwargs
            ).last_hidden_state

        features = self.dropout(torch.tanh(self.projection(contextualized)))
        title_representation = _masked_attention(
            self.attention, features, flat_mask
        ).reshape(batch_size, news_num, self.text_embedding_dim)
        category_representation = self.dropout(
            self.category_embedding(category.long())
        )
        subcategory_representation = self.dropout(
            self.subCategory_embedding(subCategory.long())
        )
        news_representation = torch.cat(
            [
                title_representation,
                category_representation,
                subcategory_representation,
            ],
            dim=2,
        )
        return _restore_news_dim(news_representation, squeeze_news_dim)


class TransformerUserEncoder(_BaseUserEncoder):
    """Encode clicked-news sequences with a Transformer encoder."""

    def __init__(self, news_encoder, config: Config):
        super().__init__(news_encoder, config)
        self.attention = Attention(
            self.news_embedding_dim, config.attention_dim
        )
        head_num = config.nrms_user_transformer_heads
        if self.news_embedding_dim % head_num != 0:
            raise ValueError(
                "news embedding dimension must be divisible by "
                "nrms_user_transformer_heads"
            )
        layer = nn.TransformerEncoderLayer(
            d_model=self.news_embedding_dim,
            nhead=head_num,
            dim_feedforward=config.nrms_user_transformer_ffn_dim,
            dropout=config.dropout_rate,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer, num_layers=config.nrms_user_transformer_layers
        )
        self.position_embedding = nn.Parameter(
            torch.empty(config.max_history_num, self.news_embedding_dim)
        )

    def initialize(self):
        nn.init.normal_(self.position_embedding, mean=0.0, std=0.02)
        self.attention.initialize()

    def forward(
        self,
        user_title_text,
        user_title_mask,
        user_title_entity,
        user_content_text,
        user_content_mask,
        user_content_entity,
        user_category,
        user_subCategory,
        user_history_mask,
        user_history_graph,
        user_history_category_mask,
        user_history_category_indices,
        user_embedding,
        candidate_news_representation,
    ):
        history = self._encode_history(
            user_title_text,
            user_title_mask,
            user_title_entity,
            user_content_text,
            user_content_mask,
            user_content_entity,
            user_category,
            user_subCategory,
            user_history_mask,
            user_embedding,
        )
        valid_rows = user_history_mask.bool().any(dim=1)
        safe_mask = user_history_mask.bool().clone()
        if (~valid_rows).any():
            safe_mask[~valid_rows, 0] = True

        history = history + self.position_embedding[:history.size(1)].unsqueeze(0)
        features = self.transformer(
            history,
            src_key_padding_mask=~safe_mask,
        )
        features = features * user_history_mask.unsqueeze(2).to(features.dtype)
        representation = _masked_attention(
            self.attention, features, user_history_mask
        )
        return representation * valid_rows.unsqueeze(1).to(
            representation.dtype
        )

import hashlib
import json
import os
import re

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Config
from models.modules.layers import Attention, MultiHeadAttention
from utils import _transformers_compat


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


def _default_vocabulary_path(config):
    filename = (
        f"vocabulary-{config.word_threshold}-{config.tokenizer}-"
        f"{config.max_title_length}-{config.max_abstract_length}-"
        f"{config.dataset_size}.json"
    )
    dataset_name = config.dataset_name.lower()
    if dataset_name in {"ebnerd", "adressa"}:
        return os.path.join("cache", dataset_name, filename)
    return os.path.join("cache", filename)


def _safe_model_name(model_name):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model_name)


class PLMNewsEncoder(nn.Module):
    """
    Contextualize the repository's word-level IDs with a pretrained language model.

    The corpus uses custom word IDs, which cannot be sent directly to RoBERTa.
    Each corpus word is therefore aligned to the mean of its RoBERTa subword
    embeddings. The resulting title sequence is contextualized through RoBERTa
    via ``inputs_embeds`` and pooled with the paper's additive attention layer.
    """

    def __init__(self, config: Config):
        super().__init__()
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "transformers is required for the PLM-NR variant"
            ) from exc

        self.config = config
        self.model_name_or_path = getattr(
            config, "plm_model_name_or_path", "roberta-base"
        )
        self.freeze_plm = bool(getattr(config, "plm_freeze", False))
        self.freeze_vocabulary_mapping = bool(
            getattr(config, "plm_freeze_vocabulary_mapping", True)
        )
        hf_kwargs = {
            "local_files_only": bool(
                getattr(config, "plm_local_files_only", False)
            ),
            "trust_remote_code": bool(
                getattr(config, "plm_trust_remote_code", False)
            ),
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

        vocabulary_path = (
            getattr(config, "plm_vocabulary_path", "")
            or _default_vocabulary_path(config)
        )
        vocabulary_mapping = self._load_or_build_vocabulary_mapping(
            vocabulary_path
        )
        self.word_embedding = nn.Embedding.from_pretrained(
            vocabulary_mapping,
            freeze=self.freeze_vocabulary_mapping,
            padding_idx=0,
        )

        self._configure_finetuning()
        self.gradient_checkpointing = (
            not self.freeze_plm
            and bool(getattr(config, "plm_gradient_checkpointing", False))
        )
        if self.gradient_checkpointing:
            self.transformer.gradient_checkpointing_enable()

        self.text_embedding_dim = int(
            getattr(config, "nrms_variant_text_dim", self.plm_hidden_dim)
        )
        self.projection = (
            nn.Identity()
            if self.text_embedding_dim == self.plm_hidden_dim
            else nn.Linear(self.plm_hidden_dim, self.text_embedding_dim)
        )
        self.attention = Attention(self.text_embedding_dim, config.attention_dim)
        self.dropout = nn.Dropout(config.dropout_rate)
        # PLM-NR models news from titles only; category features are deliberately
        # excluded to match the paper's RoBERTa-NRMS setup.
        self.news_embedding_dim = self.text_embedding_dim
        self.auxiliary_loss = None

    def _transformer_layers(self):
        """Return the ordered Transformer blocks for common HF base models."""
        modules = [self.transformer]
        base_model = getattr(self.transformer, "base_model", None)
        if base_model is not None and base_model is not self.transformer:
            modules.append(base_model)
        for module in modules:
            encoder = getattr(module, "encoder", None)
            layers = getattr(encoder, "layer", None)
            if layers is not None:
                return layers
            transformer = getattr(module, "transformer", None)
            layers = getattr(transformer, "layer", None)
            if layers is not None:
                return layers
        return None

    def _configure_finetuning(self):
        """Fine-tune only the final PLM blocks, as done in PLM-NR."""
        for parameter in self.transformer.parameters():
            parameter.requires_grad = False
        if self.freeze_plm:
            return

        trainable_layer_count = int(
            getattr(self.config, "plm_trainable_layers", 2)
        )
        layers = self._transformer_layers()
        if layers is None:
            # Lightweight/custom models used in tests may not expose the common
            # encoder.layer structure. Keep their non-embedding body trainable.
            for parameter in self.transformer.parameters():
                parameter.requires_grad = True
        elif trainable_layer_count > 0:
            for layer in layers[-trainable_layer_count:]:
                for parameter in layer.parameters():
                    parameter.requires_grad = True

        # ``inputs_embeds`` supplies the fixed corpus-to-RoBERTa mapping, so the
        # original token embedding table is not part of optimization.
        for parameter in self.transformer.get_input_embeddings().parameters():
            parameter.requires_grad = False

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
                "PLM-NR needs the corpus vocabulary cache before model "
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
        if isinstance(self.projection, nn.Linear):
            nn.init.xavier_uniform_(
                self.projection.weight, gain=nn.init.calculate_gain("tanh")
            )
            nn.init.zeros_(self.projection.bias)
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

        inputs_embeds = self.word_embedding(flat_title.long())
        if self.training and self.gradient_checkpointing:
            # Re-entrant PyTorch checkpointing needs at least one differentiable
            # input in order to propagate into the trainable final PLM layers.
            inputs_embeds = inputs_embeds.detach().requires_grad_(True)
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

        features = self.projection(contextualized)
        if isinstance(self.projection, nn.Linear):
            features = torch.tanh(features)
        features = self.dropout(features)
        title_representation = _masked_attention(
            self.attention, features, flat_mask
        ).reshape(batch_size, news_num, self.text_embedding_dim)
        return _restore_news_dim(title_representation, squeeze_news_dim)


class NRMSUserEncoder(_BaseUserEncoder):
    """The NRMS multi-head self-attention and additive-attention user encoder."""

    def __init__(self, news_encoder, config: Config):
        super().__init__(news_encoder, config)
        head_num = int(getattr(config, "head_num", 20))
        head_dim = int(getattr(config, "head_dim", 20))
        self.self_attention = MultiHeadAttention(
            head_num,
            self.news_embedding_dim,
            config.max_history_num,
            config.max_history_num,
            head_dim,
            head_dim,
        )
        self.projection = nn.Linear(
            head_num * head_dim, self.news_embedding_dim
        )
        self.attention = Attention(self.news_embedding_dim, config.attention_dim)

    def initialize(self):
        self.self_attention.initialize()
        nn.init.xavier_uniform_(
            self.projection.weight, gain=nn.init.calculate_gain("relu")
        )
        nn.init.zeros_(self.projection.bias)
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

        features = self.self_attention(
            history, history, history, safe_mask
        )
        features = self.dropout(F.relu(self.projection(features)))
        features = features * user_history_mask.unsqueeze(2).to(features.dtype)
        representation = _masked_attention(
            self.attention, features, user_history_mask
        )
        return representation * valid_rows.unsqueeze(1).to(
            representation.dtype
        )

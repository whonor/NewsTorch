import os
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedAttentionPool(nn.Module):
    def __init__(self, input_dim: int, attention_dim: int):
        super().__init__()
        self.affine = nn.Linear(input_dim, attention_dim)
        self.query = nn.Linear(attention_dim, 1, bias=False)

    def forward(self, feature, mask=None):
        score = self.query(torch.tanh(self.affine(feature))).squeeze(-1)
        if mask is not None:
            mask = mask.to(dtype=torch.bool, device=feature.device)
            score = score.masked_fill(~mask, -1e9)
        alpha = F.softmax(score, dim=-1).unsqueeze(1)
        return torch.bmm(alpha, feature).squeeze(1)


class CoAttentionBlock(nn.Module):
    def __init__(self, hidden_dim: int, head_num: int, dropout: float):
        super().__init__()
        self.text_to_image = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=head_num,
            dropout=dropout,
            batch_first=True,
        )
        self.image_to_text = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=head_num,
            dropout=dropout,
            batch_first=True,
        )
        self.text_norm = nn.LayerNorm(hidden_dim)
        self.image_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text_feature, image_feature, text_mask=None, image_mask=None):
        text_padding_mask = None
        image_padding_mask = None
        if text_mask is not None:
            text_padding_mask = ~text_mask.to(dtype=torch.bool, device=text_feature.device)
        if image_mask is not None:
            image_padding_mask = ~image_mask.to(dtype=torch.bool, device=image_feature.device)

        text_context, _ = self.text_to_image(
            query=text_feature,
            key=image_feature,
            value=image_feature,
            key_padding_mask=image_padding_mask,
            need_weights=False,
        )
        image_context, _ = self.image_to_text(
            query=image_feature,
            key=text_feature,
            value=text_feature,
            key_padding_mask=text_padding_mask,
            need_weights=False,
        )

        text_feature = self.text_norm(text_feature + self.dropout(text_context))
        image_feature = self.image_norm(image_feature + self.dropout(image_context))
        return text_feature, image_feature


class NewsEncoder(nn.Module):
    """MM-Rec news encoder without loading external pretrained model weights.

    The paper encodes text and image features jointly with a visiolinguistic
    encoder. This implementation keeps the same multimodal/co-attention shape,
    but all trainable parameters are initialized inside this repository.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.max_title_length = config.max_title_length
        self.hidden_dim = getattr(config, "mmrec_hidden_dim", getattr(config, "bi_hidden_size", 400))
        self.vocab_size = getattr(config, "vocabulary_size", getattr(config, "vocab_size", 30522))
        self.image_embedding_dim = getattr(config, "image_embedding_dim", 2048)
        self.dropout_prob = getattr(config, "dropout_prob", getattr(config, "dropout_rate", 0.1))
        self.pretrained_word_embedding_path = self._word_embedding_path(config)
        pretrained_word_embedding = self._load_pretrained_word_embedding()
        self.word_embedding_dim = (
            pretrained_word_embedding.size(1)
            if pretrained_word_embedding is not None
            else getattr(config, "word_embedding_dim", self.hidden_dim)
        )
        self._pretrained_word_embedding = pretrained_word_embedding

        head_num = getattr(config, "mmrec_head_num", getattr(config, "head_num", 8))
        head_num = max(1, min(head_num, self.hidden_dim))
        while self.hidden_dim % head_num != 0:
            head_num -= 1

        attention_dim = getattr(config, "attention_dim", self.hidden_dim)
        layer_num = getattr(config, "mmrec_coattention_layers", 1)

        self.word_embedding = nn.Embedding(self.vocab_size, self.word_embedding_dim, padding_idx=0)
        self.text_projection = nn.Linear(self.word_embedding_dim, self.hidden_dim)
        self.image_projection = nn.Linear(self.image_embedding_dim, self.hidden_dim)
        self.use_metadata = getattr(config, "mmrec_use_metadata", True) and all(
            hasattr(config, attr)
            for attr in ("category_num", "subCategory_num", "category_embedding_dim", "subCategory_embedding_dim")
        )
        if self.use_metadata:
            self.category_embedding = nn.Embedding(config.category_num, config.category_embedding_dim)
            self.subCategory_embedding = nn.Embedding(config.subCategory_num, config.subCategory_embedding_dim)
            self.metadata_projection = nn.Linear(
                config.category_embedding_dim + config.subCategory_embedding_dim,
                self.hidden_dim,
            )
            self.metadata_norm = nn.LayerNorm(self.hidden_dim)
        self.position_embedding = nn.Embedding(self.max_title_length, self.hidden_dim)
        self.text_self_attention = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=head_num,
            dropout=self.dropout_prob,
            batch_first=True,
        )
        self.image_self_attention = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=head_num,
            dropout=self.dropout_prob,
            batch_first=True,
        )
        self.coattention_layers = nn.ModuleList(
            CoAttentionBlock(self.hidden_dim, head_num, self.dropout_prob)
            for _ in range(layer_num)
        )
        self.text_norm = nn.LayerNorm(self.hidden_dim)
        self.image_norm = nn.LayerNorm(self.hidden_dim)
        self.text_pooler = MaskedAttentionPool(self.hidden_dim, attention_dim)
        self.image_pooler = MaskedAttentionPool(self.hidden_dim, attention_dim)
        self.dropout = nn.Dropout(self.dropout_prob)

        self.initialize()

    def _word_embedding_path(self, config):
        file_name = (
            'word_embedding-'
            + str(config.word_threshold)
            + '-'
            + str(config.word_embedding_dim)
            + '-'
            + config.tokenizer
            + '-'
            + str(config.max_title_length)
            + '-'
            + str(config.max_abstract_length)
            + '-'
            + config.dataset_size
            + '.pkl'
        )
        if getattr(config, "dataset_name", None) == 'MIND':
            if getattr(config, "model", None) == 'IPNR':
                return os.path.join('cache', 'IPNR', file_name)
            return os.path.join('cache', file_name)
        if getattr(config, "dataset_name", None) == 'ebnerd':
            return os.path.join('cache', config.dataset_name, file_name)
        return None

    def _load_pretrained_word_embedding(self):
        if not getattr(self.config, "mmrec_use_pretrained_word_embedding", True):
            return None
        if self.pretrained_word_embedding_path is None or not os.path.exists(self.pretrained_word_embedding_path):
            return None
        with open(self.pretrained_word_embedding_path, 'rb') as word_embedding_f:
            embedding = pickle.load(word_embedding_f)
        embedding = torch.as_tensor(embedding, dtype=torch.float32)
        if embedding.dim() != 2:
            raise ValueError(
                f"Expected a 2D word embedding matrix at {self.pretrained_word_embedding_path}, "
                f"got shape {tuple(embedding.shape)}."
            )
        return embedding

    def initialize(self):
        pretrained_word_embedding = self._pretrained_word_embedding
        if pretrained_word_embedding is None:
            pretrained_word_embedding = self._load_pretrained_word_embedding()

        if pretrained_word_embedding is None:
            nn.init.normal_(self.word_embedding.weight, mean=0.0, std=0.02)
        else:
            nn.init.normal_(self.word_embedding.weight, mean=0.0, std=0.02)
            row_num = min(self.word_embedding.weight.size(0), pretrained_word_embedding.size(0))
            col_num = min(self.word_embedding.weight.size(1), pretrained_word_embedding.size(1))
            with torch.no_grad():
                self.word_embedding.weight[:row_num, :col_num].copy_(pretrained_word_embedding[:row_num, :col_num])
        nn.init.zeros_(self.word_embedding.weight[0])
        self.word_embedding.weight.requires_grad = not getattr(self.config, "mmrec_freeze_word_embedding", False)
        nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)
        if self.use_metadata:
            nn.init.uniform_(self.category_embedding.weight, -0.1, 0.1)
            nn.init.uniform_(self.subCategory_embedding.weight, -0.1, 0.1)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        self._pretrained_word_embedding = None

    def _flatten_news(self, input_ids, attention_mask, input_imgs, image_attention_mask, category, subCategory):
        if input_ids.dim() == 3:
            batch_size, news_num, seq_len = input_ids.shape
            input_ids = input_ids.reshape(batch_size * news_num, seq_len)
            if attention_mask is not None:
                attention_mask = attention_mask.reshape(batch_size * news_num, seq_len)
            if category is not None:
                category = category.reshape(batch_size * news_num)
            if subCategory is not None:
                subCategory = subCategory.reshape(batch_size * news_num)

            if input_imgs.dim() == 3:
                input_imgs = input_imgs.unsqueeze(2)
            region_num, image_dim = input_imgs.shape[-2], input_imgs.shape[-1]
            input_imgs = input_imgs.reshape(batch_size * news_num, region_num, image_dim)
            if image_attention_mask is not None:
                image_attention_mask = image_attention_mask.reshape(batch_size * news_num, region_num)
            reshape_to = (batch_size, news_num)
        else:
            reshape_to = None
            if category is not None:
                category = category.reshape(-1)
            if subCategory is not None:
                subCategory = subCategory.reshape(-1)
            if input_imgs.dim() == 2:
                input_imgs = input_imgs.unsqueeze(1)

        return input_ids, attention_mask, input_imgs, image_attention_mask, category, subCategory, reshape_to

    def _add_metadata(self, pooled_text, category, subCategory):
        if not self.use_metadata or category is None or subCategory is None:
            return pooled_text

        category = category.to(dtype=torch.long, device=pooled_text.device).clamp(
            min=0,
            max=self.category_embedding.num_embeddings - 1,
        )
        subCategory = subCategory.to(dtype=torch.long, device=pooled_text.device).clamp(
            min=0,
            max=self.subCategory_embedding.num_embeddings - 1,
        )
        metadata = torch.cat(
            [self.category_embedding(category), self.subCategory_embedding(subCategory)],
            dim=-1,
        )
        metadata = self.metadata_projection(self.dropout(metadata))
        return self.metadata_norm(pooled_text + metadata)

    def forward(
        self,
        input_ids,
        input_imgs,
        category=None,
        subCategory=None,
        image_loc=None,
        token_type_ids=None,
        attention_mask=None,
        image_attention_mask=None,
        co_attention_mask=None,
        output_all_encoded_layers=False,
    ):
        del image_loc, token_type_ids, co_attention_mask, output_all_encoded_layers

        input_ids, attention_mask, input_imgs, image_attention_mask, category, subCategory, reshape_to = self._flatten_news(
            input_ids,
            attention_mask,
            input_imgs,
            image_attention_mask,
            category,
            subCategory,
        )

        if attention_mask is None:
            attention_mask = input_ids.ne(0)
        else:
            attention_mask = attention_mask.to(dtype=torch.bool, device=input_ids.device)

        if image_attention_mask is None:
            image_attention_mask = torch.ones(input_imgs.shape[:-1], dtype=torch.bool, device=input_imgs.device)
        else:
            image_attention_mask = image_attention_mask.to(dtype=torch.bool, device=input_imgs.device)

        position_ids = torch.arange(input_ids.size(1), device=input_ids.device).unsqueeze(0)
        text_feature = self.word_embedding(input_ids)
        text_feature = self.text_projection(text_feature)
        text_feature = text_feature + self.position_embedding(position_ids)
        text_feature = self.dropout(text_feature)

        text_self, _ = self.text_self_attention(
            query=text_feature,
            key=text_feature,
            value=text_feature,
            key_padding_mask=~attention_mask,
            need_weights=False,
        )
        text_feature = self.text_norm(text_feature + self.dropout(text_self))

        image_feature = self.dropout(self.image_projection(input_imgs.float()))
        image_self, _ = self.image_self_attention(
            query=image_feature,
            key=image_feature,
            value=image_feature,
            key_padding_mask=~image_attention_mask,
            need_weights=False,
        )
        image_feature = self.image_norm(image_feature + self.dropout(image_self))

        for layer in self.coattention_layers:
            text_feature, image_feature = layer(text_feature, image_feature, attention_mask, image_attention_mask)

        pooled_text = self.text_pooler(text_feature, attention_mask)
        pooled_image = self.image_pooler(image_feature, image_attention_mask)
        pooled_text = self._add_metadata(pooled_text, category, subCategory)

        if reshape_to is not None:
            pooled_text = pooled_text.view(*reshape_to, self.hidden_dim)
            pooled_image = pooled_image.view(*reshape_to, self.hidden_dim)

        return pooled_text, pooled_image

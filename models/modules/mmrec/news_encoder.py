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
        self.max_title_length = config.max_title_length
        self.hidden_dim = getattr(config, "mmrec_hidden_dim", getattr(config, "bi_hidden_size", 400))
        self.word_embedding_dim = getattr(config, "word_embedding_dim", self.hidden_dim)
        self.vocab_size = getattr(config, "vocabulary_size", getattr(config, "vocab_size", 30522))
        self.image_embedding_dim = getattr(config, "image_embedding_dim", 2048)
        self.dropout_prob = getattr(config, "dropout_prob", getattr(config, "dropout_rate", 0.1))

        head_num = getattr(config, "mmrec_head_num", getattr(config, "head_num", 8))
        head_num = max(1, min(head_num, self.hidden_dim))
        while self.hidden_dim % head_num != 0:
            head_num -= 1

        attention_dim = getattr(config, "attention_dim", self.hidden_dim)
        layer_num = getattr(config, "mmrec_coattention_layers", 1)

        self.word_embedding = nn.Embedding(self.vocab_size, self.word_embedding_dim, padding_idx=0)
        self.text_projection = nn.Linear(self.word_embedding_dim, self.hidden_dim)
        self.image_projection = nn.Linear(self.image_embedding_dim, self.hidden_dim)
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

    def initialize(self):
        nn.init.normal_(self.word_embedding.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.word_embedding.weight[0])
        nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _flatten_news(self, input_ids, attention_mask, input_imgs, image_attention_mask):
        if input_ids.dim() == 3:
            batch_size, news_num, seq_len = input_ids.shape
            input_ids = input_ids.reshape(batch_size * news_num, seq_len)
            if attention_mask is not None:
                attention_mask = attention_mask.reshape(batch_size * news_num, seq_len)

            if input_imgs.dim() == 3:
                input_imgs = input_imgs.unsqueeze(2)
            region_num, image_dim = input_imgs.shape[-2], input_imgs.shape[-1]
            input_imgs = input_imgs.reshape(batch_size * news_num, region_num, image_dim)
            if image_attention_mask is not None:
                image_attention_mask = image_attention_mask.reshape(batch_size * news_num, region_num)
            reshape_to = (batch_size, news_num)
        else:
            reshape_to = None
            if input_imgs.dim() == 2:
                input_imgs = input_imgs.unsqueeze(1)

        return input_ids, attention_mask, input_imgs, image_attention_mask, reshape_to

    def forward(
        self,
        input_ids,
        input_imgs,
        image_loc=None,
        token_type_ids=None,
        attention_mask=None,
        image_attention_mask=None,
        co_attention_mask=None,
        output_all_encoded_layers=False,
    ):
        del image_loc, token_type_ids, co_attention_mask, output_all_encoded_layers

        input_ids, attention_mask, input_imgs, image_attention_mask, reshape_to = self._flatten_news(
            input_ids,
            attention_mask,
            input_imgs,
            image_attention_mask,
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

        if reshape_to is not None:
            pooled_text = pooled_text.view(*reshape_to, self.hidden_dim)
            pooled_image = pooled_image.view(*reshape_to, self.hidden_dim)

        return pooled_text, pooled_image

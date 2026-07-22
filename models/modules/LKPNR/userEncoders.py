import torch.nn as nn
import torch.nn.functional as F

from config import Config
from models.modules.LKPNR.newsEncoders import NewsEncoder
from models.modules.layers import Attention, MultiHeadAttention


class UserEncoder(nn.Module):
    def __init__(self, news_encoder: NewsEncoder, config: Config):
        super().__init__()
        self.news_embedding_dim = news_encoder.news_embedding_dim
        self.news_encoder = news_encoder
        self.auxiliary_loss = None


class MHSA(UserEncoder):
    def __init__(self, news_encoder: NewsEncoder, config: Config):
        super().__init__(news_encoder, config)
        self.multiheadAttention = MultiHeadAttention(
            config.head_num,
            self.news_embedding_dim,
            config.max_history_num,
            config.max_history_num,
            config.head_dim,
            config.head_dim,
        )
        self.affine = nn.Linear(
            config.head_num * config.head_dim,
            self.news_embedding_dim,
            bias=True,
        )
        self.attention = Attention(self.news_embedding_dim, config.attention_dim)

    def initialize(self):
        self.multiheadAttention.initialize()
        nn.init.xavier_uniform_(self.affine.weight, gain=nn.init.calculate_gain("relu"))
        nn.init.zeros_(self.affine.bias)
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
        history_index=None,
    ):
        del (
            user_history_graph,
            user_history_category_mask,
            user_history_category_indices,
        )
        news_num = candidate_news_representation.size(1)
        history_embedding = self.news_encoder(
            user_title_text,
            user_title_mask,
            user_title_entity,
            user_content_text,
            user_content_mask,
            user_content_entity,
            user_category,
            user_subCategory,
            user_embedding,
            history_index,
        )

        history_mask = user_history_mask.to(history_embedding.dtype)
        history_embedding = history_embedding * history_mask.unsqueeze(dim=2)
        history_context = self.multiheadAttention(
            history_embedding,
            history_embedding,
            history_embedding,
            user_history_mask,
        )
        history_context = F.relu(
            F.dropout(self.affine(history_context), training=self.training)
        )
        history_context = history_context * history_mask.unsqueeze(dim=2)
        user_representation = self.attention(
            history_context, mask=user_history_mask
        )
        has_history = user_history_mask.any(dim=1, keepdim=True).to(
            user_representation.dtype
        )
        user_representation = user_representation * has_history
        return user_representation.unsqueeze(dim=1).expand(-1, news_num, -1)

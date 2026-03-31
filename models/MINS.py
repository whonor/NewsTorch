import torch
import torch.nn as nn
import torch.nn.functional as F

from models.modules.click_predictor import DotProduct
from models.modules.newsEncoders import MINS_NE
from models.modules.userEncoders import MINS_UE


class MINS(torch.nn.Module):
    def __init__(self, config):
        super(MINS, self).__init__()
        self.config = config
        self.news_encoder = MINS_NE(config)
        self.user_encoder = MINS_UE(self.news_encoder, config)
        self.click_predictor = DotProduct()
        assert int(config.word_embedding_dim % config.layers) == 0
        self.use_user_embedding = False
        self.user_embedding = nn.Embedding(
            config.user_num,
            int(config.word_embedding_dim / config.layers),
            padding_idx=0)
        self.dropout = nn.Dropout(p=config.dropout_rate)

    def initialize(self):
        self.news_encoder.initialize()
        if self.use_user_embedding:
            nn.init.uniform_(self.user_embedding.weight, -0.1, 0.1)
            nn.init.zeros_(self.user_embedding.weight[0])

    def forward(self, user_ID, user_category, user_subCategory, user_title_text, user_title_mask, user_title_entity, user_content_text, user_content_mask, user_content_entity, user_history_mask, user_history_graph, user_history_category_mask, user_history_category_indices, \
                      news_category, news_subCategory, news_title_text, news_title_mask, news_title_entity, news_content_text, news_content_mask, news_content_entity):
        user_embedding = self.dropout(self.user_embedding(user_ID)) if self.use_user_embedding else None
        # [batch, 5, 300]
        news_representation = self.news_encoder(news_title_text, news_title_mask, news_title_entity, news_content_text,
                                                news_content_mask, news_content_entity, news_category, news_subCategory,
                                                user_embedding)
        # [batch, 300]
        user_representation = self.user_encoder(user_title_text, user_title_mask, user_title_entity, user_content_text,
                                                user_content_mask, user_content_entity, user_category, user_subCategory, \
                                                user_history_mask, user_history_graph, user_history_category_mask,
                                                user_history_category_indices, user_embedding, news_representation)
        # [batch, 5]
        logits = self.click_predictor(
            news_representation,
            user_representation.unsqueeze(dim=-1)).squeeze(dim=-1)

        return logits
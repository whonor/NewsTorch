from config import Config
import torch.nn as nn
from models.modules import userEncoders, newsEncoders
from models.modules.click_predictor import DotProduct


class CenNewsRec(nn.Module):
    def __init__(self, config: Config):
        super(CenNewsRec, self).__init__()
        self.news_encoder = newsEncoders.CNNMHSAAddAtt(config)
        self.user_encoder = userEncoders.CenNewsRec(self.news_encoder, config)
        self.click_predictor = DotProduct()
        self.use_user_embedding = False
        self.model_name = config.model
        self.batch_size = config.batch_size
        self.config = config

    def initialize(self):
        if self.use_user_embedding:
            nn.init.uniform_(self.user_embedding.weight, -0.1, 0.1)
            nn.init.zeros_(self.user_embedding.weight[0])

    def forward(self, user_ID, user_category, user_subCategory, user_title_text, user_title_mask, user_title_entity, user_content_text, user_content_mask, user_content_entity, user_history_mask, user_history_graph, user_history_category_mask, user_history_category_indices, \
                      news_category, news_subCategory, news_title_text, news_title_mask, news_title_entity, news_content_text, news_content_mask, news_content_entity):
        user_embedding = self.dropout(self.user_embedding(user_ID)) if self.use_user_embedding else None
        # [batch, 400]
        news_representation = self.news_encoder(news_title_text, news_title_mask, news_title_entity, news_content_text, news_content_mask, news_content_entity, news_category, news_subCategory, user_embedding)
        # [batch, 400]
        user_representation = self.user_encoder(user_title_text, user_title_mask, user_title_entity, user_content_text, user_content_mask, user_content_entity, user_category, user_subCategory, \
                                                user_history_mask, user_history_graph, user_history_category_mask, user_history_category_indices, user_embedding, news_representation)
        # [batch, 5]
        logits = self.click_predictor(user_representation.unsqueeze(dim=1), news_representation.permute(0, 2, 1)).squeeze(dim=1)

        return logits


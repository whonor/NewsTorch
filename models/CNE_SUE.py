from config import Config
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.modules import userEncoders, newsEncoders


class CNE_SUE(nn.Module):
    def __init__(self, config: Config):
        super(CNE_SUE, self).__init__()
        self.config = config
        if config.model == 'CNE-SUE':
            self.news_encoder = newsEncoders.CNE(config)
            self.user_encoder = userEncoders.SUE(self.news_encoder, config)
        else:
            raise Exception(config.model + ' is not implemented')

        self.model_name = config.model
        self.news_embedding_dim = self.news_encoder.news_embedding_dim
        self.dropout = nn.Dropout(p=config.dropout_rate)
        self.use_user_embedding = False
        self.click_predictor = config.click_predictor
        if self.click_predictor == 'mlp':
            self.mlp = nn.Linear(in_features=self.news_embedding_dim * 2, out_features=self.news_embedding_dim // 2, bias=True)
            self.out = nn.Linear(in_features=self.news_embedding_dim // 2, out_features=1, bias=True)

    def initialize(self):
        self.news_encoder.initialize()
        self.user_encoder.initialize()
        if self.use_user_embedding:
            nn.init.uniform_(self.user_embedding.weight, -0.1, 0.1)
            nn.init.zeros_(self.user_embedding.weight[0])
        if self.click_predictor == 'mlp':
            nn.init.xavier_uniform_(self.mlp.weight, gain=nn.init.calculate_gain('relu'))
            nn.init.zeros_(self.mlp.bias)


    def forward(self, user_ID, user_category, user_subCategory, user_title_text, user_title_mask, user_title_entity, user_content_text, user_content_mask, user_content_entity, user_history_mask, user_history_graph, user_history_category_mask, user_history_category_indices, \
                      news_category, news_subCategory, news_title_text, news_title_mask, news_title_entity, news_content_text, news_content_mask, news_content_entity):
        user_embedding = self.dropout(self.user_embedding(user_ID)) if self.use_user_embedding else None
        # [batch, 5, 900]
        news_representation = self.news_encoder(news_title_text, news_title_mask, news_title_entity, news_content_text, news_content_mask, news_content_entity, news_category, news_subCategory, user_embedding)
        # [batch, 5, 900]
        user_representation = self.user_encoder(user_title_text, user_title_mask, user_title_entity, user_content_text, user_content_mask, user_content_entity, user_category, user_subCategory, \
                                                user_history_mask, user_history_graph, user_history_category_mask, user_history_category_indices, user_embedding, news_representation)
        if self.click_predictor == 'dot_product':
            logits = (user_representation * news_representation).sum(dim=2)
        elif self.click_predictor == 'mlp':
            context = self.dropout(F.relu(self.mlp(torch.cat([user_representation, news_representation], dim=2)), inplace=True))
            logits = self.out(context).squeeze(dim=2)

        return logits

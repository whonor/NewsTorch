from config import Config
import torch.nn as nn
from models.modules import userEncoders, newsEncoders
from models.modules.click_predictor import DotProduct


class LSTUR(nn.Module):
    def __init__(self, config: Config):
        super(LSTUR, self).__init__()
        self.news_encoder = newsEncoders.CNN(config)
        self.user_encoder = userEncoders.LSTUR(self.news_encoder, config)
        self.click_predictor = DotProduct()
        self.model_name = config.model
        self.batch_size = config.batch_size
        self.config = config
        self.news_embedding_dim = self.news_encoder.news_embedding_dim
        self.user_embedding = nn.Embedding(num_embeddings=config.user_num, embedding_dim=self.news_embedding_dim)
        self.use_user_embedding = True
        self.dropout = nn.Dropout(p=config.dropout_rate)

    def initialize(self):
        self.news_encoder.initialize()
        self.user_encoder.initialize()
        if self.use_user_embedding:
            nn.init.uniform_(self.user_embedding.weight, -0.1, 0.1)
            nn.init.zeros_(self.user_embedding.weight[0])

    def forward(self, user_ID, user_category, user_subCategory, user_title_text, user_title_mask, user_title_entity, user_content_text, user_content_mask, user_content_entity, user_history_mask, user_history_graph, user_history_category_mask, user_history_category_indices, \
                      news_category, news_subCategory, news_title_text, news_title_mask, news_title_entity, news_content_text, news_content_mask, news_content_entity):
        user_embedding = self.dropout(self.user_embedding(user_ID)) if self.use_user_embedding else None
        # [batch, 5, 500]
        news_representation = self.news_encoder(news_title_text, news_title_mask, news_title_entity, news_content_text, news_content_mask, news_content_entity, news_category, news_subCategory, user_embedding)
        # [batch, 500]
        user_representation = self.user_encoder(user_title_text, user_title_mask, user_title_entity, user_content_text, user_content_mask, user_content_entity, user_category, user_subCategory, \
                                                user_history_mask, user_history_graph, user_history_category_mask, user_history_category_indices, user_embedding, news_representation)
        # Ensure user_representation is 2D before unsqueeze
        if user_representation.dim() != 2:
            user_representation = user_representation.reshape(self.config.batch_size, -1)

        # Ensure news_representation is 3D before permute
        if news_representation.dim() != 3:
            news_representation = news_representation.view(self.config.batch_size, self.config.candidate_news_num, -1)
        # [batch, 5]
        logits = self.click_predictor(user_representation.unsqueeze(dim=1), news_representation.permute(0, 2, 1)).squeeze(1)
        # logits = (user_representation * news_representation).sum(dim=2)
        return logits


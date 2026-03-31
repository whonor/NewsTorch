from torch.nn import CrossEntropyLoss

from config import Config
import torch
import torch.nn as nn
from models.modules import userEncoders, newsEncoders
from models.modules.click_predictor import DotProduct


class TANR(nn.Module):
    def __init__(self, config: Config):
        super(TANR, self).__init__()
        self.news_encoder = newsEncoders.TANR(config)
        self.user_encoder = userEncoders.TANR(self.news_encoder, config)
        self.click_predictor = DotProduct()
        self.topic_predictor = nn.Linear(
            in_features=config.cnn_kernel_num, out_features=config.category_num + 1
        )
        self.use_user_embedding = False
        self.model_name = config.model
        self.batch_size = config.batch_size
        self.config = config

    def initialize(self):
        self.news_encoder.initialize()
        self.user_encoder.initialize()
        nn.init.xavier_uniform_(self.topic_predictor.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.zeros_(self.topic_predictor.bias)
        if self.use_user_embedding:
            nn.init.uniform_(self.user_embedding.weight, -0.1, 0.1)
            nn.init.zeros_(self.user_embedding.weight[0])
        if self.click_predictor == 'mlp':
            nn.init.xavier_uniform_(self.mlp.weight, gain=nn.init.calculate_gain('relu'))
            nn.init.zeros_(self.mlp.bias)
        elif self.click_predictor == 'FIM':
            nn.init.xavier_uniform_(self.fc.weight)
            nn.init.zeros_(self.fc.bias)

    def forward(self, user_ID, user_category, user_subCategory, user_title_text, user_title_mask, user_title_entity, user_content_text, user_content_mask, user_content_entity, user_history_mask, user_history_graph, user_history_category_mask, user_history_category_indices, \
                      news_category, news_subCategory, news_title_text, news_title_mask, news_title_entity, news_content_text, news_content_mask, news_content_entity):
        user_embedding = self.dropout(self.user_embedding(user_ID)) if self.use_user_embedding else None
        # [batch, 5, 400]
        news_representation = self.news_encoder(news_title_text, news_title_mask, news_title_entity, news_content_text, news_content_mask, news_content_entity, news_category, news_subCategory, user_embedding)
        # [batch, 400]
        user_representation = self.user_encoder(user_title_text, user_title_mask, user_title_entity, user_content_text, user_content_mask, user_content_entity, user_category, user_subCategory, \
                                                user_history_mask, user_history_graph, user_history_category_mask, user_history_category_indices, user_embedding, news_representation)
        # [batch, 5]
        logits = self.click_predictor(user_representation.unsqueeze(dim=1), news_representation.permute(0, 2, 1)).squeeze(dim=1)

        # [batch, 1 + negative_sample_num + max_history_num, 400]
        mixed_vector = torch.cat((news_representation, user_representation.unsqueeze(dim=1).expand(-1, self.config.max_history_num, -1)), dim=1)
        # [batch * (1 + negative_sample_num + max_history_num), category_num + 1]
        topic_scores = self.topic_predictor(mixed_vector).cuda()
        # [batch * (1 + negative_sample_num + max_history_num)]
        topic_mixed_vector = torch.cat((news_category, user_category), dim=1).cuda()
        class_weight = torch.ones(self.config.category_num + 1).cuda()
        class_weight[0] = 0
        criterion = CrossEntropyLoss(weight=class_weight)
        topic_pred_loss = criterion(topic_scores.view(-1, self.config.category_num + 1), topic_mixed_vector.flatten().long())


        return logits, topic_pred_loss


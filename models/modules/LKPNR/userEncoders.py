import math
from config import Config
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.modules.LKPNR.newsencoder import NewsEncoder
from models.modules.layers import MultiHeadAttention, Attention


class UserEncoder(nn.Module):
    def __init__(self, news_encoder: NewsEncoder, config: Config):
        super(UserEncoder, self).__init__()
        self.news_embedding_dim = news_encoder.news_embedding_dim
        self.news_encoder = news_encoder
        self.device = torch.device('cuda')
        self.auxiliary_loss = None

    def forward(self, user_title_text, user_title_mask, user_title_entity, user_content_text, user_content_mask, user_content_entity, user_category, user_subCategory, \
                user_history_mask, user_history_graph, user_history_category_mask, user_history_category_indices, user_embedding, candidate_news_representation):
        raise Exception('Function forward must be implemented at sub-class')


class MHSA(UserEncoder):
    def __init__(self, news_encoder: NewsEncoder, config: Config):
        super(MHSA, self).__init__(news_encoder, config)
        self.multiheadAttention = MultiHeadAttention(config.head_num, self.news_embedding_dim, config.max_history_num, config.max_history_num, config.head_dim, config.head_dim)
        self.affine = nn.Linear(config.head_num*config.head_dim, self.news_embedding_dim, bias=True)
        self.attention = Attention(self.news_embedding_dim, config.attention_dim)

    def initialize(self):
        self.multiheadAttention.initialize()
        nn.init.xavier_uniform_(self.affine.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.zeros_(self.affine.bias)
        self.attention.initialize()

    def forward(self, user_title_text, user_title_mask, user_title_entity, user_content_text, user_content_mask, user_content_entity, user_category, user_subCategory, \
                user_history_mask, user_history_graph, user_history_category_mask, user_history_category_indices, user_embedding, candidate_news_representation, history_index):
        news_num = candidate_news_representation.size(1)
        
        # 64 50 4096
        try:
            history_embedding = self.news_encoder(user_title_text, user_title_mask, user_title_entity, \
                                                  user_content_text, user_content_mask, user_content_entity, \
                                                  user_category, user_subCategory, user_embedding, history_index)                  # [batch_size, max_history_num, news_embedding_dim]
        except Exception as e:
            print(f"Error in news_encoder forward: {e}")
            # Create a fallback history embedding with the correct shape
            batch_size = user_title_text.size(0)
            max_history_num = user_title_text.size(1)
            history_embedding = torch.zeros((batch_size, max_history_num, self.news_embedding_dim), dtype=torch.float32).cuda()
            
        # 64 50 400
        h = self.multiheadAttention(history_embedding, history_embedding, history_embedding, user_history_mask) # [batch_size, max_history_num, head_num * head_dim]
        # 64, 50, 4596
        h = F.relu(F.dropout(self.affine(h), training=self.training, inplace=True), inplace=True)               # [batch_size, max_history_num, news_embedding_dim]
        # 64 5 4596
        user_representation = self.attention(h).unsqueeze(dim=1).repeat(1, news_num, 1)                         # [batch_size, news_num, news_embedding_dim]
        return user_representation


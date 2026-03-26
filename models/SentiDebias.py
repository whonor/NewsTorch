from config import Config
import torch
import torch.nn as nn
from models.modules.newsEncoders import MHSA as NewsEncoder
from models.modules.click_predictor import DotProduct
import torch.nn.functional as F
from models.modules.layers import MultiHeadAttention, Attention


class Discriminator(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, hist_news_vector: torch.Tensor, cand_news_vector: torch.Tensor):
        pred_hist_sent = self.linear2(torch.tanh(self.linear1(hist_news_vector)))
        pred_cand_sent = self.linear2(torch.tanh(self.linear1(cand_news_vector)))
        return pred_hist_sent, pred_cand_sent


class SentiDebiasUserEncoder(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.news_encoder = NewsEncoder(config)
        self.multiheadAttention = MultiHeadAttention(config.head_num, self.news_encoder.news_embedding_dim,
                                                     config.max_history_num, config.max_history_num,
                                                     self.news_encoder.news_embedding_dim // config.head_num,
                                                     self.news_encoder.news_embedding_dim // config.head_num)
        self.affine = nn.Linear(self.news_encoder.news_embedding_dim, self.news_encoder.news_embedding_dim, bias=True)
        self.attention = Attention(self.news_encoder.news_embedding_dim, self.news_encoder.news_embedding_dim)
        self.config = config

    def initialize(self):
        self.multiheadAttention.initialize()
        nn.init.xavier_uniform_(self.affine.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.zeros_(self.affine.bias)
        self.attention.initialize()

    def forward(self, history_embedding, user_history_mask):
        # [batch_size, max_history_num, head_num * head_dim]
        h = self.multiheadAttention(history_embedding, history_embedding, history_embedding, user_history_mask)
        # [batch_size, max_history_num, news_embedding_dim]
        h = F.relu(F.dropout(self.affine(h), training=self.training, inplace=True), inplace=True)
        # [batch_size, news_embedding_dim]
        user_representation = self.attention(h)
        return user_representation


class Generator(nn.Module):
    def __init__(self, config: Config):
        super(Generator, self).__init__()
        self.news_encoder = NewsEncoder(config)
        self.user_encoder = SentiDebiasUserEncoder(config)
        self.sentiment_encoder = nn.Embedding(config.num_sent_classes, self.news_encoder.news_embedding_dim)
        self.click_predictor_bias_free = DotProduct()
        self.click_predictor_bias_aware = DotProduct()
        self.config = config

    def forward(self, user_ID, user_category, user_subCategory, user_title_text, user_title_mask, user_title_entity, user_content_text, user_content_mask, user_content_entity, user_history_mask, user_history_graph, user_history_category_mask, user_history_category_indices, \
                news_category, news_subCategory, news_title_text, news_title_mask, news_title_entity, news_content_text, news_content_mask, news_content_entity, \
                user_hist_sentiment, news_sentiment):
        
        # Encode history news
        hist_news_vector = self.news_encoder(user_title_text, user_title_mask, user_title_entity, user_content_text, user_content_mask, user_content_entity, user_category, user_subCategory, None)

        # Encode candidate news
        cand_news_vector = self.news_encoder(news_title_text, news_title_mask, news_title_entity, news_content_text, news_content_mask, news_content_entity, news_category, news_subCategory, None)

        # Project sentiment scores to vectors
        hist_sentiment_vector = self.sentiment_encoder(user_hist_sentiment)
        cand_sentiment_vector = self.sentiment_encoder(news_sentiment)

        # User representations
        user_representation_bias_free = self.user_encoder(hist_news_vector, user_history_mask)
        user_representation_bias_aware = self.user_encoder(hist_sentiment_vector, user_history_mask)

        # Orthogonality loss
        loss_orth_hist_news = torch.mean(
            (torch.sum(hist_news_vector * hist_sentiment_vector, dim=-1) /
             (1e-8 + torch.linalg.norm(hist_news_vector, dim=-1, ord=2) * torch.linalg.norm(hist_sentiment_vector, dim=-1, ord=2))),
            dim=-1
        )
        loss_orth_cand_news = torch.mean(
            (torch.sum(cand_news_vector * cand_sentiment_vector, dim=-1) /
             (1e-8 + torch.linalg.norm(cand_news_vector, dim=-1, ord=2) * torch.linalg.norm(cand_sentiment_vector, dim=-1, ord=2))),
            dim=-1
        )
        loss_orth_user = torch.div(
            torch.bmm(
                user_representation_bias_free.unsqueeze(dim=1), user_representation_bias_aware.unsqueeze(dim=-1)
            ).squeeze(dim=1),
            (
                1e-8
                + torch.linalg.norm(user_representation_bias_free, dim=1, ord=2)
                * (torch.linalg.norm(user_representation_bias_aware, dim=1, ord=2))
            ).unsqueeze(dim=1),
        )
        
        loss_orth = (
            torch.abs(loss_orth_hist_news).mean()
            + torch.abs(loss_orth_cand_news).mean()
            + torch.abs(loss_orth_user).mean()
        )

        # Click scores
        if cand_news_vector.dim() == 2:
            cand_news_vector = cand_news_vector.unsqueeze(1)
        if cand_sentiment_vector.dim() == 2:
            cand_sentiment_vector = cand_sentiment_vector.unsqueeze(1)
        bias_free_scores = self.click_predictor_bias_free(user_representation_bias_free.unsqueeze(dim=1), cand_news_vector.permute(0, 2, 1)).squeeze(dim=1)
        bias_aware_scores = self.click_predictor_bias_aware(user_representation_bias_aware.unsqueeze(dim=1), cand_sentiment_vector.permute(0, 2, 1)).squeeze(dim=1)
        
        combined_scores = bias_free_scores + bias_aware_scores

        return combined_scores, bias_free_scores, loss_orth, hist_news_vector, cand_news_vector


class SentiDebias(nn.Module):
    def __init__(self, config: Config):
        super(SentiDebias, self).__init__()
        self.generator = Generator(config)
        self.discriminator = Discriminator(self.generator.news_encoder.news_embedding_dim, config.hidden_dim, config.num_sent_classes)
        self.config = config
        self.model_name = 'SentiDebias'

    def initialize(self):
        self.generator.news_encoder.initialize()
        self.generator.user_encoder.initialize()
        nn.init.xavier_uniform_(self.generator.sentiment_encoder.weight)

    def forward(self, user_ID, user_category, user_subCategory, user_title_text, user_title_mask, user_title_entity, user_content_text, user_content_mask, user_content_entity, user_history_mask, user_history_graph, user_history_category_mask, user_history_category_indices, \
                news_category, news_subCategory, news_title_text, news_title_mask, news_title_entity, news_content_text, news_content_mask, news_content_entity, \
                user_hist_sentiment, news_sentiment):
        
        return self.generator(user_ID, user_category, user_subCategory, user_title_text, user_title_mask, user_title_entity, user_content_text, user_content_mask, user_content_entity, user_history_mask, user_history_graph, user_history_category_mask, user_history_category_indices, \
                news_category, news_subCategory, news_title_text, news_title_mask, news_title_entity, news_content_text, news_content_mask, news_content_entity, \
                user_hist_sentiment, news_sentiment)


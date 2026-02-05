import torch
import torch.nn as nn
import torch.nn.functional as F
from config import Config
from models.modules import newsEncoders, userEncoders
from models.modules.click_predictor import DotProduct


class SentiRecUserEncoder(userEncoders.MHSA):
    """
    A specific user encoder for SentiRec.
    It inherits from the NRMS user encoder (MHSA) but modifies the forward pass
    to return not only the user representation but also the history news representations.
    This is necessary for calculating the sentiment prediction loss (L_senti) on history items
    without re-computing the news encodings.
    """

    def __init__(self, news_encoder, config):
        super(SentiRecUserEncoder, self).__init__(news_encoder, config)

    def forward(self, user_title_text, user_title_mask, user_title_entity, user_content_text, user_content_mask,
                user_content_entity, user_category, user_subCategory,
                user_history_mask, user_history_graph, user_history_category_mask, user_history_category_indices,
                user_embedding, candidate_news_representation):
        # [batch_size, max_history_num, news_embedding_dim]
        history_embedding = self.news_encoder(user_title_text, user_title_mask, user_title_entity, \
                                              user_content_text, user_content_mask, user_content_entity, \
                                              user_category, user_subCategory, user_embedding)

        # The rest of the logic is identical to the base userEncoders.MHSA
        # [batch_size, max_history_num, head_num * head_dim]
        h = self.multiheadAttention(history_embedding, history_embedding, history_embedding, user_history_mask)
        # [batch_size, max_history_num, news_embedding_dim]
        h = F.relu(F.dropout(self.affine(h), p=self.training, inplace=True), inplace=True)
        # [batch_size, news_embedding_dim]
        user_representation = self.attention(h)

        # Return both the final user representation and the history news representations
        return user_representation, history_embedding


class SentiRec(nn.Module):
    def __init__(self, config: Config):
        super(SentiRec, self).__init__()
        self.config = config
        self.model_name = config.model

        # Use the standard news encoder
        self.news_encoder = newsEncoders.MHSA(config)
        # Use our SentiRec-specific user encoder
        self.user_encoder = SentiRecUserEncoder(self.news_encoder, config)
        self.click_predictor = DotProduct()

        # SentiRec specific component: Sentiment Predictor
        # The input dimension must match the output dimension of the news encoder
        news_embedding_dim = self.news_encoder.news_embedding_dim
        self.sentiment_predictor = nn.Linear(news_embedding_dim, 1)
        
        # Loss function for sentiment prediction (L_senti)
        self.sentiment_loss_fn = nn.L1Loss()

    def initialize(self):
        self.news_encoder.initialize()
        self.user_encoder.initialize()
        nn.init.xavier_uniform_(self.sentiment_predictor.weight)
        if self.sentiment_predictor.bias is not None:
            nn.init.zeros_(self.sentiment_predictor.bias)

    def forward(self, user_ID, user_category, user_subCategory, user_title_text, user_title_mask, user_title_entity, user_content_text, user_content_mask, user_content_entity, user_history_mask, user_history_graph, user_history_category_mask, user_history_category_indices,
                      news_category, news_subCategory, news_title_text, news_title_mask, news_title_entity, news_content_text, news_content_mask, news_content_entity,
                      history_index=None, sample_index=None, history_sentiment=None, candidate_sentiment=None):
        
        # === 1. Standard Recommendation Path (always runs) ===
        
        # Encode candidate news: [B, N_c, D]
        cand_news_repr = self.news_encoder(news_title_text, news_title_mask, news_title_entity, news_content_text, news_content_mask, news_content_entity, news_category, news_subCategory, None)

        # Encode user based on history. Our custom user encoder returns both user and history representations.
        # user_repr: [B, D], hist_news_repr: [B, N_h, D]
        user_repr, hist_news_repr = self.user_encoder(user_title_text, user_title_mask, user_title_entity, user_content_text, user_content_mask, user_content_entity, user_category, user_subCategory,
                                                    user_history_mask, user_history_graph, user_history_category_mask, user_history_category_indices, None, cand_news_repr)

        # Predict click scores
        click_logits = self.click_predictor(user_repr.unsqueeze(dim=1), cand_news_repr.permute(0, 2, 1)).squeeze(dim=1)

        # === 2. Sentiment-Aware Path (only runs during training) ===
        
        # If sentiment scores are not provided (i.e., during evaluation), just return click logits.
        if history_sentiment is None or candidate_sentiment is None:
            return click_logits

        # If we are training, compute and return the additional losses.
        batch_size, hist_num, _ = hist_news_repr.shape
        cand_num = cand_news_repr.shape[1]
        
        # Flatten representations for sentiment predictor
        hist_news_repr_flat = hist_news_repr.view(batch_size * hist_num, -1)
        cand_news_repr_flat = cand_news_repr.view(batch_size * cand_num, -1)
        all_news_repr = torch.cat([hist_news_repr_flat, cand_news_repr_flat], dim=0)

        # Get sentiment predictions
        sentiment_preds = self.sentiment_predictor(all_news_repr).squeeze(-1)
        
        # Ground truth sentiments, flattened
        hist_sent_true = history_sentiment.view(-1)
        cand_sent_true = candidate_sentiment.view(-1)
        all_sent_true = torch.cat([hist_sent_true, cand_sent_true], dim=0)

        # --- Calculate L_senti ---
        l_senti = self.sentiment_loss_fn(sentiment_preds, all_sent_true)

        # --- Calculate L_div ---
        s_u = history_sentiment.sum(dim=1, keepdim=True) / torch.clamp(user_history_mask.sum(dim=1, keepdim=True), min=1)
        s_u = s_u.expand(-1, cand_num)

        s_c = candidate_sentiment
        y_hat = torch.sigmoid(click_logits)
        p = F.relu(s_u * s_c * y_hat)
        l_div = p.mean()

        return click_logits, l_senti, l_div

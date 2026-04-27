import torch
import torch.nn as nn
from config import Config
from models.modules import newsEncoders
from models.modules.click_predictor import DotProduct
from models.modules.layers import Attention

class CPRS(nn.Module):
    def __init__(self, config: Config):
        super(CPRS, self).__init__()
        self.config = config
        
        # TextEncoder using MHSA
        self.news_encoder = newsEncoders.MHSA(config)
        
        # Embedding dimension of news representations
        self.news_dim = config.head_num * config.head_dim + config.category_embedding_dim + config.subCategory_embedding_dim
        
        # Attention Networks
        self.content_sat_attention = Attention(self.news_dim, config.attention_dim)
        self.title_attention = Attention(self.news_dim, config.attention_dim)
        
        # Behavior Attention Network
        self.behavior_proj = nn.Linear(self.news_dim * 2, self.news_dim)
        
        # Predictors
        self.click_predictor = DotProduct()
        self.satisfaction_predictor = nn.Linear(self.news_dim * 2, 1)

    def initialize(self):
        self.news_encoder.initialize()
        self.content_sat_attention.initialize()
        self.title_attention.initialize()
        nn.init.xavier_uniform_(self.behavior_proj.weight)
        nn.init.zeros_(self.behavior_proj.bias)
        nn.init.xavier_uniform_(self.satisfaction_predictor.weight)
        nn.init.zeros_(self.satisfaction_predictor.bias)

    def forward(self, user_ID, user_category, user_subCategory, user_title_text, user_title_mask, user_title_entity, user_content_text, user_content_mask, user_content_entity, user_history_mask, user_history_graph, user_history_category_mask, user_history_category_indices, \
                      news_category, news_subCategory, news_title_text, news_title_mask, news_title_entity, news_content_text, news_content_mask, news_content_entity, *extra_args):
        
        # extra_args contains items from index 21 onwards of the data_batch in EBNeRD_Corpus
        # index 21: news_sentiment_history (extra_args[0])
        # index 22: news_sentiment_sample (extra_args[1])
        # index 23: history_index (extra_args[2])
        # index 24: sample_index (extra_args[3])
        # index 25: user_image_embeddings (extra_args[4])
        # index 26: news_image_embeddings (extra_args[5])
        # index 27: user_history_read_time (extra_args[6])
        # index 28: next_read_time (extra_args[7])
        next_read_time = extra_args[7] if len(extra_args) > 7 else None

        # [batch_size, max_history, news_dim]
        # news_encoder for user history
        history_repr = self.news_encoder(user_title_text, user_title_mask, user_title_entity, user_content_text, user_content_mask, user_content_entity, user_category, user_subCategory, None)
        
        # Content and Title attention
        u_c = self.content_sat_attention(history_repr, mask=user_history_mask)
        u_t = self.title_attention(history_repr, mask=user_history_mask)
        
        # User Representation
        u = self.behavior_proj(torch.cat([u_c, u_t], dim=-1))
        
        # Candidate News Representation
        candidate_repr = self.news_encoder(news_title_text, news_title_mask, news_title_entity, news_content_text, news_content_mask, news_content_entity, news_category, news_subCategory, None)
        if candidate_repr.dim() == 2:
            candidate_repr = candidate_repr.unsqueeze(1)
            
        # Click Prediction
        # [batch_size, 1 + negative_sample_num]
        logits = self.click_predictor(u.unsqueeze(1), candidate_repr.permute(0, 2, 1)).squeeze(1)
        
        # Satisfaction Prediction (\hat{s} = w_s^T [u; d^c])
        u_expanded = u.unsqueeze(1).expand(-1, candidate_repr.size(1), -1)
        sat_preds = self.satisfaction_predictor(torch.cat([u_expanded, candidate_repr], dim=-1)).squeeze(-1)
        
        # We only compute gold satisfaction for the positive candidate during training
        if next_read_time is not None and len(next_read_time.shape) > 0 and self.training:
            # Positive candidate is at index 0
            # Calculate cand_len
            cand_len = news_title_mask[:, 0].sum(dim=-1) + news_content_mask[:, 0].sum(dim=-1)
            
            # Handle NaN in next_read_time
            valid_mask = ~torch.isnan(next_read_time)
            t_i = torch.where(valid_mask, next_read_time, torch.ones_like(next_read_time))
            t_i = torch.clamp(t_i, min=1.0)
            
            # Ensure t_i is broadcastable with cand_len
            if t_i.dim() > cand_len.dim():
                t_i = t_i.view(cand_len.shape)
                valid_mask = valid_mask.view(cand_len.shape)
                
            v_i = cand_len / t_i
            
            # Compute v_mean using only valid samples to avoid NaN
            if valid_mask.any():
                v_mean = torch.mean(v_i[valid_mask])
                v_mean = torch.clamp(v_mean, min=1e-5)
                s_i = torch.log2((v_i / v_mean) + 1e-5)
                # Set s_i to 0 for invalid samples and also return the mask if needed
                # But the trainer expects (logits, sat_preds, s_i)
                # We can set s_i to a value that will result in 0 loss when masked or just handle it here
                s_i = torch.where(valid_mask, s_i, torch.zeros_like(s_i))
                
                # To communicate to the trainer which samples are valid, 
                # we could return valid_mask too, but the trainer doesn't expect it.
                # Alternatively, we can make sat_preds equal to s_i for invalid samples
                # so that torch.abs(s_i - sat_preds).mean() doesn't get messed up (though mean still includes them)
                return logits, sat_preds[:, 0], s_i, valid_mask
            else:
                # If no valid samples in batch, return zeros for s_i and a false mask
                return logits, sat_preds[:, 0], torch.zeros_like(v_i), valid_mask

        # Return empty tensors instead of None for DataParallel compatibility
        return logits, sat_preds, torch.tensor([], device=logits.device), torch.tensor([], device=logits.device, dtype=torch.bool)

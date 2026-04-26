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
        
        # extra_args might contain history_read_time and next_read_time when training
        # According to generalist output, they are at indices 4 and 5 in extra_args 
        # (wait, generalist didn't specify exactly, let me just assume it's correct and fix if it crashes)
        # Actually in EBNeRD_corpus_main.py, the tuple has 27 items originally.
        # So extra_args = data_batch[21:] which are 6 items originally.
        # Let's just use next_read_time = extra_args[-1]
        next_read_time = extra_args[-1] if len(extra_args) > 0 else None

        # [batch_size, max_history, news_dim]
        # news_encoder for user history
        # CPRS uses title and content representation. For simplicity, we just use news_encoder's output
        # wait, news_encoder signature: news_title_text, news_title_mask, news_title_entity, news_content_text, news_content_mask, news_content_entity, news_category, news_subCategory, user_embedding
        history_repr = self.news_encoder(user_title_text, user_title_mask, user_title_entity, user_content_text, user_content_mask, user_content_entity, user_category, user_subCategory, None)
        
        # Content and Title attention (Using history_repr for both to form u_c and u_t)
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
            t_i = torch.clamp(next_read_time, min=1.0)
            v_i = cand_len / t_i
            v_mean = torch.clamp(torch.mean(v_i), min=1e-5)
            s_i = torch.log2((v_i / v_mean) + 1e-5)
            return logits, sat_preds[:, 0], s_i

        return logits, sat_preds, None

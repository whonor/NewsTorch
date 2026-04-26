from config import Config
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.modules import userEncoders, newsEncoders
from models.modules.click_predictor import DotProduct
import math

class AttentionPooling(nn.Module):
    def __init__(self, d_h, hidden_size=200):
        super(AttentionPooling, self).__init__()
        self.w_1 = nn.Linear(d_h, hidden_size)
        self.w_2 = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, x, mask=None):
        # x: [batch_size, seq_len, d_h]
        att_weights = self.w_2(torch.tanh(self.w_1(x))).squeeze(-1) # [batch_size, seq_len]
        if mask is not None:
            att_weights = att_weights.masked_fill(mask == 0, -1e9)
        att_weights = F.softmax(att_weights, dim=-1)
        # [batch_size, d_h]
        return torch.bmm(att_weights.unsqueeze(1), x).squeeze(1)

class TCCM(nn.Module):
    def __init__(self, config: Config):
        super(TCCM, self).__init__()
        self.model_name = "TCCM"
        self.config = config
        self.news_encoder = newsEncoders.MHSA(config)
        self.user_encoder = userEncoders.MHSA(self.news_encoder, config)
        
        # Pop embeddings
        self.pop_emb_dim = 200
        self.pop_embedding = nn.Embedding(201, self.pop_emb_dim)
        
        # Self and Cross Attention for Pop
        self.num_heads = 20
        self.ze_mhsa = nn.MultiheadAttention(self.pop_emb_dim, self.num_heads, batch_first=True)
        self.ze_mhca = nn.MultiheadAttention(self.pop_emb_dim, self.num_heads, batch_first=True)
        self.pe_att = AttentionPooling(self.pop_emb_dim, 20)
        
        self.zw_mhsa = nn.MultiheadAttention(self.pop_emb_dim, self.num_heads, batch_first=True)
        self.zw_mhca = nn.MultiheadAttention(self.pop_emb_dim, self.num_heads, batch_first=True)
        self.pw_att = AttentionPooling(self.pop_emb_dim, 20)
        
        self.unified_pop_att = AttentionPooling(self.pop_emb_dim, 20)
        self.pop_dense = nn.Sequential(
            nn.Linear(self.pop_emb_dim, self.pop_emb_dim // 2),
            nn.ReLU(),
            nn.Linear(self.pop_emb_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Time module
        self.time_proj = nn.Linear(1, self.pop_emb_dim)
        self.time_dense = nn.Sequential(
            nn.Linear(self.pop_emb_dim, self.pop_emb_dim // 2),
            nn.ReLU(),
            nn.Linear(self.pop_emb_dim // 2, 1),
            nn.Sigmoid()
        )
        self.lmbda = nn.Parameter(torch.tensor(1.0))
        self.alpha = nn.Parameter(torch.tensor(0.5))
        
    def initialize(self):
        self.news_encoder.initialize()
        self.user_encoder.initialize()

    def get_pop_score(self, word_pop, entity_pop):
        # word_pop, entity_pop: [batch, seq_len]
        # Embeddings: [batch, seq_len, dim]
        ZW = self.pop_embedding(word_pop)
        ZE = self.pop_embedding(entity_pop)
        
        # mask
        mask_w = (word_pop == 0)
        mask_e = (entity_pop == 0)
        
        all_masked_w = mask_w.all(dim=1, keepdim=True)
        mask_w = mask_w.masked_fill(all_masked_w, False)
        
        all_masked_e = mask_e.all(dim=1, keepdim=True)
        mask_e = mask_e.masked_fill(all_masked_e, False)

        # MHSA and MHCA for Entity
        z1_e, _ = self.ze_mhsa(ZE, ZE, ZE, key_padding_mask=mask_e)
        z2_e, _ = self.ze_mhca(ZE, ZW, ZW, key_padding_mask=mask_w)
        pe = self.pe_att(z1_e + z2_e, mask=~mask_e) # [batch, dim]
        
        # MHSA and MHCA for Word
        z1_w, _ = self.zw_mhsa(ZW, ZW, ZW, key_padding_mask=mask_w)
        z2_w, _ = self.zw_mhca(ZW, ZE, ZE, key_padding_mask=mask_e)
        pw = self.pw_att(z1_w + z2_w, mask=~mask_w) # [batch, dim]
        
        # Unified popularity
        unified = torch.stack([pe, pw], dim=1) # [batch, 2, dim]
        unified_rep = self.unified_pop_att(unified) # [batch, dim]
        sp = self.pop_dense(unified_rep).squeeze(1) # [batch]
        return sp
        
    def forward(self, user_ID, user_category, user_subCategory, user_title_text, user_title_mask, user_title_entity, user_content_text, user_content_mask, user_content_entity, user_history_mask, user_history_graph, user_history_category_mask, user_history_category_indices, \
                      news_category, news_subCategory, news_title_text, news_title_mask, news_title_entity, news_content_text, news_content_mask, news_content_entity, news_sentiment_history, news_sentiment_sample, history_index, sample_index, user_image_embeddings, news_image_embeddings, user_history_read_time, next_read_time, history_word_pop, history_entity_pop, sample_word_pop, sample_entity_pop):
        
        # Match module
        is_eval = (sample_word_pop.dim() == 2)
        if is_eval:
            sample_word_pop = sample_word_pop.unsqueeze(1)
            sample_entity_pop = sample_entity_pop.unsqueeze(1)
            
        # news_representation: [batch, 1+neg, 400] or [batch, 400] (eval)
        news_representation = self.news_encoder(news_title_text, news_title_mask, news_title_entity, news_content_text, news_content_mask, news_content_entity, news_category, news_subCategory, None)
        
        if news_representation.dim() == 2:
            news_representation = news_representation.unsqueeze(1)
            
        # user_representation: [batch, 400]
        user_representation = self.user_encoder(user_title_text, user_title_mask, user_title_entity, user_content_text, user_content_mask, user_content_entity, user_category, user_subCategory, \
                                                user_history_mask, user_history_graph, user_history_category_mask, user_history_category_indices, None, news_representation)
        
        sm = (user_representation.unsqueeze(1) * news_representation).sum(dim=2) # [batch, 1+neg]
        
        # Popularity score for candidate news: sample_word_pop is [batch, 1+neg, seq_len]
        batch_size = sample_word_pop.size(0)
        num_cands = sample_word_pop.size(1)
        seq_len = sample_word_pop.size(2)
        
        word_pop_flat = sample_word_pop.view(batch_size * num_cands, seq_len)
        entity_pop_flat = sample_entity_pop.view(batch_size * num_cands, seq_len)
        sp_flat = self.get_pop_score(word_pop_flat, entity_pop_flat)
        sp = sp_flat.view(batch_size, num_cands) # [batch, 1+neg]
        
        # Time score
        next_read_time = torch.nan_to_num(next_read_time, nan=0.0)
        tn = next_read_time.unsqueeze(1).expand(-1, num_cands).unsqueeze(2) # [batch, 1+neg, 1]
        time_emb = self.time_proj(tn)
        fn_prime = self.time_dense(time_emb).squeeze(2) # [batch, 1+neg]
        # Avoid division by zero
        fn_prime = torch.clamp(fn_prime, min=1e-5)
        
        # Safe lambda and alpha
        lmbda_val = torch.nn.functional.softplus(self.lmbda)
        alpha_val = torch.sigmoid(self.alpha)
        
        st = torch.pow(1.0 / fn_prime, lmbda_val) # [batch, 1+neg]
        st = torch.clamp(st, max=20.0) # Prevent exploding st
        
        # Fusion
        if not self.training:
            sp = torch.full_like(sp, 0.05)  # causal intervention: set popularity to a low level
            
        s = (1 - alpha_val) * sm + alpha_val * (sp * st)
        
        if is_eval:
            s = s.squeeze(1)
            
        return s

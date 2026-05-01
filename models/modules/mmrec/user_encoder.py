import torch
from torch import nn


class CandidateAttention(nn.Module):
    def __init__(self):
        super(CandidateAttention, self).__init__()

    def forward(self, query, key, attn_mask=None):
        bz = key.shape[0]
        score = torch.bmm(key, query.unsqueeze(2)).squeeze(2)
        if attn_mask is not None:
            score = score.masked_fill(attn_mask <= 0, -1e12)
        alpha = torch.nn.functional.softmax(score, -1)
        if attn_mask is not None:
            alpha = alpha * attn_mask
            alpha = alpha / alpha.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        x = torch.bmm(key.permute(0, 2, 1), alpha.unsqueeze(2))
        x = torch.reshape(x, (bz, -1))
        return x


class UserEncoder(nn.Module):
    def __init__(self):
        super(UserEncoder, self).__init__()
        self.t2t = CandidateAttention()
        self.t2i = CandidateAttention()
        self.i2i = CandidateAttention()
        self.i2t = CandidateAttention()

    def forward(self, candidate_t, candidate_i, his_t, his_i, log_mask):
        bz, candidate_len, _ = candidate_t.shape
        log_mask = log_mask.to(dtype=his_t.dtype, device=his_t.device) if log_mask is not None else None
        candidate_user = []
        for idx in range(candidate_len):
            candidate_news_t = candidate_t[:, idx, :]
            candidate_news_i = candidate_i[:, idx, :]
            candidate_user.append(
                self.t2i(candidate_news_t, his_i, log_mask) +
                self.t2t(candidate_news_t, his_t, log_mask) +
                self.i2t(candidate_news_i, his_t, log_mask) +
                self.i2i(candidate_news_i, his_i, log_mask)
            )
        # [batch_size,candi_len,user_dim]
        return torch.stack(candidate_user, 1)

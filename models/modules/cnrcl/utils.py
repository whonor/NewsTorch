import torch
import torch.nn as nn

class NewsRecSupConLoss_with_neg_pos(nn.Module):
    def __init__(self, temperature=0.07, base_temperature=0.07):
        super(NewsRecSupConLoss_with_neg_pos, self).__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature

    def create_mask(self, batch_size, negative_sample_num, positive_sample_num):
        mask = torch.zeros((batch_size, negative_sample_num + positive_sample_num))
        mask[:, negative_sample_num:] = 1
        return mask

    def forward(self, features, negative_sample_num):
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) != 3:
            raise ValueError('`features` needs to be [batch_size, 1 + negative_sample_num + positive_sample_num, news_embedding_dim],'
                             'exactly 3 dimensions are required')

        batch_size = features.shape[0]
        # features shape: [batch, 1 + neg + pos?, dim] ??
        # In trainer.py call: sloss = supcon_loss_nlp(logits_unsum, 4)
        # logits_unsum shape is [batch, 1 + neg, dim] usually.
        # Wait, CNRCL trainer uses `negative_sample_num=4`.
        # features[:,0] is anchor/positive click?
        # features[:,1:,:] is contrast?
        
        # Let's trace `trainer.py` in CNRCL.
        # `logits_unsum` comes from `model()`.
        # `model()` returns `user_rep * news_rep`.
        # `news_rep` has 1 positive + N negatives.
        # So `logits_unsum` has shape [batch, 1+N, dim].
        # In `supcon_loss_nlp(logits_unsum, 4)`:
        # features = logits_unsum.
        # negative_sample_num = 4.
        # positive_sample_num = features.shape[1] - 4 - 1. 
        # If shape[1] is 5 (1+4), then pos_num = 0.
        
        positive_sample_num = features.shape[1] - negative_sample_num - 1
        mask = self.create_mask(batch_size, negative_sample_num, positive_sample_num).to(device)

        anchor_feature = features[:, 0]

        contrast=features[:,1:,:]
        contrast_feature = contrast.reshape(batch_size * (negative_sample_num + positive_sample_num), -1)

        # Compute logits
        # anchor_feature: [batch, dim]
        # contrast_feature: [batch * (neg+pos), dim]
        # We want [batch, neg+pos]
        
        # anchor_dot_contrast = torch.div(torch.bmm(anchor_feature.unsqueeze(1), contrast_feature.view(batch_size, negative_sample_num + positive_sample_num , -1).transpose(1, 2)), self.temperature).view(batch_size, -1)
        # This matches the code.

        anchor_dot_contrast = torch.div(torch.bmm(anchor_feature.unsqueeze(1), contrast_feature.view(batch_size, negative_sample_num + positive_sample_num , -1).transpose(1, 2)), self.temperature).view(batch_size, -1)

        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        anchor_dot_contrast = anchor_dot_contrast - logits_max.detach()


        # Apply mask
        logits_mask = torch.zeros(batch_size, negative_sample_num + positive_sample_num ).to(device)
        logits_mask[:, :negative_sample_num] = 0
        logits_mask[:, negative_sample_num:] = 1

        
        mask = mask * logits_mask

        # Compute log_prob
        exp_logits = torch.exp(anchor_dot_contrast) + 1e-20

        log_prob = anchor_dot_contrast - torch.log(exp_logits.sum(1, keepdim=True))

        # Compute mean of log-likelihood over positive

        mean_log_prob_pos=0
        if positive_sample_num!=0:
            mean_log_prob_pos = ((mask * log_prob).sum(1) + 1e-20) / (positive_sample_num + 1e-20)

        # Compute mean of log-likelihood over negative
        mean_log_prob_neg = (((1 - mask) * log_prob).sum(1)  + 1e-20) / (negative_sample_num + 1e-20)

        #loss = - mean_log_prob_pos + mean_log_prob_neg
        #loss = - mean_log_prob_pos + torch.log(torch.exp(mean_log_prob_neg) + 1)
        loss = - mean_log_prob_pos
        loss = loss.mean()
        #print(loss)
        return loss

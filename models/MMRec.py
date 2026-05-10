import torch
from config import Config
import torch.nn as nn
from models.modules.mmrec.news_encoder import NewsEncoder
from models.modules.mmrec.user_encoder import UserEncoder


class MMRec(torch.nn.Module):
    def __init__(self, config: Config):
        super(MMRec, self).__init__()
        self.config = config
        self.model_name = config.model
        self.batch_size = config.batch_size
        self.news_encoder = NewsEncoder(config)
        self.user_encoder = UserEncoder()
        self.criterion = torch.nn.CrossEntropyLoss()
        self.use_user_embedding = False

    def initialize(self):
        # self.news_encoder.initialize()
        # self.user_encoder.initialize()
        if self.use_user_embedding:
            nn.init.uniform_(self.user_embedding.weight, -0.1, 0.1)
            nn.init.zeros_(self.user_embedding.weight[0])

    def forward(self, news_feature, history_feature, log_mask, targets, compute_loss=True):
        imp_news_vecs_t, imp_news_vecs_v = self.news_encoder(**news_feature)
        user_click_news_t, user_click_news_v = self.news_encoder(**history_feature)

        if not compute_loss:
            imp_news_vecs_t = imp_news_vecs_t.unsqueeze(1)
            imp_news_vecs_v = imp_news_vecs_v.unsqueeze(1)

        user_vector = self.user_encoder(imp_news_vecs_t, imp_news_vecs_v, user_click_news_t, user_click_news_v,
                                        log_mask)
        score = (imp_news_vecs_t + imp_news_vecs_v) * user_vector
        score = torch.sum(score, -1)
        if compute_loss:
            loss = self.criterion(score, targets)
            return loss, score
        else:
            return score

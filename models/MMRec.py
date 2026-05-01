import os
import torch
from config import Config
import torch.nn as nn
from models.modules.mmrec.news_encoder import NewsEncoder, BertConfig
from models.modules.mmrec.user_encoder import UserEncoder


class MMRec(torch.nn.Module):
    def __init__(self, config: Config):
        super(MMRec, self).__init__()
        self.config = config
        self.model_name = config.model
        self.batch_size = config.batch_size
        bert_config = BertConfig(
            vocab_size_or_config_json_file=self.config.vocab_size,
            hidden_size=self.config.hidden_size,
            num_hidden_layers=self.config.num_hidden_layers,
            num_attention_heads=self.config.num_attention_heads,
            intermediate_size=self.config.intermediate_size,
            hidden_act=self.config.hidden_act,
            hidden_dropout_prob=self.config.hidden_dropout_prob,
            attention_probs_dropout_prob=self.config.attention_probs_dropout_prob,
            max_position_embeddings=self.config.max_position_embeddings,
            type_vocab_size=self.config.type_vocab_size,
            initializer_range=self.config.initializer_range,
            v_feature_size=self.config.v_feature_size,
            v_target_size=self.config.v_target_size,
            v_hidden_size=self.config.v_hidden_size,
            v_num_hidden_layers=self.config.v_num_hidden_layers,
            v_num_attention_heads=self.config.v_num_attention_heads,
            v_intermediate_size=self.config.v_intermediate_size,
            bi_hidden_size=self.config.bi_hidden_size,
            bi_num_attention_heads=self.config.bi_num_attention_heads,
            v_attention_probs_dropout_prob=self.config.v_attention_probs_dropout_prob,
            v_hidden_act=self.config.v_hidden_act,
            v_hidden_dropout_prob=self.config.v_hidden_dropout_prob,
            v_initializer_range=self.config.v_initializer_range,
            v_biattention_id=self.config.v_biattention_id,
            t_biattention_id=self.config.t_biattention_id,
            predict_feature=self.config.predict_feature,
            fast_mode=self.config.fast_mode,
            fixed_v_layer=self.config.fixed_v_layer,
            fixed_t_layer=self.config.fixed_t_layer,
            in_batch_pairs=self.config.in_batch_pairs,
            fusion_method=self.config.fusion_method,
            with_coattention=self.config.with_coattention,
            image_embedding_dim=self.config.image_embedding_dim
        )
        
        if hasattr(self.config, 'from_pretrained') and self.config.from_pretrained:
            print(f"Loading pre-trained BERT weights from: {self.config.from_pretrained}")
            self.news_encoder = NewsEncoder.from_pretrained(self.config.from_pretrained, config=bert_config)
        else:
            self.news_encoder = NewsEncoder(config=bert_config)
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

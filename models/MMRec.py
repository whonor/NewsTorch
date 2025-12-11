import torch
from config import Config
from models.modules.mmrec.bert_modules import BertConfig
from models.modules.mmrec.news_encoder import NewsEncoder
from models.modules.mmrec.user_encoder import UserEncoder


class MMRec(torch.nn.Module):
    def __init__(self, config: Config):
        super(MMRec, self).__init__()
        self.config = config
        bert_config = BertConfig(
            vocab_size_or_config_json_file=config.vocab_size,
            hidden_size=config.hidden_size,
            num_hidden_layers=config.num_hidden_layers,
            num_attention_heads=config.num_attention_heads,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            hidden_dropout_prob=config.hidden_dropout_prob,
            attention_probs_dropout_prob=config.attention_probs_dropout_prob,
            max_position_embeddings=config.max_position_embeddings,
            type_vocab_size=config.type_vocab_size,
            initializer_range=config.initializer_range,
            v_feature_size=config.v_feature_size,
            v_target_size=config.v_target_size,
            v_hidden_size=config.v_hidden_size,
            v_num_hidden_layers=config.v_num_hidden_layers,
            v_num_attention_heads=config.v_num_attention_heads,
            v_intermediate_size=config.v_intermediate_size,
            bi_hidden_size=config.bi_hidden_size,
            bi_num_attention_heads=config.bi_num_attention_heads,
            v_attention_probs_dropout_prob=config.v_attention_probs_dropout_prob,
            v_hidden_act=config.v_hidden_act,
            v_hidden_dropout_prob=config.v_hidden_dropout_prob,
            v_initializer_range=config.v_initializer_range,
            v_biattention_id=config.v_biattention_id,
            t_biattention_id=config.t_biattention_id,
            predict_feature=config.predict_feature,
            fast_mode=config.fast_mode,
            fixed_v_layer=config.fixed_v_layer,
            fixed_t_layer=config.fixed_t_layer,
            in_batch_pairs=config.in_batch_pairs,
            fusion_method=config.fusion_method,
            with_coattention=config.with_coattention
        )
        self.news_encoder = NewsEncoder.from_pretrained(config.from_pretrained, bert_config, default_gpu=True)
        self.user_encoder = UserEncoder()
        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, news_feature, input_ids, log_ids, log_mask, targets, compute_loss=True):
        batch_size, seq_len = input_ids.shape
        news_vecs_t, news_vecs_v = self.news_encoder(**news_feature)
        # [batch_size,candi,dim]
        imp_news_vecs_t = news_vecs_t[input_ids].view(batch_size, -1, 1024)
        imp_news_vecs_v = news_vecs_v[input_ids].view(batch_size, -1, 1024)

        user_click_news_t = news_vecs_t[log_ids].view(batch_size, -1, 1024)
        user_click_news_v = news_vecs_v[log_ids].view(batch_size, -1, 1024)

        user_vector = self.user_encoder(imp_news_vecs_t, imp_news_vecs_v, user_click_news_t, user_click_news_v,
                                        log_mask)
        score = (imp_news_vecs_t + imp_news_vecs_v) * user_vector
        score = torch.sum(score, -1)
        if compute_loss:
            loss = self.criterion(score, targets)
            return loss, score
        else:
            return score

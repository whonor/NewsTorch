from torch import nn
from models.modules.mmrec.bert_modules import BertPreTrainedModel, BertModel


class NewsEncoder(BertPreTrainedModel):
    def __init__(self, config, dropout_prob=0.1, default_gpu=True):
        super(NewsEncoder, self).__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(dropout_prob)
        self.apply(self.init_bert_weights)

    def forward(
            self,
            input_txt,
            input_imgs,
            image_loc,
            token_type_ids=None,
            attention_mask=None,
            image_attention_mask=None,
            co_attention_mask=None,
            output_all_encoded_layers=False,
    ):
        sequence_output_t, sequence_output_v, pooled_output_t, pooled_output_v, _ = self.bert(
            input_txt,
            input_imgs,
            image_loc,
            token_type_ids,
            attention_mask,
            image_attention_mask,
            co_attention_mask,
            output_all_encoded_layers=False,
        )

        return pooled_output_t, pooled_output_v

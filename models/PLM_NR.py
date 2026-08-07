import torch.nn as nn

from config import Config
from models.modules.click_predictor import DotProduct
from models.modules.plm_nr import NRMSUserEncoder, PLMNewsEncoder


class PLM_NR(nn.Module):
    """RoBERTa-empowered NRMS implementation of the PLM-NR framework."""

    def __init__(self, config: Config):
        super().__init__()
        self.news_encoder = PLMNewsEncoder(config)
        self.user_encoder = NRMSUserEncoder(self.news_encoder, config)
        self.click_predictor = DotProduct()
        self.model_name = config.model
        self.config = config
        self.batch_size = config.batch_size

    def initialize(self):
        self.news_encoder.initialize()
        self.user_encoder.initialize()

    def forward(
        self,
        user_ID,
        user_category,
        user_subCategory,
        user_title_text,
        user_title_mask,
        user_title_entity,
        user_content_text,
        user_content_mask,
        user_content_entity,
        user_history_mask,
        user_history_graph,
        user_history_category_mask,
        user_history_category_indices,
        news_category,
        news_subCategory,
        news_title_text,
        news_title_mask,
        news_title_entity,
        news_content_text,
        news_content_mask,
        news_content_entity,
    ):
        news_representation = self.news_encoder(
            news_title_text,
            news_title_mask,
            news_title_entity,
            news_content_text,
            news_content_mask,
            news_content_entity,
            news_category,
            news_subCategory,
            None,
        )
        if news_representation.dim() == 2:
            news_representation = news_representation.unsqueeze(1)

        user_representation = self.user_encoder(
            user_title_text,
            user_title_mask,
            user_title_entity,
            user_content_text,
            user_content_mask,
            user_content_entity,
            user_category,
            user_subCategory,
            user_history_mask,
            user_history_graph,
            user_history_category_mask,
            user_history_category_indices,
            None,
            news_representation,
        )
        logits = self.click_predictor(
            user_representation.unsqueeze(1),
            news_representation.transpose(1, 2),
        )
        return logits.squeeze(1) if logits.size(1) == 1 else logits

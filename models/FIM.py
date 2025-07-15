from config import Config
import torch.nn as nn
from models.modules import userEncoders, newsEncoders


class FIM(nn.Module):
    def __init__(self, config: Config):
        super(FIM, self).__init__()
        self.news_encoder = newsEncoders.HDC(config)
        self.user_encoder = userEncoders.FIM(self.news_encoder, config)

        self.use_user_embedding = False
        self.model_name = config.model
        self.batch_size = config.batch_size
        self.config = config

        def compute_convolution_pooling_output_size(input_size):
            conv1_size = input_size - config.conv3D_kernel_size_first + 1
            pool1_size = (conv1_size - config.maxpooling3D_size) // config.maxpooling3D_stride + 1
            conv2_size = pool1_size - config.conv3D_kernel_size_second + 1
            pool2_size = (conv2_size - config.maxpooling3D_size) // config.maxpooling3D_stride + 1
            return pool2_size

        feature_size = compute_convolution_pooling_output_size(self.news_encoder.HDC_sequence_length) * \
                       compute_convolution_pooling_output_size(self.news_encoder.HDC_sequence_length) * \
                       compute_convolution_pooling_output_size(config.max_history_num) * \
                       config.conv3D_filter_num_second
        self.click_predictor = nn.Linear(in_features=feature_size, out_features=1, bias=True)


    def initialize(self):
        self.news_encoder.initialize()
        self.user_encoder.initialize()
        if self.use_user_embedding:
            nn.init.uniform_(self.user_embedding.weight, -0.1, 0.1)
            nn.init.zeros_(self.user_embedding.weight[0])
        nn.init.xavier_uniform_(self.click_predictor.weight)
        nn.init.zeros_(self.click_predictor.bias)

    def forward(self, user_ID, user_category, user_subCategory, user_title_text, user_title_mask, user_title_entity, user_content_text, user_content_mask, user_content_entity, user_history_mask, user_history_graph, user_history_category_mask, user_history_category_indices, \
                      news_category, news_subCategory, news_title_text, news_title_mask, news_title_entity, news_content_text, news_content_mask, news_content_entity):
        user_embedding = self.dropout(self.user_embedding(user_ID)) if self.use_user_embedding else None
        # [batch, 5, 400]
        news_representation = self.news_encoder(news_title_text, news_title_mask, news_title_entity, news_content_text, news_content_mask, news_content_entity, news_category, news_subCategory, user_embedding)
        # [batch, 400]
        user_representation = self.user_encoder(user_title_text, user_title_mask, user_title_entity, user_content_text, user_content_mask, user_content_entity, user_category, user_subCategory, \
                                                user_history_mask, user_history_graph, user_history_category_mask, user_history_category_indices, user_embedding, news_representation)
        #
        logits = self.click_predictor(user_representation).squeeze(dim=2)

        return logits


import pickle
from config import Config
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence
from models.modules.layers import Conv1D, Conv2D_Pool, MultiHeadAttention, Attention, \
    ScaledDotProduct_CandidateAttention, CandidateAttention, AdditiveAttention, MultiHeadSelfAttention, \
    PositionalEncoding


class NewsEncoder(nn.Module):
    def __init__(self, config: Config):
        super(NewsEncoder, self).__init__()
        self.word_embedding_dim = config.word_embedding_dim
        self.word_embedding = nn.Embedding(num_embeddings=config.vocabulary_size, embedding_dim=self.word_embedding_dim)
        if config.dataset_name == 'MIND':
            with open('cache/word_embedding-' + str(config.word_threshold) + '-' + str(
                    config.word_embedding_dim) + '-' + config.tokenizer + '-' + str(
                    config.max_title_length) + '-' + str(config.max_abstract_length) + '-' + config.dataset_size + '.pkl',
                      'rb') as word_embedding_f:
                self.word_embedding.weight.data.copy_(pickle.load(word_embedding_f))
        elif config.dataset_name == 'ebnerd':
            with open('cache/%s/word_embedding-' % config.dataset_name + str(config.word_threshold) + '-' + str(
                    config.word_embedding_dim) + '-' + config.tokenizer + '-' + str(
                    config.max_title_length) + '-' + str(config.max_abstract_length) + '-' + config.dataset_size + '.pkl',
                      'rb') as word_embedding_f:
                self.word_embedding.weight.data.copy_(pickle.load(word_embedding_f))

        self.category_embedding = nn.Embedding(num_embeddings=config.category_num, embedding_dim=config.category_embedding_dim)
        self.subCategory_embedding = nn.Embedding(num_embeddings=config.subCategory_num, embedding_dim=config.subCategory_embedding_dim)
        self.dropout = nn.Dropout(p=config.dropout_rate, inplace=True)
        self.dropout_ = nn.Dropout(p=config.dropout_rate, inplace=False)
        self.auxiliary_loss = None

    def initialize(self):
        nn.init.uniform_(self.category_embedding.weight, -0.1, 0.1)
        nn.init.uniform_(self.subCategory_embedding.weight, -0.1, 0.1)
        nn.init.zeros_(self.subCategory_embedding.weight[0])

    # Input
    # title_text          : [batch_size, news_num, max_title_length]
    # title_mask          : [batch_size, news_num, max_title_length]
    # title_entity        : [batch_size, news_num, max_title_length]
    # content_text        : [batch_size, news_num, max_content_length]
    # content_mask        : [batch_size, news_num, max_content_length]
    # content_entity      : [batch_size, news_num, max_content_length]
    # category            : [batch_size, news_num]
    # subCategory         : [batch_size, news_num]
    # user_embedding      : [batch_size, user_embedding_dim]
    # Output
    # news_representation : [batch_size, news_num, news_embedding_dim]
    def forward(self, title_text, title_mask, title_entity, content_text, content_mask, content_entity, category, subCategory, user_embedding):
        raise Exception('Function forward must be implemented at sub-class')

    # Input
    # news_representation : [batch_size, news_num, unfused_news_embedding_dim]
    # category            : [batch_size, news_num]
    # subCategory         : [batch_size, news_num]
    # Output
    # news_representation : [batch_size, news_num, news_embedding_dim]
    def feature_fusion(self, news_representation, category, subCategory):
        # [batch_size, news_num, category_embedding_dim]
        category_representation = self.category_embedding(category)
        # [batch_size, news_num, subCategory_embedding_dim]
        subCategory_representation = self.subCategory_embedding(subCategory)
        # [batch_size, news_num, news_embedding_dim]
        news_representation = torch.cat([news_representation, self.dropout(category_representation), self.dropout(subCategory_representation)], dim=2)
        return news_representation

########################################################################################################################
# CNE-SUE
class CNE(NewsEncoder):
    def __init__(self, config: Config):
        super(CNE, self).__init__(config)
        self.max_title_length = config.max_title_length
        self.max_content_length = config.max_abstract_length
        self.word_embedding_dim = config.word_embedding_dim
        self.hidden_dim = config.hidden_dim
        self.news_embedding_dim = config.hidden_dim * 4 + config.category_embedding_dim + config.subCategory_embedding_dim
        # selective LSTM encoder
        self.title_lstm = nn.LSTM(self.word_embedding_dim, self.hidden_dim, batch_first=True, bidirectional=True)
        self.content_lstm = nn.LSTM(self.word_embedding_dim, self.hidden_dim, batch_first=True, bidirectional=True)
        self.title_H = nn.Linear(self.hidden_dim * 2, self.hidden_dim * 2, bias=False)
        self.title_M = nn.Linear(self.hidden_dim * 2, self.hidden_dim * 2, bias=True)
        self.content_H = nn.Linear(self.hidden_dim * 2, self.hidden_dim * 2, bias=False)
        self.content_M = nn.Linear(self.hidden_dim * 2, self.hidden_dim * 2, bias=True)
        # self-attention
        self.title_self_attention = Attention(self.hidden_dim * 2, config.attention_dim)
        self.content_self_attention = Attention(self.hidden_dim * 2, config.attention_dim)
        # cross-attention
        self.title_cross_attention = ScaledDotProduct_CandidateAttention(self.hidden_dim * 2, self.hidden_dim * 2, config.attention_dim)
        self.content_cross_attention = ScaledDotProduct_CandidateAttention(self.hidden_dim * 2, self.hidden_dim * 2, config.attention_dim)

    def initialize(self):
        super().initialize()
        for parameter in self.title_lstm.parameters():
            if len(parameter.size()) >= 2:
                nn.init.orthogonal_(parameter.data)
            else:
                nn.init.zeros_(parameter.data)
        for parameter in self.content_lstm.parameters():
            if len(parameter.size()) >= 2:
                nn.init.orthogonal_(parameter.data)
            else:
                nn.init.zeros_(parameter.data)
        nn.init.xavier_uniform_(self.title_H.weight, gain=nn.init.calculate_gain('sigmoid'))
        nn.init.xavier_uniform_(self.title_M.weight, gain=nn.init.calculate_gain('sigmoid'))
        nn.init.zeros_(self.title_M.bias)
        nn.init.xavier_uniform_(self.content_H.weight, gain=nn.init.calculate_gain('sigmoid'))
        nn.init.xavier_uniform_(self.content_M.weight, gain=nn.init.calculate_gain('sigmoid'))
        nn.init.zeros_(self.content_M.bias)
        self.title_self_attention.initialize()
        self.content_self_attention.initialize()
        self.title_cross_attention.initialize()
        self.content_cross_attention.initialize()

    def forward(self, title_text, title_mask, title_entity, content_text, content_mask, content_entity, category, subCategory, user_embedding):
        batch_size = title_text.size(0)
        news_num = title_text.size(1)
        batch_news_num = batch_size * news_num
        title_mask = title_mask.view([batch_news_num, self.max_title_length])
        content_mask = content_mask.view([batch_news_num, self.max_content_length])
        title_mask[:, 0] = 1   # To avoid empty input of LSTM
        content_mask[:, 0] = 1 # To avoid empty input of LSTM
        title_length = title_mask.sum(dim=1, keepdim=False).long()
        content_length = content_mask.sum(dim=1, keepdim=False).long()
        sorted_title_length, sorted_title_indices = torch.sort(title_length, descending=True)
        _, desorted_title_indices = torch.sort(sorted_title_indices, descending=False)
        sorted_content_length, sorted_content_indices = torch.sort(content_length, descending=True)
        _, desorted_content_indices = torch.sort(sorted_content_indices, descending=False)
        # 1. word embedding
        title = self.dropout(self.word_embedding(title_text)).view([batch_news_num, self.max_title_length, self.word_embedding_dim])
        content = self.dropout(self.word_embedding(content_text)).view([batch_news_num, self.max_content_length, self.word_embedding_dim])
        sorted_title = pack_padded_sequence(title.index_select(0, sorted_title_indices), sorted_title_length.cpu(), batch_first=True)
        sorted_content = pack_padded_sequence(content.index_select(0, sorted_content_indices), sorted_content_length.cpu(), batch_first=True)
        # 2. selective LSTM encoding
        sorted_title_h, (sorted_title_h_n, sorted_title_c_n) = self.title_lstm(sorted_title)
        sorted_content_h, (sorted_content_h_n, sorted_content_c_n) = self.content_lstm(sorted_content)
        sorted_title_m = torch.cat([sorted_title_c_n[0], sorted_title_c_n[1]], dim=1)
        sorted_content_m = torch.cat([sorted_content_c_n[0], sorted_content_c_n[1]], dim=1)
        sorted_title_h, _ = pad_packed_sequence(sorted_title_h, batch_first=True, total_length=self.max_title_length)
        sorted_content_h, _ = pad_packed_sequence(sorted_content_h, batch_first=True, total_length=self.max_content_length)
        sorted_title_gate = torch.sigmoid(self.title_H(sorted_title_h) + self.title_M(sorted_content_m).unsqueeze(dim=1))
        sorted_content_gate = torch.sigmoid(self.content_H(sorted_content_h) + self.content_M(sorted_title_m).unsqueeze(dim=1))
        title_h = (sorted_title_h * sorted_title_gate).index_select(0, desorted_title_indices)
        content_h = (sorted_content_h * sorted_content_gate).index_select(0, desorted_content_indices)
        # 3. self-attention
        title_self = self.title_self_attention(title_h, title_mask)
        content_self = self.content_self_attention(content_h, content_mask)
        # 4. cross-attention
        title_cross = self.title_cross_attention(title_h, content_self, title_mask)
        content_cross = self.content_cross_attention(content_h, title_self, content_mask)
        news_representation = torch.cat([title_self + title_cross, content_self + content_cross], dim=1).view([batch_size, news_num, self.hidden_dim * 4])
        # 5. feature fusion
        # [batch_size, news_num, news_embedding_dim]
        news_representation = self.feature_fusion(news_representation, category, subCategory)
        return news_representation

########################################################################################################################
# LSTUR
class CNN(NewsEncoder):
    def __init__(self, config: Config):
        super(CNN, self).__init__(config)
        self.max_sentence_length = config.max_title_length
        self.cnn_kernel_num = config.cnn_kernel_num
        self.conv = Conv1D(config.cnn_method, config.word_embedding_dim, config.cnn_kernel_num, config.cnn_window_size)
        self.attention = Attention(config.cnn_kernel_num, config.attention_dim)
        self.news_embedding_dim = config.cnn_kernel_num + config.category_embedding_dim + config.subCategory_embedding_dim

    def initialize(self):
        super().initialize()
        self.attention.initialize()

    def forward(self, title_text, title_mask, title_entity, content_text, content_mask, content_entity, category, subCategory, user_embedding):
        batch_size = title_text.size(0)
        news_num = title_text.size(1)
        batch_news_num = batch_size * news_num
        # [batch_size * news_num, max_sentence_length]
        mask = title_mask.view([batch_news_num, self.max_sentence_length])
        # 1. word embedding
        w = self.dropout(self.word_embedding(title_text)).view([batch_news_num, self.max_sentence_length, self.word_embedding_dim])
        # 2. CNN encoding
        c = self.dropout_(self.conv(w.permute(0, 2, 1)).permute(0, 2, 1))
        # 3. attention layer
        news_representation = self.attention(c, mask=mask).view([batch_size, news_num, self.cnn_kernel_num])
        # 4. feature fusion
        # [batch_size, news_num, news_embedding_dim]
        news_representation = self.feature_fusion(news_representation, category, subCategory)
        return news_representation

########################################################################################################################
# NRMS
class MHSA(NewsEncoder):
    def __init__(self, config: Config):
        super(MHSA, self).__init__(config)
        self.max_sentence_length = config.max_title_length
        self.feature_dim = config.head_num * config.head_dim
        self.multiheadAttention = MultiHeadAttention(config.head_num, config.word_embedding_dim, config.max_title_length, config.max_title_length, config.head_dim, config.head_dim)
        self.attention = Attention(config.head_num*config.head_dim, config.attention_dim)
        self.news_embedding_dim = config.head_num * config.head_dim + config.category_embedding_dim + config.subCategory_embedding_dim

    def initialize(self):
        super().initialize()
        self.multiheadAttention.initialize()
        self.attention.initialize()

    def forward(self, title_text, title_mask, title_entity, content_text, content_mask, content_entity, category, subCategory, user_embedding):
        batch_size = title_text.size(0)
        news_num = title_text.size(1)
        batch_news_num = batch_size * news_num
        mask = title_mask.view([batch_news_num, self.max_sentence_length])
        # 1. word embedding
        w = self.dropout(self.word_embedding(title_text)).view([batch_news_num, self.max_sentence_length, self.word_embedding_dim])
        # 2. multi-head self-attention
        c = self.dropout(self.multiheadAttention(w, w, w, mask))
        # 3. attention layer
        news_representation = self.attention(c, mask=mask).view([batch_size, news_num, self.feature_dim])
        # 4. feature fusion
        news_representation = self.feature_fusion(news_representation, category, subCategory)
        # [batch_size, news_num, news_embedding_dim]
        return news_representation

########################################################################################################################
# DKN
class KCNN(NewsEncoder):
    def __init__(self, config: Config):
        super(KCNN, self).__init__(config)
        self.max_title_length = config.max_title_length
        self.cnn_kernel_num = config.cnn_kernel_num
        self.entity_embedding_dim = config.entity_embedding_dim
        self.context_embedding_dim = config.context_embedding_dim
        self.entity_embedding = nn.Embedding(num_embeddings=config.entity_size, embedding_dim=self.entity_embedding_dim)
        self.context_embedding = nn.Embedding(num_embeddings=config.entity_size, embedding_dim=self.context_embedding_dim)
        with open(config.data_path + '/entity_embedding-%s.pkl' % config.dataset, 'rb') as entity_embedding_f:
            self.entity_embedding.weight.data.copy_(pickle.load(entity_embedding_f))
        with open(config.data_path + '/context_embedding-%s.pkl' % config.dataset, 'rb') as context_embedding_f:
            self.context_embedding.weight.data.copy_(pickle.load(context_embedding_f))
        self.M_entity = nn.Linear(self.entity_embedding_dim, self.word_embedding_dim, bias=True)
        self.M_context = nn.Linear(self.context_embedding_dim, self.word_embedding_dim, bias=True)
        self.knowledge_cnn = Conv2D_Pool(config.cnn_method, config.word_embedding_dim, config.cnn_kernel_num, config.cnn_window_size, 3)
        self.news_embedding_dim = config.cnn_kernel_num + config.category_embedding_dim + config.subCategory_embedding_dim

    def initialize(self):
        super().initialize()
        nn.init.xavier_uniform_(self.M_entity.weight, gain=nn.init.calculate_gain('tanh'))
        nn.init.zeros_(self.M_entity.bias)
        nn.init.xavier_uniform_(self.M_context.weight, gain=nn.init.calculate_gain('tanh'))
        nn.init.zeros_(self.M_context.bias)

    def forward(self, title_text, title_mask, title_entity, content_text, content_mask, content_entity, category, subCategory, user_embedding):
        batch_size = title_text.size(0)
        news_num = title_text.size(1)
        batch_news_num = batch_size * news_num
        # 1. word & entity & context embedding
        word_embedding = self.word_embedding(title_text).view([batch_news_num, self.max_title_length, self.word_embedding_dim])
        entity_embedding = self.entity_embedding(title_entity).view([batch_news_num, self.max_title_length, self.entity_embedding_dim])
        context_embedding = self.context_embedding(title_entity).view([batch_news_num, self.max_title_length, self.context_embedding_dim])
        W = torch.stack([word_embedding, torch.tanh(self.M_entity(entity_embedding)), torch.tanh(self.M_context(context_embedding))], dim=3).permute(0, 2, 1, 3)
        # 2. knowledge-aware CNN
        news_representation = self.knowledge_cnn(W).view([batch_size, news_num, self.cnn_kernel_num])
        # 3. feature fusion
        news_representation = self.feature_fusion(news_representation, category, subCategory)
        # [batch_size, news_num, news_embedding_dim]
        return news_representation

########################################################################################################################
# FIM
class HDC(NewsEncoder):
    def __init__(self, config: Config):
        super(HDC, self).__init__(config)
        self.category_embedding = nn.Embedding(num_embeddings=config.category_num, embedding_dim=config.word_embedding_dim)
        self.subCategory_embedding = nn.Embedding(num_embeddings=config.subCategory_num, embedding_dim=config.word_embedding_dim)
        self.HDC_sequence_length = config.max_title_length + 2
        self.HDC_filter_num = config.HDC_filter_num
        self.dilated_conv1 = nn.Conv1d(in_channels=config.word_embedding_dim, out_channels=self.HDC_filter_num, kernel_size=config.HDC_window_size, padding=(config.HDC_window_size - 1) // 2, dilation=1)
        self.dilated_conv2 = nn.Conv1d(in_channels=self.HDC_filter_num, out_channels=self.HDC_filter_num, kernel_size=config.HDC_window_size, padding=(config.HDC_window_size - 1) // 2 + 1, dilation=2)
        self.dilated_conv3 = nn.Conv1d(in_channels=self.HDC_filter_num, out_channels=self.HDC_filter_num, kernel_size=config.HDC_window_size, padding=(config.HDC_window_size - 1) // 2 + 2, dilation=3)
        self.layer_norm1 = nn.LayerNorm([self.HDC_filter_num, self.HDC_sequence_length])
        self.layer_norm2 = nn.LayerNorm([self.HDC_filter_num, self.HDC_sequence_length])
        self.layer_norm3 = nn.LayerNorm([self.HDC_filter_num, self.HDC_sequence_length])
        self.news_embedding_dim = None

    def initialize(self):
        super().initialize()

    def forward(self, title_text, title_mask, title_entity, content_text, content_mask, content_entity, category, subCategory, user_embedding):
        batch_size = title_text.size(0)
        news_num = title_text.size(1)
        batch_news_num = batch_size * news_num
        # 1. sequence embeddings
        word_embedding = self.word_embedding(title_text).permute(0, 1, 3, 2)
        category_embedding = self.category_embedding(category).unsqueeze(dim=3)
        subCategory_embedding = self.subCategory_embedding(subCategory).unsqueeze(dim=3)
        d0 = torch.cat([category_embedding, subCategory_embedding, word_embedding], dim=3)
        d0 = d0.view([batch_news_num, self.word_embedding_dim, self.HDC_sequence_length])
        # 2. hierarchical dilated convolution
        d1 = F.relu(self.layer_norm1(self.dilated_conv1(d0)), inplace=True)
        d2 = F.relu(self.layer_norm2(self.dilated_conv2(d1)), inplace=True)
        d3 = F.relu(self.layer_norm3(self.dilated_conv3(d2)), inplace=True)
        d0 = d0.view([batch_size, news_num, self.word_embedding_dim, self.HDC_sequence_length])
        # [batch_size, news_num, 3, HDC_filter_num, HDC_sequence_length]
        dL = torch.stack([d1, d2, d3], dim=1).view([batch_size, news_num, 3, self.HDC_filter_num, self.HDC_sequence_length])
        return (d0, dL)

########################################################################################################################
# NAML
class NAML(NewsEncoder):
    def __init__(self, config: Config):
        super(NAML, self).__init__(config)
        self.max_title_length = config.max_title_length
        self.max_content_length = config.max_abstract_length
        self.cnn_kernel_num = config.cnn_kernel_num
        self.news_embedding_dim = config.cnn_kernel_num
        self.title_conv = Conv1D(config.cnn_method, config.word_embedding_dim, config.cnn_kernel_num, config.cnn_window_size)
        self.content_conv = Conv1D(config.cnn_method, config.word_embedding_dim, config.cnn_kernel_num, config.cnn_window_size)
        self.title_attention = Attention(config.cnn_kernel_num, config.attention_dim)
        self.content_attention = Attention(config.cnn_kernel_num, config.attention_dim)
        self.category_affine = nn.Linear(config.category_embedding_dim, config.cnn_kernel_num, bias=True)
        self.subCategory_affine = nn.Linear(config.subCategory_embedding_dim, config.cnn_kernel_num, bias=True)
        self.affine1 = nn.Linear(config.cnn_kernel_num, config.attention_dim, bias=True)
        self.affine2 = nn.Linear(config.attention_dim, 1, bias=False)

    def initialize(self):
        super().initialize()
        self.title_attention.initialize()
        self.content_attention.initialize()
        nn.init.xavier_uniform_(self.category_affine.weight)
        nn.init.zeros_(self.category_affine.bias)
        nn.init.xavier_uniform_(self.subCategory_affine.weight)
        nn.init.zeros_(self.subCategory_affine.bias)
        nn.init.xavier_uniform_(self.affine1.weight)
        nn.init.zeros_(self.affine1.bias)
        nn.init.xavier_uniform_(self.affine2.weight)

    def forward(self, title_text, title_mask, title_entity, content_text, content_mask, content_entity, category, subCategory, user_embedding):
        batch_size = title_text.size(0)
        news_num = title_text.size(1)
        batch_news_num = batch_size * news_num
        # 1. word embedding
        title_w = self.dropout(self.word_embedding(title_text)).view([batch_news_num, self.max_title_length, self.word_embedding_dim])
        content_w = self.dropout(self.word_embedding(content_text)).view([batch_news_num, self.max_content_length, self.word_embedding_dim])
        # 2. CNN encoding
        title_c = self.dropout_(self.title_conv(title_w.permute(0, 2, 1)).permute(0, 2, 1))
        content_c = self.dropout_(self.content_conv(content_w.permute(0, 2, 1)).permute(0, 2, 1))
        # 3. attention layer
        title_representation = self.title_attention(title_c).view([batch_size, news_num, self.cnn_kernel_num])
        content_representation = self.content_attention(content_c).view([batch_size, news_num, self.cnn_kernel_num])
        # 4. category and subCategory encoding
        category_representation = F.relu(self.category_affine(self.category_embedding(category)), inplace=True)
        subCategory_representation = F.relu(self.subCategory_affine(self.subCategory_embedding(subCategory)), inplace=True)
        # 5. multi-view attention
        feature = torch.stack([title_representation, content_representation, category_representation, subCategory_representation], dim=2)
        alpha = F.softmax(self.affine2(torch.tanh(self.affine1(feature))), dim=2)
        # [batch_size, news_num, cnn_kernel_num]
        news_representation = (feature * alpha).sum(dim=2, keepdim=False)
        return news_representation

########################################################################################################################
# NPA
class PNE(NewsEncoder):
    def __init__(self, config: Config):
        super(PNE, self).__init__(config)
        self.max_sentence_length = config.max_title_length
        self.cnn_kernel_num = config.cnn_kernel_num
        self.personalized_embedding_dim = config.personalized_embedding_dim
        self.conv = Conv1D(config.cnn_method, config.word_embedding_dim, config.cnn_kernel_num, config.cnn_window_size)
        self.dense = nn.Linear(config.user_embedding_dim, config.personalized_embedding_dim, bias=True)
        self.personalizedAttention = CandidateAttention(config.cnn_kernel_num, config.personalized_embedding_dim, config.attention_dim)
        self.news_embedding_dim = config.cnn_kernel_num + config.category_embedding_dim + config.subCategory_embedding_dim

    def initialize(self):
        super().initialize()
        nn.init.xavier_uniform_(self.dense.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.zeros_(self.dense.bias)
        self.personalizedAttention.initialize()

    def forward(self, title_text, title_mask, title_entity, content_text, content_mask, content_entity, category, subCategory, user_embedding):
        batch_size = title_text.size(0)
        news_num = title_text.size(1)
        batch_news_num = batch_size * news_num
        mask = title_mask.view([batch_news_num, self.max_sentence_length])
        # 1. word embedding
        w = self.dropout(self.word_embedding(title_text)).view([batch_news_num, self.max_sentence_length, self.word_embedding_dim])
        # 2. CNN encoding
        c = self.dropout_(self.conv(w.permute(0, 2, 1)).permute(0, 2, 1))
        # 3. attention layer
        q_w = F.relu(self.dense(user_embedding), inplace=True).repeat([news_num, 1])
        news_representation = self.personalizedAttention(c, q_w, mask).view([batch_size, news_num, self.cnn_kernel_num])
        # 4. feature fusion
        news_representation = self.feature_fusion(news_representation, category, subCategory)
        # [batch_size, news_num, news_embedding_dim]
        return news_representation


########################################################################################################################
# TANR
class TANR(NewsEncoder):
    def __init__(self, config: Config):
        super(TANR, self).__init__(config)
        self.max_sentence_length = config.max_title_length
        self.cnn_kernel_num = config.cnn_kernel_num
        self.conv = Conv1D(config.cnn_method, config.word_embedding_dim, config.cnn_kernel_num, config.cnn_window_size)
        self.attention = Attention(config.cnn_kernel_num, config.attention_dim)
        self.news_embedding_dim = config.cnn_kernel_num

    def initialize(self):
        super().initialize()
        self.attention.initialize()

    def forward(self, title_text, title_mask, title_entity, content_text, content_mask, content_entity, category, subCategory, user_embedding):
        batch_size = title_text.size(0)
        news_num = title_text.size(1)
        batch_news_num = batch_size * news_num
        mask = title_mask.view([batch_news_num, self.max_sentence_length])
        # 1. word embedding
        w = self.dropout(self.word_embedding(title_text)).view([batch_news_num, self.max_sentence_length, self.word_embedding_dim])
        # 2. CNN encoding
        c = self.dropout_(self.conv(w.permute(0, 2, 1)).permute(0, 2, 1))
        # 3. attention layer
        news_representation = self.attention(c, mask=mask).view([batch_size, news_num, self.cnn_kernel_num])
        # [batch_size, news_num, news_embedding_dim]
        return news_representation

########################################################################################################################
# MINS
class MINS_NE(NewsEncoder):
    def __init__(self, config: Config):
        super(MINS_NE, self).__init__(config)
        self.config = config
        self.max_title_length = config.max_title_length
        self.max_content_length = config.max_abstract_length
        self.title_multihead_self_attention = MultiHeadSelfAttention(
            self.word_embedding_dim, config.num_attention_heads)
        self.abstract_multihead_self_attention = MultiHeadSelfAttention(
            self.word_embedding_dim, config.num_attention_heads)
        self.title_attention = Attention(self.word_embedding_dim, config.attention_dim)
        self.abstract_attention = Attention(self.word_embedding_dim, config.attention_dim)
        self.category_affine = nn.Linear(config.category_embedding_dim, self.word_embedding_dim, bias=True)
        self.subCategory_affine = nn.Linear(config.subCategory_embedding_dim, self.word_embedding_dim, bias=True)
        self.affine1 = nn.Linear(self.word_embedding_dim, config.attention_dim, bias=True)
        self.affine2 = nn.Linear(config.attention_dim, 1, bias=False)
        self.addiattention = AdditiveAttention(config.query_vector_dim, self.word_embedding_dim)
        self.news_embedding_dim = self.word_embedding_dim
        self.history_length = config.max_history_num

    def initialize(self):
        super().initialize()
        self.title_attention.initialize()
        self.abstract_attention.initialize()
        nn.init.xavier_uniform_(self.category_affine.weight)
        nn.init.zeros_(self.category_affine.bias)
        nn.init.xavier_uniform_(self.subCategory_affine.weight)
        nn.init.zeros_(self.subCategory_affine.bias)
        nn.init.xavier_uniform_(self.affine1.weight)
        nn.init.zeros_(self.affine1.bias)
        nn.init.xavier_uniform_(self.affine2.weight)

    def forward(self, title_text, title_mask, title_entity, content_text, content_mask, content_entity, category, subCategory, user_embedding):
        batch_size = title_text.size(0)
        news_num = title_text.size(1)
        batch_news_num = batch_size * news_num
        # 1. word embedding
        # [batch_size * news_num, max_title_length, word_embedding_dim]
        title_w = self.dropout(self.word_embedding(title_text)).view([batch_news_num, self.max_title_length, self.word_embedding_dim])
        content_w = self.dropout(self.word_embedding(content_text)).view([batch_news_num, self.max_content_length, self.word_embedding_dim])
        # 2. MHSA encoding
        # [batch_size * news_num, max_title_length, word_embedding_dim]
        title = self.dropout_(self.title_multihead_self_attention(title_w))
        content = self.dropout_(self.abstract_multihead_self_attention(content_w))
        # 3. attention layer
        # [batch, news_num, word_embedding_dim]
        title_representation = self.title_attention(title).view([batch_size, news_num, self.word_embedding_dim])
        # [batch, news_num, word_embedding_dim]
        content_representation = self.abstract_attention(content).view([batch_size, news_num, self.word_embedding_dim])
        # 4. category and subCategory encoding
        # [batch, news_num, word_embedding_dim]
        category_representation = F.relu(self.category_affine(self.category_embedding(category)), inplace=True)
        # [batch, news_num, word_embedding_dim]
        subCategory_representation = F.relu(self.subCategory_affine(self.subCategory_embedding(subCategory)), inplace=True)
        # 5. additive attention
        # [batch_news_num, 4, word_embedding_dim]
        all_vectors = torch.stack([title_representation.view(batch_news_num, self.word_embedding_dim),
                                   content_representation.view(batch_news_num, self.word_embedding_dim),
                                   category_representation.view(batch_news_num, self.word_embedding_dim),
                                   subCategory_representation.view(batch_news_num, self.word_embedding_dim)], dim=1)
        # [batch_news_num, word_embedding_dim]
        news_representation = self.addiattention(all_vectors)
        # [batch, news_num, word_embedding_dim]
        news_representation = news_representation.view(batch_size, news_num, self.word_embedding_dim)  # Expand to match the news_num dimension
        # [batch, news_num, word_embedding_dim]
        return news_representation


########################################################################################################################
# CenNewsRec
class CNNMHSAAddAtt(NewsEncoder):
    def __init__(self,config: Config):
        super(CNNMHSAAddAtt, self).__init__(config)
        self.config = config
        if not isinstance(self.config.dropout_probability, float):
            raise ValueError(
                f"Expected keyword argument `dropout_probability` to be a `float` but got {self.config.dropout_probability}"
            )
        self.news_embedding_dim = self.word_embedding_dim
        self.max_title_length = config.max_title_length
        # initialize
        self.cnn = nn.Conv1d(
            in_channels=self.config.word_embedding_dim, out_channels=self.config.num_filters, kernel_size=self.config.window_size, padding=1
        )
        self.multihead_attention = nn.MultiheadAttention(
            embed_dim=self.config.num_filters, num_heads=self.config.num_heads
        )
        self.additive_attention = AdditiveAttention(candidate_vector_dim=self.config.num_filters, query_vector_dim=self.config.query_dim)
        self.dropout = nn.Dropout(self.config.dropout_probability)

    def forward(self, title_text, title_mask, title_entity, content_text, content_mask, content_entity, category, subCategory, user_embedding) -> torch.Tensor:
        batch_size = title_text.size(0)
        news_num = title_text.size(1)
        batch_news_num = batch_size * news_num
        # [batch_size * news_num, max_title_length, word_embedding_dim]
        title_w = self.dropout(self.word_embedding(title_text)).view([batch_news_num, self.max_title_length,
                                                                      self.word_embedding_dim])
        # [batch_news_num, num_filters, num_words_text]
        text_vector = self.cnn(title_w.permute(0, 2, 1))
        text_vector = F.relu(text_vector)
        text_vector = self.dropout(text_vector)

        # [num_words_text, batch_news_num, num_filters]
        text_vector = text_vector.permute(2, 0, 1)
        text_vector, _ = self.multihead_attention(text_vector, text_vector, text_vector)
        # [batch_news, num_words_text, num_filters]
        text_vector = self.dropout(text_vector).permute(1, 0, 2)

        # [batch_size, news_num, num_filters]
        text_vector = self.additive_attention(text_vector.view(batch_news_num, -1, self.config.num_filters))
        text_vector = text_vector.view(batch_size, news_num, self.config.num_filters)

        return text_vector


########################################################################################################################
# IPNR
class IPNR_NE(NewsEncoder):
    def __init__(self, config: Config):
        super(IPNR_NE, self).__init__(config)
        self.max_title_length = config.max_title_length
        self.max_content_length = config.max_abstract_length
        self.cnn_kernel_num = config.cnn_kernel_num
        self.news_embedding_dim = config.cnn_kernel_num
        self.title_conv = Conv1D(config.cnn_method, config.word_embedding_dim, config.cnn_kernel_num,
                                 config.cnn_window_size)
        self.content_conv = Conv1D(config.cnn_method, config.word_embedding_dim, config.cnn_kernel_num,
                                   config.cnn_window_size)
        self.title_attention = Attention(config.cnn_kernel_num, config.attention_dim)
        self.content_attention = Attention(config.cnn_kernel_num, config.attention_dim)
        self.category_affine = nn.Linear(in_features=config.category_embedding_dim, out_features=config.cnn_kernel_num,
                                         bias=True)
        self.subCategory_affine = nn.Linear(in_features=config.subCategory_embedding_dim,
                                            out_features=config.cnn_kernel_num, bias=True)
        self.affine1 = nn.Linear(in_features=config.cnn_kernel_num, out_features=config.attention_dim, bias=True)
        self.affine2 = nn.Linear(in_features=config.attention_dim, out_features=1, bias=False)

        self.transformer_encoder_layer = nn.TransformerEncoderLayer(config.cnn_kernel_num, nhead=20, dropout=0.2, activation="relu")
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_encoder_layer, num_layers=2)
        self.PE = PositionalEncoding(self.word_embedding_dim, dropout=0.2, max_len=128)
        self.news_attention = Attention(config.cnn_kernel_num, config.attention_dim)

    def initialize(self):
        super().initialize()
        self.title_attention.initialize()
        self.content_attention.initialize()
        nn.init.xavier_uniform_(self.category_affine.weight)
        nn.init.zeros_(self.category_affine.bias)
        nn.init.xavier_uniform_(self.subCategory_affine.weight)
        nn.init.zeros_(self.subCategory_affine.bias)
        nn.init.xavier_uniform_(self.affine1.weight)
        nn.init.zeros_(self.affine1.bias)
        nn.init.xavier_uniform_(self.affine2.weight)

    def forward(self, title_text, title_mask, title_entity, content_text, content_mask, content_entity, category,
                subCategory, user_embedding):
        batch_size = title_text.size(0)
        news_num = title_text.size(1)
        batch_news_num = batch_size * news_num
        # 1. word embedding
        title_w = self.dropout(self.word_embedding(title_text)).view([batch_news_num, self.max_title_length,
                                                                      self.word_embedding_dim])  # [batch_size * news_num, max_title_length, word_embedding_dim]
        content_w = self.dropout(self.word_embedding(content_text)).view([batch_news_num, self.max_content_length,
                                                                          self.word_embedding_dim])  # [batch_size * news_num, max_content_length, word_embedding_dim]
        # PE
        title_w = self.PE(title_w)
        content_w = self.PE(content_w)
        # 2. CNN encoding
        title_c = self.dropout_(self.title_conv(title_w.permute(0, 2, 1)).permute(0, 2,
                                                                                  1))  # [batch_size * news_num, max_title_length, cnn_kernel_num]
        content_c = self.dropout_(self.content_conv(content_w.permute(0, 2, 1)).permute(0, 2,
                                                                                        1))  # [batch_size * news_num, max_content_length, cnn_kernel_num]
        # 3. attention layer
        title_representation = self.title_attention(title_c).view(
            [batch_size, news_num, self.cnn_kernel_num])  # [batch_size, news_num, cnn_kernel_num]
        content_representation = self.content_attention(content_c).view(
            [batch_size, news_num, self.cnn_kernel_num])  # [batch_size, news_num, cnn_kernel_num]
        # 4. category and subCategory encoding
        category_representation = F.relu(self.category_affine(self.category_embedding(category)),
                                         inplace=True)  # [batch_size, news_num, cnn_kernel_num]
        subCategory_representation = F.relu(self.subCategory_affine(self.subCategory_embedding(subCategory)),
                                            inplace=True)  # [batch_size, news_num, cnn_kernel_num]
        # 5. multi-view attention
        # feature = torch.stack(
        #     [title_representation, content_representation, category_representation, subCategory_representation],
        #     dim=2)  # [batch_size, news_num, 4, cnn_kernel_num]
        # alpha = F.softmax(self.affine2(torch.tanh(self.affine1(feature))), dim=2)  # [batch_size, news_num, 4, 1]
        # news_representation = (feature * alpha).sum(dim=2, keepdim=False)  # [batch_size, news_num, cnn_kernel_num]
        # [batch_size, news_num, 4, cnn_kernel_num]
        news = torch.stack([title_representation, content_representation, category_representation, subCategory_representation], dim=2)
        #
        news_representation = self.news_attention(news.view(batch_news_num, -1, self.news_embedding_dim)).view(batch_size, news_num, self.news_embedding_dim)
        # [batch_size, news_num, news_embedding_dim]
        return news_representation
import math
import pickle

import numpy as np

from config import Config
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence

from models.modules.fastformer import FastformerEncoder
from models.modules.layers import MultiHeadAttention, Attention, ScaledDotProduct_CandidateAttention, \
    CandidateAttention, GCN, MultiHeadSelfAttention, AdditiveAttention, Conv1D
from models.modules.newsEncoders import NewsEncoder, HDC
from torch_scatter import scatter_sum, scatter_softmax # need to be installed by following `https://pytorch-scatter.readthedocs.io/en/latest`


class UserEncoder(nn.Module):
    def __init__(self, news_encoder: NewsEncoder, config: Config):
        super(UserEncoder, self).__init__()
        self.news_embedding_dim = news_encoder.news_embedding_dim
        self.news_encoder = news_encoder
        self.device = torch.device('cuda')
        self.auxiliary_loss = None

    # Input
    # user_title_text               : [batch_size, max_history_num, max_title_length]
    # user_title_mask               : [batch_size, max_history_num, max_title_length]
    # user_title_entity             : [batch_size, max_history_num, max_title_length]
    # user_content_text             : [batch_size, max_history_num, max_content_length]
    # user_content_mask             : [batch_size, max_history_num, max_content_length]
    # user_content_entity           : [batch_size, max_history_num, max_content_length]
    # user_category                 : [batch_size, max_history_num]
    # user_subCategory              : [batch_size, max_history_num]
    # user_history_mask             : [batch_size, max_history_num]
    # user_history_graph            : [batch_size, max_history_num, max_history_num]
    # user_history_category_mask    : [batch_size, category_num]
    # user_history_category_indices : [batch_size, max_history_num]
    # user_embedding                : [batch_size, user_embedding]
    # candidate_news_representation : [batch_size, news_num, news_embedding_dim]
    # Output
    # user_representation           : [batch_size, news_embedding_dim]
    def forward(self, user_title_text, user_title_mask, user_title_entity, user_content_text, user_content_mask, user_content_entity, user_category, user_subCategory, \
                user_history_mask, user_history_graph, user_history_category_mask, user_history_category_indices, user_embedding, candidate_news_representation):
        raise Exception('Function forward must be implemented at sub-class')

########################################################################################################################
# CNE-SUE
class SUE(UserEncoder):
    def __init__(self, news_encoder: NewsEncoder, config: Config):
        super(SUE, self).__init__(news_encoder, config)
        self.attention_dim = max(config.attention_dim, self.news_embedding_dim // 4)
        self.proxy_node_embedding = nn.Parameter(torch.zeros([config.category_num, self.news_embedding_dim]))
        self.gcn = GCN(in_dim=self.news_embedding_dim, out_dim=self.news_embedding_dim, hidden_dim=self.news_embedding_dim, num_layers=config.gcn_layer_num, dropout=config.dropout_rate / 2, residual=not config.no_gcn_residual, layer_norm=config.gcn_layer_norm)
        self.intraCluster_K = nn.Linear(self.news_embedding_dim, self.attention_dim, bias=False)
        self.intraCluster_Q = nn.Linear(self.news_embedding_dim, self.attention_dim, bias=True)
        self.clusterFeatureAffine = nn.Linear(self.news_embedding_dim, self.news_embedding_dim, bias=True)
        self.interClusterAttention = ScaledDotProduct_CandidateAttention(self.news_embedding_dim, self.news_embedding_dim, self.attention_dim)
        self.dropout = nn.Dropout(p=config.dropout_rate, inplace=True)
        self.dropout_ = nn.Dropout(p=config.dropout_rate, inplace=False)
        self.category_num = config.category_num + 1 # extra one category index for padding news
        self.max_history_num = config.max_history_num
        self.attention_scalar = math.sqrt(float(self.attention_dim))

    def initialize(self):
        self.gcn.initialize()
        nn.init.zeros_(self.proxy_node_embedding)
        nn.init.xavier_uniform_(self.intraCluster_K.weight)
        nn.init.xavier_uniform_(self.intraCluster_Q.weight)
        nn.init.zeros_(self.intraCluster_Q.bias)
        nn.init.xavier_uniform_(self.clusterFeatureAffine.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.zeros_(self.clusterFeatureAffine.bias)
        self.interClusterAttention.initialize()

    def forward(self, user_title_text, user_title_mask, user_title_entity, user_content_text, user_content_mask, user_content_entity, user_category, user_subCategory, \
                user_history_mask, user_history_graph, user_history_category_mask, user_history_category_indices, user_embedding, candidate_news_representation):
        batch_size = user_title_text.size(0)
        news_num = candidate_news_representation.size(1)
        batch_news_num = batch_size * news_num
        user_history_category_mask[:, -1] = 1
        # [batch_size, news_num, category_num]
        user_history_category_mask = user_history_category_mask.unsqueeze(dim=1).expand(-1, news_num, -1).contiguous()
        # [batch_size, news_num, max_history_num]
        user_history_category_indices = user_history_category_indices.unsqueeze(dim=1).expand(-1, news_num, -1)
        # [batch_size, max_history_num, news_embedding_dim]
        history_embedding = self.news_encoder(user_title_text, user_title_mask, user_title_entity, \
                                              user_content_text, user_content_mask, user_content_entity, \
                                              user_category, user_subCategory, user_embedding)
        # 1. GCN # [batch_size, max_history_num + category_num, news_embedding_dim]
        history_embedding = torch.cat([history_embedding, self.dropout_(self.proxy_node_embedding.unsqueeze(dim=0).expand(batch_size, -1, -1))], dim=1)
        # [batch_size, max_history_num + category_num, news_embedding_dim]
        gcn_feature = self.gcn(history_embedding, user_history_graph) + history_embedding
        gcn_feature = gcn_feature[:, :self.max_history_num, :]
        #  [batch_size, news_num, max_history_num, news_embedding_dim]
        gcn_feature = gcn_feature.unsqueeze(dim=1).expand(-1, news_num, -1, -1)
        # 2. Intra-cluster attention
        K = self.intraCluster_K(gcn_feature).view([batch_news_num, self.max_history_num, self.attention_dim])
        Q = self.intraCluster_Q(candidate_news_representation).view([batch_news_num, self.attention_dim, 1])
        a = torch.bmm(K, Q).view([batch_size, news_num, self.max_history_num]) / self.attention_scalar
        alpha_intra = scatter_softmax(a, user_history_category_indices, 2).unsqueeze(dim=3)
        # [batch_size, news_num, max_history_num, 1]
        intra_cluster_feature = scatter_sum(alpha_intra * gcn_feature, user_history_category_indices, dim=2, dim_size=self.category_num)
        # perform nonlinear transformation on intra-cluster features  # [batch_size, news_num, category_num, news_embedding_dim]
        intra_cluster_feature = self.dropout(F.relu(self.clusterFeatureAffine(intra_cluster_feature), inplace=True) + intra_cluster_feature)
        # 3. Inter-cluster attention
        inter_cluster_feature = self.interClusterAttention(
            intra_cluster_feature.view([batch_news_num, self.category_num, self.news_embedding_dim]),
            candidate_news_representation.view([batch_news_num, self.news_embedding_dim]),
            mask=user_history_category_mask.view([batch_news_num, self.category_num])
        ).view([batch_size, news_num, self.news_embedding_dim])
        # [batch_size, news_num, news_embedding_dim]
        return inter_cluster_feature

########################################################################################################################
# LSTUR
class LSTUR(UserEncoder):
    def __init__(self, news_encoder: NewsEncoder, config: Config):
        super(LSTUR, self).__init__(news_encoder, config)
        self.masking_probability = 1.0 - config.long_term_masking_probability
        self.gru = nn.GRU(self.news_embedding_dim, self.news_embedding_dim, batch_first=True)

    def initialize(self):
        for parameter in self.gru.parameters():
            if len(parameter.size()) >= 2:
                nn.init.orthogonal_(parameter.data)
            else:
                nn.init.zeros_(parameter.data)

    def forward(self, user_title_text, user_title_mask, user_title_entity, user_content_text, user_content_mask, user_content_entity, user_category, user_subCategory, \
                user_history_mask, user_history_graph, user_history_category_mask, user_history_category_indices, user_embedding, candidate_news_representation):
        batch_size = user_title_text.size(0)
        news_num = candidate_news_representation.size(1)
        user_history_num = user_history_mask.sum(dim=1, keepdim=False).long()
        history_embedding = self.news_encoder(user_title_text, user_title_mask, user_title_entity, \
                                              user_content_text, user_content_mask, user_content_entity, \
                                              user_category, user_subCategory, user_embedding)
        sorted_user_history_num, sorted_indices = torch.sort(user_history_num, descending=True)
        _, desorted_indices = torch.sort(sorted_indices, descending=False)
        nonzero_indices = sorted_user_history_num.nonzero(as_tuple=False).squeeze(dim=1)
        if nonzero_indices.size(0) == 0:
            return user_embedding
        index = nonzero_indices[-1]
        if index + 1 == batch_size:
            sorted_user_embedding = user_embedding.index_select(0, sorted_indices)
            if self.training and self.masking_probability != 1.0:
                sorted_user_embedding *= torch.bernoulli(torch.empty([batch_size, 1], device=self.device).fill_(self.masking_probability))
            sorted_history_embedding = history_embedding.index_select(0, sorted_indices)
            packed_sorted_history_embedding = pack_padded_sequence(sorted_history_embedding, sorted_user_history_num.cpu(), batch_first=True)
            _, h = self.gru(packed_sorted_history_embedding, sorted_user_embedding.unsqueeze(dim=0))
            user_representation = h.squeeze(dim=0).index_select(0, desorted_indices)
        else:
            non_empty_indices = sorted_indices[:index+1]
            empty_indices = sorted_indices[index+1:]
            sorted_user_embedding = user_embedding.index_select(0, non_empty_indices)
            if self.training and self.masking_probability != 1.0:
                sorted_user_embedding *= torch.bernoulli(torch.empty([index + 1, 1], device=self.device).fill_(self.masking_probability))
            sorted_history_embedding = history_embedding.index_select(0, non_empty_indices)
            # [batch_size, max_history_num, news_embedding_dim]
            packed_sorted_history_embedding = pack_padded_sequence(sorted_history_embedding, sorted_user_history_num[:index+1].cpu(), batch_first=True)
            _, h = self.gru(packed_sorted_history_embedding, sorted_user_embedding.unsqueeze(dim=0))
            # [batch_size, news_embedding_dim]
            user_representation = torch.cat([h.squeeze(dim=0), user_embedding.index_select(0, empty_indices)], dim=0).index_select(0, desorted_indices)
        # user_representation = user_representation.unsqueeze(dim=1).expand(-1, news_num, -1)
        return user_representation

########################################################################################################################
# NRMS
class MHSA(UserEncoder):
    def __init__(self, news_encoder: NewsEncoder, config: Config):
        super(MHSA, self).__init__(news_encoder, config)
        self.multiheadAttention = MultiHeadAttention(config.head_num, self.news_embedding_dim, config.max_history_num, config.max_history_num, config.head_dim, config.head_dim)
        self.affine = nn.Linear(config.head_num*config.head_dim, self.news_embedding_dim, bias=True)
        self.attention = Attention(self.news_embedding_dim, config.attention_dim)

    def initialize(self):
        self.multiheadAttention.initialize()
        nn.init.xavier_uniform_(self.affine.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.zeros_(self.affine.bias)
        self.attention.initialize()

    def forward(self, user_title_text, user_title_mask, user_title_entity, user_content_text, user_content_mask, user_content_entity, user_category, user_subCategory, \
                user_history_mask, user_history_graph, user_history_category_mask, user_history_category_indices, user_embedding, candidate_news_representation):
        news_num = candidate_news_representation.size(1)
        # [batch_size, max_history_num, news_embedding_dim]
        history_embedding = self.news_encoder(user_title_text, user_title_mask, user_title_entity, \
                                              user_content_text, user_content_mask, user_content_entity, \
                                              user_category, user_subCategory, user_embedding)
        # [batch_size, max_history_num, head_num * head_dim]
        h = self.multiheadAttention(history_embedding, history_embedding, history_embedding, user_history_mask)
        # [batch_size, max_history_num, news_embedding_dim]
        h = F.relu(F.dropout(self.affine(h), training=self.training, inplace=True), inplace=True)
        # [batch_size, news_num, news_embedding_dim]
        user_representation = self.attention(h)
        return user_representation


########################################################################################################################
# NAML
class ATT(UserEncoder):
    def __init__(self, news_encoder: NewsEncoder, config: Config):
        super(ATT, self).__init__(news_encoder, config)
        self.attention = Attention(self.news_embedding_dim, config.attention_dim)

    def initialize(self):
        self.attention.initialize()

    def forward(self, user_title_text, user_title_mask, user_title_entity, user_content_text, user_content_mask, user_content_entity, user_category, user_subCategory, \
                user_history_mask, user_history_graph, user_history_category_mask, user_history_category_indices, user_embedding, candidate_news_representation):
        # [batch_size, max_history_num, news_embedding_dim]
        history_embedding = self.news_encoder(user_title_text, user_title_mask, user_title_entity, \
                                              user_content_text, user_content_mask, user_content_entity, \
                                              user_category, user_subCategory, user_embedding)
        user_representation = self.attention(history_embedding)
        return user_representation

########################################################################################################################
# DKN
class CATT(UserEncoder):
    def __init__(self, news_encoder: NewsEncoder, config: Config):
        super(CATT, self).__init__(news_encoder, config)
        self.affine1 = nn.Linear(self.news_embedding_dim * 2, config.attention_dim, bias=True)
        self.affine2 = nn.Linear(config.attention_dim, 1, bias=True)
        self.max_history_num = config.max_history_num

    def initialize(self):
        nn.init.xavier_uniform_(self.affine1.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.zeros_(self.affine1.bias)
        nn.init.xavier_uniform_(self.affine2.weight)
        nn.init.zeros_(self.affine2.bias)

    def forward(self, user_title_text, user_title_mask, user_title_entity, user_content_text, user_content_mask, user_content_entity, user_category, user_subCategory, \
                user_history_mask, user_history_graph, user_history_category_mask, user_history_category_indices, user_embedding, candidate_news_representation):
        news_num = candidate_news_representation.size(1)
        # [batch_size, max_history_num, news_embedding_dim]
        history_embedding = self.news_encoder(user_title_text, user_title_mask, user_title_entity, \
                                              user_content_text, user_content_mask, user_content_entity, \
                                              user_category, user_subCategory, user_embedding)
        # [batch_size, news_num, max_history_num]
        user_history_mask = user_history_mask.unsqueeze(dim=1).expand(-1, news_num, -1)
        # [batch_size, news_num, max_history_num, news_embedding_dim]
        candidate_news_representation = candidate_news_representation.unsqueeze(dim=2).expand(-1, -1, self.max_history_num, -1)
        # [batch_size, news_num, max_history_num, news_embedding_dim]
        history_embedding = history_embedding.unsqueeze(dim=1).expand(-1, news_num, -1, -1)
        # [batch_size, news_num, max_history_num, news_embedding_dim * 2]
        concat_embeddings = torch.cat([candidate_news_representation, history_embedding], dim=3)
        # [batch_size, news_num, max_history_num, attention_dim]
        hidden = F.relu(self.affine1(concat_embeddings), inplace=True)
        # [batch_size, news_num, max_history_num]
        a = self.affine2(hidden).squeeze(dim=3)
        # [batch_size, news_num, max_history_num]
        alpha = F.softmax(a.masked_fill(user_history_mask == 0, -1e9), dim=2)
        # [batch_size, news_num, news_embedding_dim]
        user_representation = (alpha.unsqueeze(dim=3) * history_embedding).sum(dim=2, keepdim=False)
        return user_representation

########################################################################################################################
# FIM
class FIM(UserEncoder):
    def __init__(self, news_encoder: NewsEncoder, config: Config):
        super(FIM, self).__init__(news_encoder, config)
        assert type(self.news_encoder) == HDC, 'For FIM, the news encoder must be HDC'
        self.HDC_sequence_length = news_encoder.HDC_sequence_length
        self.max_history_num = config.max_history_num
        self.scalar = math.sqrt(float(config.HDC_filter_num))
        self.conv_3D_a = nn.Conv3d(in_channels=4, out_channels=config.conv3D_filter_num_first, kernel_size=config.conv3D_kernel_size_first)
        self.conv_3D_b = nn.Conv3d(in_channels=config.conv3D_filter_num_first, out_channels=config.conv3D_filter_num_second, kernel_size=config.conv3D_kernel_size_second)
        self.maxpool_3D = torch.nn.MaxPool3d(kernel_size=config.maxpooling3D_size, stride=config.maxpooling3D_stride)

    def initialize(self):
        pass

    def forward(self, user_title_text, user_title_mask, user_title_entity, user_content_text, user_content_mask, user_content_entity, user_category, user_subCategory, \
                user_history_mask, user_history_graph, user_history_category_mask, user_history_category_indices, user_embedding, candidate_news_representation):
        candidate_news_d0, candidate_news_dL = candidate_news_representation
        history_embedding_d0, history_embedding_dL = self.news_encoder(user_title_text, user_title_mask, user_title_entity, \
                                                                       user_content_text, user_content_mask, user_content_entity, \
                                                                       user_category, user_subCategory, user_embedding)
        batch_size = candidate_news_d0.size(0)
        news_num = candidate_news_d0.size(1)
        batch_news_num = batch_size * news_num
        # 1. compute 3D matching images # [batch_size, news_num, 1, HDC_sequence_length, HDC_filter_num]
        candidate_news_d0 = candidate_news_d0.unsqueeze(dim=2).permute(0, 1, 2, 4 ,3)
        # [batch_size, news_num, 1, 3, HDC_sequence_length, HDC_filter_num]
        candidate_news_dL = candidate_news_dL.unsqueeze(dim=2).permute(0, 1, 2, 3 ,5, 4)
        # [batch_size, 1, max_history_num, HDC_filter_num, HDC_sequence_length]
        history_embedding_d0 = history_embedding_d0.unsqueeze(dim=1)
        # [batch_size, 1, max_history_num, 3, HDC_filter_num, HDC_sequence_length]
        history_embedding_dL = history_embedding_dL.unsqueeze(dim=1)
        # [batch_size, news_num, max_history_num, HDC_sequence_length, HDC_sequence_length]
        matching_images_d0 = torch.matmul(candidate_news_d0, history_embedding_d0) / self.scalar
        # [batch_size, news_num, max_history_num, 3, HDC_sequence_length, HDC_sequence_length]
        matching_images_dL = torch.matmul(candidate_news_dL, history_embedding_dL) / self.scalar
        # [batch_size, news_num, 4, max_history_num, HDC_sequence_length, HDC_sequence_length]
        matching_images = torch.cat([matching_images_d0.unsqueeze(dim=3), matching_images_dL], dim=3).permute(0, 1, 3, 2, 4, 5)
        # [batch_size * news_num, 4, max_history_num, HDC_sequence_length, HDC_sequence_length]
        matching_images = matching_images.view(batch_news_num, 4, self.max_history_num, self.HDC_sequence_length, self.HDC_sequence_length)
        # 2. 3D convolution layers
        Q1 = F.elu(self.conv_3D_a(matching_images), inplace=True)
        Q1 = self.maxpool_3D(Q1)
        Q2 = F.elu(self.conv_3D_b(Q1), inplace=True)
        Q2 = self.maxpool_3D(Q2)
        # [batch_size * news_num, feature_size]
        salient_signals = Q2.view([batch_size, news_num, -1])
        return salient_signals

########################################################################################################################
# NPA
class PUE(UserEncoder):
    def __init__(self, news_encoder: NewsEncoder, config: Config):
        super(PUE, self).__init__(news_encoder, config)
        self.dense = nn.Linear(config.user_embedding_dim, config.personalized_embedding_dim, bias=True)
        self.personalizedAttention = CandidateAttention(self.news_embedding_dim, config.personalized_embedding_dim, config.attention_dim)

    def initialize(self):
        nn.init.xavier_uniform_(self.dense.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.zeros_(self.dense.bias)
        self.personalizedAttention.initialize()

    def forward(self, user_title_text, user_title_mask, user_title_entity, user_content_text, user_content_mask, user_content_entity, user_category, user_subCategory, \
                user_history_mask, user_history_graph, user_history_category_mask, user_history_category_indices, user_embedding, candidate_news_representation):
        news_num = candidate_news_representation.size(1)
        # [batch_size, max_history_num, news_embedding_dim]
        history_embedding = self.news_encoder(user_title_text, user_title_mask, user_title_entity, \
                                              user_content_text, user_content_mask, user_content_entity, \
                                              user_category, user_subCategory, user_embedding)
        # [batch_size, personalized_embedding_dim]
        q_d = F.relu(self.dense(user_embedding), inplace=True)
        user_representation = self.personalizedAttention(history_embedding, q_d, user_history_mask)
        return user_representation

########################################################################################################################
# TANR
class TANR(UserEncoder):
    def __init__(self, news_encoder: NewsEncoder, config: Config):
        super(TANR, self).__init__(news_encoder, config)
        self.attention = Attention(self.news_embedding_dim, config.attention_dim)
        self.config = config

    def initialize(self):
        self.attention.initialize()

    def forward(self, user_title_text, user_title_mask, user_title_entity, user_content_text, user_content_mask, user_content_entity, user_category, user_subCategory, \
                user_history_mask, user_history_graph, user_history_category_mask, user_history_category_indices, user_embedding, candidate_news_representation):

        history_embedding = self.news_encoder(user_title_text, user_title_mask, user_title_entity, \
                                              user_content_text, user_content_mask, user_content_entity, \
                                              user_category, user_subCategory, user_embedding)
        user_representation = self.attention(history_embedding)
        # [batch, news_embedding_dim]
        return user_representation


########################################################################################################################
# MINS
class MINS_UE(UserEncoder):
    def __init__(self, news_encoder: NewsEncoder, config: Config):
        # torch.backends.cudnn.enabled = False
        super(MINS_UE, self).__init__(news_encoder, config)
        assert config.word_embedding_dim % config.layers == 0
        self.config = config
        self.news_encoder = news_encoder
        self.multihead_self_attention = MultiHeadSelfAttention(
            config.word_embedding_dim, config.layers)
        self.additive_attention = AdditiveAttention(config.query_vector_dim,
                                                    config.word_embedding_dim)
        self.gru = nn.GRU(
            int(config.word_embedding_dim / config.layers),
            int(config.word_embedding_dim / config.layers))
        self.multi_channel_gru = nn.ModuleList([self.gru for _ in range(self.config.layers)])

    def forward(self, user_title_text, user_title_mask, user_title_entity, user_content_text, user_content_mask, user_content_entity, user_category, user_subCategory, \
                                                user_history_mask, user_history_graph, user_history_category_mask, user_history_category_indices, user_embedding, news_representation):
        # batch
        clicked_news_length = torch.tensor([torch.where(user_history_mask[i])[0].shape[0] for i in range(user_history_mask.shape[0])]).to(user_history_mask.device)
        clicked_news_length[clicked_news_length == 0] = 1

        history_embedding = self.news_encoder(user_title_text, user_title_mask, user_title_entity, \
                                              user_content_text, user_content_mask, user_content_entity, \
                                              user_category, user_subCategory, user_embedding)
        # batch_size, num_clicked_news_a_user, word_embedding_dim
        multihead_user_vector = self.multihead_self_attention(history_embedding)
        # batch_size, num_clicked_news_a_user, word_embedding_dim
        one_channel = torch.chunk(multihead_user_vector , self.config.layers, dim=2)
        channels = []
        # batch_size, num_clicked_news_a_user, word_embedding_dim/layers
        for n, g in zip(range(self.config.layers), self.multi_channel_gru):
            packed_clicked_news_vector = pack_padded_sequence(
                input=one_channel[n],
                lengths=clicked_news_length.cpu(),
                batch_first=True,
                enforce_sorted=False)
            _, last_hidden = g(packed_clicked_news_vector)
            # 1,batch,config.word_embedding_dim / config.layers
            channels.append(last_hidden)
        # batch, 1, word_embedding_dim
        multi_channel_vector = torch.cat(channels, dim=2).transpose(0, 1)
        # batch, word_embedding_dim
        final_user_vector = self.additive_attention(multi_channel_vector)
        # batch, word_embedding_dim
        return final_user_vector


########################################################################################################################
# CenNewsRec
class CenNewsRec(UserEncoder):
    def __init__(self, news_encoder: NewsEncoder, config: Config):
        super(CenNewsRec, self).__init__(news_encoder, config)
        self.config = config
        self.num_recent_news = self.config.num_recent_news

        self.multihead_attention = nn.MultiheadAttention(
            embed_dim=self.config.num_filters, num_heads=self.config.num_heads
        )
        self.additive_attention = AdditiveAttention(candidate_vector_dim=self.config.num_filters, query_vector_dim=self.config.query_dim)
        self.gru = nn.GRU(input_size=self.config.num_filters, hidden_size=self.config.gru_hidden_dim, batch_first=True)
        self.final_additive_attention = AdditiveAttention(
            candidate_vector_dim=self.config.gru_hidden_dim, query_vector_dim=self.config.query_dim
        )
        self.dropout = nn.Dropout(p=self.config.dropout_probability)

    def forward(self, user_title_text, user_title_mask, user_title_entity, user_content_text, user_content_mask, user_content_entity, user_category, user_subCategory, \
                                                user_history_mask, user_history_graph, user_history_category_mask, user_history_category_indices, user_embedding, news_representation):
        # long-term user representation
        batch_size = user_title_text.size(0)
        news_num = user_title_text.size(1)
        # [batch_size, news_num, num_filters]
        history_embedding = self.news_encoder(user_title_text, user_title_mask, user_title_entity, \
                                              user_content_text, user_content_mask, user_content_entity, \
                                              user_category, user_subCategory, user_embedding)
        # [num_clicked_news, batch_size, num_filters]
        # history_embedding = history_embedding.unsqueeze(dim=1).expand(-1, news_num, -1)
        # batch_size, num_clicked_news, num_filters
        longterm_user_vector, _ = self.multihead_attention(
            history_embedding, history_embedding, history_embedding
        )
        longterm_user_vector = self.dropout(longterm_user_vector)

        # batch_size, num_filters
        longterm_user_vector = self.additive_attention(longterm_user_vector)

        # short-term user representation
        # batch_size, num_recent_news, num_filters
        recent_hist_news = history_embedding[:, -self.num_recent_news :, :]

        # 1, batch_size, gru_hidden_dim
        _, hidden = self.gru(recent_hist_news)

        # batch_size, gru_hidden_dim
        shortterm_user_vector = hidden.squeeze(dim=0)

        # aggregated user representation
        # batch_size, 2, gru_hidden_dim
        user_vector = torch.stack([shortterm_user_vector, longterm_user_vector], dim=1)

        # batch_size, gru_hidden_dim
        user_vector = self.final_additive_attention(user_vector)

        return user_vector


########################################################################################################################
# IPNR
class IPNR(UserEncoder):
    def __init__(self, news_encoder: NewsEncoder, config: Config):
        super(IPNR, self).__init__(news_encoder, config)
        self.multiheadAttention = MultiHeadAttention(config.head_num, self.news_embedding_dim, config.max_history_num, config.max_history_num, config.head_dim, config.head_dim)
        self.affine = nn.Linear(in_features=config.head_num*config.head_dim, out_features=self.news_embedding_dim, bias=True)
        self.attention = Attention(self.news_embedding_dim, config.attention_dim)
        self.attention_decoder = Attention(config.word_embedding_dim, config.attention_dim)
        self.attention_dim = max(config.attention_dim, self.news_embedding_dim // 4)
        self.gcn = GCN(in_dim=config.word_embedding_dim, out_dim=config.word_embedding_dim,
                       hidden_dim=config.word_embedding_dim, num_layers=config.gcn_layer_num,
                       dropout=config.dropout_rate / 2, residual=not config.no_gcn_residual,
                       layer_norm=config.gcn_layer_norm)
        self.dropout = nn.Dropout(p=config.dropout_rate, inplace=True)
        self.dropout_ = nn.Dropout(p=config.dropout_rate, inplace=False)
        self.concept_num_per_news = config.concept_num_per_news  # extra one category index for padding news
        self.max_history_num = config.max_history_num
        self.num_concepts = config.num_concepts
        self.title_length = config.max_title_length
        self.content_length = config.max_abstract_length
        self.word_embedding_dim = config.word_embedding_dim

        # gate net
        self.gate_layer = nn.Sequential(
            nn.Linear(self.news_embedding_dim + config.word_embedding_dim, self.news_embedding_dim, bias=False),
            nn.Sigmoid()
        )
        self.fuse_layer1 = nn.Sequential(
            nn.Linear(config.word_embedding_dim, self.news_embedding_dim, bias=False),
            nn.Tanh()
        )
        # intention parameter
        self.transform_matrix = nn.Parameter(
            torch.empty(config.word_embedding_dim,
                        config.word_embedding_dim).uniform_(-0.1, 0.1))
        # intention encoder
        self.dense = nn.Linear(in_features=config.batch_size * config.concept_num_per_news * config.max_history_num, out_features=config.batch_size * config.max_history_num * config.num_concepts,
                                bias=True)
        # CNN
        self.conv = Conv1D(config.cnn_method, self.concept_num_per_news, self.num_concepts, config.cnn_window_size)
        #
        self.FusionAttention = ScaledDotProduct_CandidateAttention(self.news_embedding_dim, self.news_embedding_dim, self.attention_dim)
        self.fastformer = FastformerEncoder(config)

        self.pretrained_concept_embedding = torch.from_numpy(
            np.load(
                '%s/all_concept_word_embedding.npy' %config.DATASET_ROOT)).float().to(
            device=self.device)  # build by 03_generate_concept_embedding.py
        with open('cache/IPNR/word_embedding-' + str(config.word_threshold) + '-' + str(
                config.word_embedding_dim) + '-' + config.tokenizer + '-' + str(config.max_title_length) + '-' + str(
            config.max_abstract_length) + '-' + config.dataset_size + '.pkl', 'rb') as word_embedding_f:
            self.word_embedding = nn.Embedding.from_pretrained(pickle.load(word_embedding_f))

    def initialize(self):
        self.multiheadAttention.initialize()
        nn.init.xavier_uniform_(self.affine.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.zeros_(self.affine.bias)
        self.attention.initialize()
        self.attention_decoder.initialize()
        nn.init.xavier_uniform_(self.dense.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.zeros_(self.dense.bias)
        self.FusionAttention.initialize()

    def forward(self, user_title_text, user_title_mask, user_title_entity, user_content_text, user_content_mask, user_content_entity, user_category, user_subCategory, \
                user_history_mask, user_history_graph, user_concept_text, user_concept_mask, user_embedding, candidate_news_representation):
        batch_size = user_title_text.size(0)
        news_num = candidate_news_representation.size(1)
        batch_news_num = batch_size * news_num
        # user reading preference # [batch_size, max_history_num, news_embedding_dim]
        history_embedding = self.news_encoder(user_title_text, user_title_mask, user_title_entity, \
                                              user_content_text, user_content_mask, user_content_entity, \
                                              user_category, user_subCategory, user_embedding)
        h = self.fastformer(history_embedding.view(batch_size, -1, self.news_embedding_dim))
        # [batch_size, news_num, news_embedding_dim]
        user_representation = h.unsqueeze(dim=1).repeat(1, news_num, 1)
        # user reading intention # [batch_size, max_history_num, concept_length, word_embedding_dim]
        clicked_concept_emebedding = self.dropout(
            self.word_embedding(user_concept_text)).reshape(batch_size*self.max_history_num, -1, self.word_embedding_dim)
        # batch_size*max_history_num*num_concepts, word_embedding_dim
        c = self.dropout_(self.conv(clicked_concept_emebedding))
        # batch_size*max_history_num*concept_length, word_embedding_dim
        temp = torch.matmul(c.reshape(-1, self.word_embedding_dim), self.transform_matrix)
        # batch_size*num_clicked_news_a_user*concept_length, k
        t = torch.matmul(temp, self.pretrained_concept_embedding.transpose(0, 1))
        # batch_size*num_clicked_news_a_user*num_concept, k
        concept_weight = F.softmax(t, dim=1)
        # batch_size*num_clicked_news_a_user*num_concept, word_embedding_dim
        personalized_concept_vector = torch.matmul(concept_weight,
                                                   self.pretrained_concept_embedding).reshape(batch_size, -1, self.word_embedding_dim)
        # [batch_size, max_history_num*num_concepts, news_embedding_dim]
        gcn_feature = self.gcn(personalized_concept_vector,
                               user_history_graph)
        # [batch_size, max_history_num, news_embedding_dim]
        gcn_feature = gcn_feature[:, :self.max_history_num, :]
        # [batch_size, news_num, max_history_num, news_embedding_dim]
        gcn_feature = gcn_feature.unsqueeze(dim=1).expand(-1, news_num, -1,
                                                          -1)
        # [batch_size*news_num=5, news_embedding_dim]
        user_vector = self.attention_decoder(gcn_feature.reshape(batch_news_num, -1, self.word_embedding_dim)).reshape(batch_size, -1, self.word_embedding_dim)
        # [batch_size, news_num=5, news_embedding_dim+word_embedding_dim]
        all_perferences = torch.cat([user_representation, user_vector], dim=-1)
        # [batch_size, news_num=5, news_embedding_dim]
        gate = self.gate_layer(all_perferences)
        # [batch_size, 5, news_embedding_dim]
        final_user_representation = gate * self.fuse_layer1(user_vector) + (1.0 - gate) * user_representation
        #
        inter_cluster_feature = self.FusionAttention(
            final_user_representation.view([batch_news_num, 1, self.news_embedding_dim]),
            candidate_news_representation.view([batch_news_num, self.news_embedding_dim])).view([batch_size, news_num, self.news_embedding_dim])
        # [batch_size, 5, news_embedding_dim]
        return inter_cluster_feature

from config import Config
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence
from models.modules.layers import Attention, ScaledDotProduct_CandidateAttention, GCN
from torch_scatter import scatter_sum, scatter_softmax
import math
import pickle
import numpy as np

from models.modules.newsEncoders import NewsEncoder


class CNRCLNewsEncoder(NewsEncoder):
    def __init__(self, config: Config):
        super(CNRCLNewsEncoder, self).__init__(config)
        self.max_title_length = config.max_title_length
        self.max_content_length = config.max_abstract_length
        self.word_embedding_dim = config.word_embedding_dim
        self.hidden_dim = config.hidden_dim
        # 4 * hidden_dim (title/content * bidirectional) + category + subcategory
        self.news_embedding_dim = config.hidden_dim * 4 + config.category_embedding_dim + config.subCategory_embedding_dim

        self.category_embedding_dim = config.category_embedding_dim
        self.subCategory_embedding_dim = config.subCategory_embedding_dim

        self.word_embedding = nn.Embedding(num_embeddings=config.vocabulary_size, embedding_dim=self.word_embedding_dim)
        # Load pre-trained word embeddings
        if config.dataset_name == 'MIND':
            with open('cache/word_embedding-' + str(config.word_threshold) + '-' + str(
                    config.word_embedding_dim) + '-' + config.tokenizer + '-' + str(
                config.max_title_length) + '-' + str(
                config.max_abstract_length) + '-' + config.dataset_size + '.pkl',
                      'rb') as word_embedding_f:
                self.word_embedding.weight.data.copy_(pickle.load(word_embedding_f))
        elif config.dataset_name in {'ebnerd', 'Adressa'}:
            with open('cache/%s/word_embedding-' % config.dataset_name.lower() + str(config.word_threshold) + '-' + str(
                    config.word_embedding_dim) + '-' + config.tokenizer + '-' + str(
                    config.max_title_length) + '-' + str(config.max_abstract_length) + '-' + config.dataset_size + '.pkl',
                      'rb') as word_embedding_f:
                self.word_embedding.weight.data.copy_(pickle.load(word_embedding_f))
        
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
        
        self.dropout = nn.Dropout(p=config.dropout_rate, inplace=False)
        self.auxiliary_loss = None

    def initialize(self):
        nn.init.uniform_(self.category_embedding.weight, -0.1, 0.1)
        nn.init.uniform_(self.subCategory_embedding.weight, -0.1, 0.1)
        nn.init.zeros_(self.subCategory_embedding.weight[0])
        
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
        category_representation = self.category_embedding(category)           # [batch_size, news_num, category_embedding_dim]
        subCategory_representation = self.subCategory_embedding(subCategory)
        
        batch_size = title_text.size(0)
        news_num = title_text.size(1)
        batch_news_num = batch_size * news_num
        
        title_mask = title_mask.view([batch_news_num, self.max_title_length])                                                                              # [batch_size * news_num, max_title_length]
        content_mask = content_mask.view([batch_news_num, self.max_content_length])                                                                        # [batch_size * news_num, max_content_length]
        
        # To avoid empty input of LSTM (ensure at least length 1)
        # Note: In NewsTorch preprocessing, masks might be floats. LSTM needs lengths.
        # title_mask is float in Ebnerd Dataset.
        title_length = title_mask.sum(dim=1).long()
        content_length = content_mask.sum(dim=1).long()
        
        # clamp to at least 1 to prevent pack_padded_sequence error if empty
        title_length = torch.clamp(title_length, min=1)
        content_length = torch.clamp(content_length, min=1)

        sorted_title_length, sorted_title_indices = torch.sort(title_length, descending=True)                                                              # [batch_size * news_num]
        _, desorted_title_indices = torch.sort(sorted_title_indices, descending=False)                                                                     # [batch_size * news_num]
        sorted_content_length, sorted_content_indices = torch.sort(content_length, descending=True)                                                        # [batch_size * news_num]
        _, desorted_content_indices = torch.sort(sorted_content_indices, descending=False)                                                                 # [batch_size * news_num]
        
        # 1. word embedding
        title = self.dropout(self.word_embedding(title_text)).view([batch_news_num, self.max_title_length, self.word_embedding_dim])                       # [batch_size * news_num, max_title_length, word_embedding_dim]
        content = self.dropout(self.word_embedding(content_text)).view([batch_news_num, self.max_content_length, self.word_embedding_dim])                 # [batch_size * news_num, max_content_length, word_embedding_dim]
        
        sorted_title = pack_padded_sequence(title.index_select(0, sorted_title_indices), sorted_title_length.cpu(), batch_first=True)                      # [batch_size * news_num, max_title_length, word_embedding_dim]
        sorted_content = pack_padded_sequence(content.index_select(0, sorted_content_indices), sorted_content_length.cpu(), batch_first=True)              # [batch_size * news_num, max_content_length, word_embedding_dim]
        
        # 2. selective LSTM encoding
        sorted_title_h, (sorted_title_h_n, sorted_title_c_n) = self.title_lstm(sorted_title)
        sorted_content_h, (sorted_content_h_n, sorted_content_c_n) = self.content_lstm(sorted_content)
        
        sorted_title_m = torch.cat([sorted_title_c_n[0], sorted_title_c_n[1]], dim=1)  # [batch_size * news_num, hidden_dim * 2]
        sorted_content_m = torch.cat([sorted_content_c_n[0], sorted_content_c_n[1]], dim=1)  # [batch_size * news_num, hidden_dim * 2]

        sorted_title_h, _ = pad_packed_sequence(sorted_title_h, batch_first=True, total_length=self.max_title_length)                                      # [batch_size * news_num, max_title_length, hidden_dim * 2]
        sorted_content_h, _ = pad_packed_sequence(sorted_content_h, batch_first=True, total_length=self.max_content_length)                                # [batch_size * news_num, max_content_length, hidden_dim * 2]

        sorted_title_gate = torch.sigmoid(self.title_H(sorted_title_h) + self.title_M(sorted_content_m).unsqueeze(dim=1))                                  # [batch_size * news_num, max_title_length, hidden_dim * 2]
        sorted_content_gate = torch.sigmoid(self.content_H(sorted_content_h) + self.content_M(sorted_title_m).unsqueeze(dim=1))                            # [batch_size * news_num, max_content_length, hidden_dim * 2]

        title_h = (sorted_title_h * sorted_title_gate).index_select(0, desorted_title_indices)                                                             # [batch_size * news_num, max_title_length, hidden_dim * 2]
        content_h = (sorted_content_h * sorted_content_gate).index_select(0, desorted_content_indices)
                                                          # [batch_size * news_num, max_content_length, hidden_dim * 2]
        # 3. self-attention
        title_self = self.title_self_attention(title_h, title_mask)                                                                                        # [batch_size * news_num, hidden_dim * 2]
        content_self = self.content_self_attention(content_h, content_mask)                                                                                # [batch_size * news_num, hidden_dim * 2]
        # 4. cross-attention
        title_cross = self.title_cross_attention(title_h, content_self, title_mask)                                                                        # [batch_size * news_num, hidden_dim * 2]
        content_cross = self.content_cross_attention(content_h, title_self, content_mask)                                                                  # [batch_size * news_num, hidden_dim * 2]

        news_representation_0 = torch.cat([title_self + title_cross, content_self + content_cross], dim=1).view([batch_size, news_num, self.hidden_dim * 4])
        news_representation = torch.cat([news_representation_0, self.dropout(category_representation), self.dropout(subCategory_representation)], dim=2)
        return news_representation


class CNRCLUserEncoder(nn.Module):
    def __init__(self, news_encoder: CNRCLNewsEncoder, config: Config):
        super(CNRCLUserEncoder, self).__init__()
        self.news_embedding_dim = news_encoder.news_embedding_dim
        self.news_encoder = news_encoder
        self.hidden_dim = config.hidden_dim
        self.attention_dim = max(config.attention_dim, self.news_embedding_dim // 4)
        self.proxy_node_embedding = nn.Parameter(torch.zeros([config.category_num, self.news_embedding_dim]))
        
        no_gcn_residual = getattr(config, 'no_gcn_residual', False)
        self.gcn = GCN(in_dim=self.news_embedding_dim, out_dim=self.news_embedding_dim, hidden_dim=self.news_embedding_dim, num_layers=config.gcn_layer_num, dropout=config.dropout_rate / 2, residual=not no_gcn_residual, layer_norm=False)
        
        self.intraCluster_K = nn.Linear(self.news_embedding_dim, self.attention_dim, bias=False)
        self.intraCluster_Q = nn.Linear(self.news_embedding_dim, self.attention_dim, bias=True)
        self.clusterFeatureAffine = nn.Linear(self.news_embedding_dim, self.news_embedding_dim, bias=True)
        self.interClusterAttention = ScaledDotProduct_CandidateAttention(self.news_embedding_dim, self.news_embedding_dim, self.attention_dim)
        self.dropout = nn.Dropout(p=config.dropout_rate, inplace=False)
        self.dropout_ = nn.Dropout(p=config.dropout_rate, inplace=False)
        self.category_num = config.category_num + 1 # extra one category index for padding news
        self.max_history_num = config.max_history_num
        self.attention_scalar = math.sqrt(float(self.attention_dim))
        self.auxiliary_loss = None

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
        
        # user_history_category_mask is [batch_size, category_num + 1]
        # In EBNeRD_corpus_main.py, it's [batch_size, category_num + 1]
        # We need to expand it for news_num as in CNRCL code
        
        user_history_category_mask = user_history_category_mask.clone()
        user_history_category_mask[:, -1] = 1
        user_history_category_mask = user_history_category_mask.unsqueeze(dim=1).expand(-1, news_num, -1).contiguous()                                  # [batch_size, news_num, category_num]
        user_history_category_indices = user_history_category_indices.unsqueeze(dim=1).expand(-1, news_num, -1)  # [batch_size, news_num, max_history_num]

        history_embedding = self.news_encoder(user_title_text, user_title_mask, user_title_entity, \
                                              user_content_text, user_content_mask, user_content_entity, \
                                              user_category, user_subCategory, user_embedding)
        # [batch_size, max_history_num, news_embedding_dim]

        # 1. GCN
        # proxy_node_embedding: [category_num, dim] -> expand to [batch, category_num, dim]
        # history_embedding: [batch, max_history, dim]
        # cat -> [batch, max_history + category_num, dim]
        
        history_embedding = torch.cat([history_embedding, self.dropout_(self.proxy_node_embedding.unsqueeze(dim=0).expand(batch_size, -1, -1))], dim=1) 
        
        gcn_feature = self.gcn(history_embedding, user_history_graph) + history_embedding  # [batch_size, max_history_num + category_num, news_embedding_dim]
        gcn_feature = gcn_feature[:, :self.max_history_num, :]       # [batch_size, max_history_num, news_embedding_dim]
        gcn_feature = gcn_feature.unsqueeze(dim=1).expand(-1, news_num, -1, -1)  # [batch_size, news_num, max_history_num, news_embedding_dim]

        # 2. Intra-cluster attention
        K = self.intraCluster_K(gcn_feature).view([batch_news_num, self.max_history_num, self.attention_dim]) # [batch_size * news_num, max_history_num, attention_dim]
        Q = self.intraCluster_Q(candidate_news_representation).view([batch_news_num, self.attention_dim, 1])   # [batch_size * news_num, attention_dim, 1]
        a = torch.bmm(K, Q).view([batch_size, news_num, self.max_history_num]) / self.attention_scalar      # [batch_size, news_num, max_history_num]
        
        alpha_intra = scatter_softmax(a, user_history_category_indices, dim=2).unsqueeze(dim=3)               # [batch_size, news_num, max_history_num, 1]
        
        intra_cluster_feature = scatter_sum(alpha_intra * gcn_feature, user_history_category_indices, dim=2, dim_size=self.category_num)                # [batch_size, news_num, category_num, news_embedding_dim]
        
        # perform nonlinear transformation on intra-cluster features
        intra_cluster_feature = self.dropout(F.relu(self.clusterFeatureAffine(intra_cluster_feature), inplace=True) + intra_cluster_feature)            # [batch_size, news_num, category_num, news_embedding_dim]
        
        # 3. Inter-cluster attention
        inter_cluster_feature = self.interClusterAttention(
            intra_cluster_feature.view([batch_news_num, self.category_num, self.news_embedding_dim]),
            candidate_news_representation.view([batch_news_num, self.news_embedding_dim]),
            mask=user_history_category_mask.view([batch_news_num, self.category_num])
        ).view([batch_size, news_num, self.news_embedding_dim])      # [batch_size, news_num, news_embedding_dim]

        return inter_cluster_feature

class CNRCL(nn.Module):
    def __init__(self, config: Config):
        super(CNRCL, self).__init__()
        self.config = config
        self.news_encoder = CNRCLNewsEncoder(config)
        self.user_encoder = CNRCLUserEncoder(self.news_encoder, config)
        self.model_name = config.model
        self.batch_size = config.batch_size
        self.dropout = nn.Dropout(p=config.dropout_rate)
        
        # Click predictor (CNRCL mostly uses dot product in paper or MLP, config default is dot_product)
        # We'll support dot_product and FIM (as per original code)
        self.click_predictor = config.click_predictor
        if self.click_predictor == 'mlp':
            self.mlp = nn.Linear(in_features=self.news_encoder.news_embedding_dim * 2, out_features=self.news_encoder.news_embedding_dim // 2, bias=True)
            self.out = nn.Linear(in_features=self.news_encoder.news_embedding_dim // 2, out_features=1, bias=True)
        elif self.click_predictor == 'FIM':
             # Placeholder for FIM if needed, but for now mostly dot_product
             pass

    def initialize(self):
        self.news_encoder.initialize()
        self.user_encoder.initialize()
        if self.click_predictor == 'mlp':
            nn.init.xavier_uniform_(self.mlp.weight, gain=nn.init.calculate_gain('relu'))
            nn.init.zeros_(self.mlp.bias)

    def forward(self, user_ID, user_category, user_subCategory, user_title_text, user_title_mask, user_title_entity, user_content_text, user_content_mask, user_content_entity, user_history_mask, user_history_graph, user_history_category_mask, user_history_category_indices, \
                      news_category, news_subCategory, news_title_text, news_title_mask, news_title_entity, news_content_text, news_content_mask, news_content_entity):
        
        # Handle inference case where news inputs are 2D [batch, length] -> [batch, 1, length]
        if news_title_text.dim() == 2:
            news_title_text = news_title_text.unsqueeze(1)
            news_title_mask = news_title_mask.unsqueeze(1)
            news_title_entity = news_title_entity.unsqueeze(1)
            news_content_text = news_content_text.unsqueeze(1)
            news_content_mask = news_content_mask.unsqueeze(1)
            news_content_entity = news_content_entity.unsqueeze(1)
            news_category = news_category.unsqueeze(1)
            news_subCategory = news_subCategory.unsqueeze(1)

        user_embedding = None # CNRCL doesn't use user_ID embedding by default for CNRCL encoder

        news_representation = self.news_encoder(news_title_text, news_title_mask, news_title_entity, news_content_text, news_content_mask, news_content_entity, news_category, news_subCategory, user_embedding)
        # [batch_size, 1 + negative_sample_num, news_embedding_dim]
        
        user_representation = self.user_encoder(user_title_text, user_title_mask, user_title_entity, user_content_text, user_content_mask, user_content_entity, user_category, user_subCategory, \
                                                user_history_mask, user_history_graph, user_history_category_mask, user_history_category_indices, user_embedding, news_representation)
        # [batch_size, 1 + negative_sample_num, news_embedding_dim]
        
        logits_unsum = user_representation * news_representation    # [batch_size, 1 + negative_sample_num, news_embedding_dim]
        
        if self.click_predictor == 'dot_product':
            logits = (logits_unsum).sum(dim=2) # dot-product
        elif self.click_predictor == 'mlp':
            context = self.dropout(F.relu(self.mlp(torch.cat([user_representation, news_representation], dim=2)), inplace=True))
            logits = self.out(context).squeeze(dim=2)
        else:
             logits = (logits_unsum).sum(dim=2)

        # Slice logits to include only the main positive and standard negatives (1 + negative_sample_num)
        # Similarity samples are used only for auxiliary contrastive loss.
        logits_main = logits[:, :1 + self.config.negative_sample_num]

        if self.training:
            return logits_main, logits_unsum
        else:
            return logits_main.squeeze(-1)

import pickle
from config import Config
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence
import numpy as np
import json
import random
from models.modules.layers import Conv1D, Conv2D_Pool, MultiHeadAttention, Attention, QAttention, ScaledDotProduct_CandidateAttention, CandidateAttention


class NewsEncoder(nn.Module):
    def __init__(self, config: Config):
        super(NewsEncoder, self).__init__()
        self.config = config
        self.word_embedding_dim = config.word_embedding_dim
        self.word_embedding = nn.Embedding(num_embeddings=config.vocabulary_size, embedding_dim=self.word_embedding_dim)
        with open('word_embedding-' + str(config.word_threshold) + '-' + str(config.word_embedding_dim) + '-' + config.tokenizer + '-' + str(config.max_title_length) + '-' + str(config.max_abstract_length) + '-' + config.dataset + '.pkl', 'rb') as word_embedding_f:
            self.word_embedding.weight.data.copy_(pickle.load(word_embedding_f))
        self.category_embedding = nn.Embedding(num_embeddings=config.category_num, embedding_dim=config.category_embedding_dim)
        self.subCategory_embedding = nn.Embedding(num_embeddings=config.subCategory_num, embedding_dim=config.subCategory_embedding_dim)
        self.dropout = nn.Dropout(p=config.dropout_rate, inplace=True)
        self.dropout_ = nn.Dropout(p=config.dropout_rate, inplace=False)
        self.auxiliary_loss = None
        # 预训练emb
        self.item_emb_dic = np.load('./pretrain_emb/item_emb.npy', allow_pickle=True).item()
        # 加载ID到NEWS的映射字典
        with open('ID_news-200k.pkl', 'rb') as file:
            self.ID_news_200k = pickle.load(file)        
        # news to linked entity list
        with open('../graph/link_entity_dic.pkl', 'rb') as file:
            self.link_entity_dic = pickle.load(file)
        # entity to embedding
        with open('../graph/all_entity_emb_dic.pkl', 'rb') as file:
            self.all_entity_emb_dic = pickle.load(file)
        

    def initialize(self):
        nn.init.uniform_(self.category_embedding.weight, -0.1, 0.1)
        nn.init.uniform_(self.subCategory_embedding.weight, -0.1, 0.1)
        nn.init.zeros_(self.subCategory_embedding.weight[0])

    def forward(self, title_text, title_mask, title_entity, content_text, content_mask, content_entity, category, subCategory, user_embedding):
        raise Exception('Function forward must be implemented at sub-class')

    def feature_fusion(self, news_representation, category, subCategory, pretrain_emb, batch_linked_entity_emb_sum):
        # if pretrain_emb:
        category_representation = self.category_embedding(category)                                                                                    # [batch_size, news_num, category_embedding_dim]
        subCategory_representation = self.subCategory_embedding(subCategory)                                                                     # [batch_size, news_num, subCategory_embedding_dim]
        news_representation = torch.cat([news_representation, self.dropout(category_representation), self.dropout(subCategory_representation), pretrain_emb, batch_linked_entity_emb_sum], dim=2) # [batch_size, news_num, news_embedding_dim]
        return news_representation
        # else:
        #     category_representation = self.category_embedding(category)                                                                                    # [batch_size, news_num, category_embedding_dim]
        #     subCategory_representation = self.subCategory_embedding(subCategory)                                                                     # [batch_size, news_num, subCategory_embedding_dim]
        #     news_representation = torch.cat([news_representation, self.dropout(category_representation), self.dropout(subCategory_representation), batch_linked_entity_emb_sum], dim=2) # [batch_size, news_num, news_embedding_dim]
        #     return news_representation
    
    def get_pretrain_emb_val(self, batch_his):
        item_pretrain_emb = np.zeros((batch_his.shape[0], batch_his.shape[1], self.config.pretrain_emb_d))
        # 遍历原始列表
        for i in range(batch_his.shape[0]):
            for j in range(batch_his.shape[1]):
                id_read = batch_his[i, j].item()  # 获取id
                news_read = self.ID_news_200k[id_read] # 转为new
                # 忽略padding
                if news_read != "<PAD>":
                    vector = self.item_emb_dic[news_read]  # 查找对应的向量
                    item_pretrain_emb[i, j, :] = vector  # 将向量放入新的列表中的相应位置
        return item_pretrain_emb

    def get_linked_entity_emb(self, batch_his):
        batch_linked_entity_mask = np.zeros((batch_his.shape[0], batch_his.shape[1], self.config.max_linked_entity_length), dtype=bool)
        batch_linked_entity_emb = np.zeros((batch_his.shape[0], batch_his.shape[1], self.config.max_linked_entity_length, self.config.entity_embedding_dim))
        for i in range(batch_his.shape[0]):
            for j in range(batch_his.shape[1]):
                id_read = batch_his[i, j].item()
                news_read = self.ID_news_200k[id_read]
                if news_read != "<PAD>":
                    # read linked entity
                    linked_entity = self.link_entity_dic[news_read]
                    # generate mask
                    batch_linked_entity_mask[i, j, :min(len(linked_entity), self.config.max_linked_entity_length)] = 1
                    # sample 20 linked entity
                    if len(linked_entity) > self.config.max_linked_entity_length:
                        linked_entity = random.sample(linked_entity, self.config.max_linked_entity_length)
                    # read entity emb
                    for k in range(len(linked_entity)):
                        batch_linked_entity_emb[i, j, k, :] = self.all_entity_emb_dic[linked_entity[k]]
        return batch_linked_entity_emb, batch_linked_entity_mask


class MHSA(NewsEncoder):
    def __init__(self, config: Config):
        super(MHSA, self).__init__(config)
        self.max_sentence_length = config.max_title_length
        self.feature_dim = config.head_num * config.head_dim
        self.multiheadAttention = MultiHeadAttention(config.head_num, config.word_embedding_dim, config.max_title_length, config.max_title_length, config.head_dim, config.head_dim)
        self.attention = Attention(config.head_num*config.head_dim, config.attention_dim)
        self.entity_attention = QAttention(config.entity_embedding_dim*3, config.entity_attention_dim)
        self.news_embedding_dim = config.head_num * config.head_dim + config.category_embedding_dim + config.subCategory_embedding_dim + config.pretrain_rep_d + config.entity_hidden_dim
        # self.news_embedding_dim = config.head_num * config.head_dim + config.category_embedding_dim + config.subCategory_embedding_dim  + config.entity_hidden_dim
        # 得到大模型emb后进行操作
        self.dense1 = nn.Linear(config.pretrain_emb_d, config.pretrain_hidden_dim, bias=True)
        self.dense2 = nn.Linear(config.pretrain_hidden_dim, config.pretrain_rep_d, bias=True)
        # 查询到entity向量后进行操作
        self.dense3 = nn.Linear(config.entity_embedding_dim*config.entity_att_head_num, config.entity_hidden_dim*2, bias=True)
        self.dense4 = nn.Linear(config.entity_hidden_dim*2,  config.entity_hidden_dim, bias=True)

        # 得到大模型emb后进行Q pre操作
        self.denseQ_pre1 = nn.Linear(config.pretrain_emb_d, config.pretrain_hidden_dim, bias=True)
        self.denseQ_pre2 = nn.Linear(config.pretrain_hidden_dim, config.pretrain_rep_d, bias=True)

        # KG多头注意力
        self.denseHead1 = nn.Linear(config.pretrain_rep_d,  config.entity_embedding_dim, bias=True)
        self.denseHead2 = nn.Linear(config.pretrain_rep_d,  config.entity_embedding_dim, bias=True)
        self.denseHead3 = nn.Linear(config.pretrain_rep_d,  config.entity_embedding_dim, bias=True)
    def initialize(self):
        super().initialize()
        self.multiheadAttention.initialize()
        self.attention.initialize()
        self.entity_attention.initialize()
        nn.init.xavier_uniform_(self.dense1.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.zeros_(self.dense1.bias)
        nn.init.xavier_uniform_(self.dense2.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.zeros_(self.dense2.bias)
        nn.init.xavier_uniform_(self.dense3.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.zeros_(self.dense3.bias)
        nn.init.xavier_uniform_(self.dense4.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.zeros_(self.dense4.bias)
        nn.init.xavier_uniform_(self.denseQ_pre1.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.zeros_(self.denseQ_pre1.bias)
        nn.init.xavier_uniform_(self.denseQ_pre2.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.zeros_(self.denseQ_pre2.bias)
        nn.init.xavier_uniform_(self.denseHead1.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.zeros_(self.denseHead1.bias)
        nn.init.xavier_uniform_(self.denseHead2.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.zeros_(self.denseHead2.bias)
        nn.init.xavier_uniform_(self.denseHead3.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.zeros_(self.denseHead3.bias)

    def forward(self, title_text, title_mask, title_entity, content_text, content_mask, content_entity, category, subCategory, user_embedding, sample_index):
        batch_size = title_text.size(0)
        news_num = title_text.size(1)
        batch_news_num = batch_size * news_num
        mask = title_mask.view([batch_news_num, self.max_sentence_length])                                                          # [batch_size * news_num, max_sentence_length]
        # 1. word embedding
        w = self.dropout(self.word_embedding(title_text)).view([batch_news_num, self.max_sentence_length, self.word_embedding_dim]) # [batch_size * news_num, max_sentence_length, word_embedding_dim]
        # 2. multi-head self-attention
        c = self.dropout(self.multiheadAttention(w, w, w, mask))                                                                    # [batch_size * news_num, max_sentence_length, news_embedding_dim]
        # 3. attention layer
        news_representation = self.attention(c, mask=mask).view([batch_size, news_num, self.feature_dim])                           # [batch_size, news_num, news_embedding_dim]
        # 4. pretrain
        llm_orignal = torch.from_numpy(self.get_pretrain_emb_val(sample_index)).float().cuda(non_blocking=True) #
        llm_orignal = F.normalize(llm_orignal, dim=2)
        pretrain_representation = F.relu(self.dropout(self.dense1(llm_orignal)), inplace=True)
        pretrain_representation = F.relu(self.dropout(self.dense2(pretrain_representation)), inplace=True) 
        # 5. linked node emb
        # entity Query
        llm_query = F.relu(self.dropout(self.denseQ_pre1(llm_orignal)), inplace=True)
        llm_query = F.relu(self.dropout(self.denseQ_pre2(llm_query)), inplace=True) 
        entityQuery_head1 = F.relu(self.dropout(self.denseHead1(llm_query)), inplace=True).view([batch_news_num, self.config.entity_embedding_dim])  
        entityQuery_head1 = entityQuery_head1.unsqueeze(1).expand(-1, self.config.max_linked_entity_length, -1)
        entityQuery_head2 = F.relu(self.dropout(self.denseHead2(llm_query)), inplace=True).view([batch_news_num, self.config.entity_embedding_dim])  
        entityQuery_head2 = entityQuery_head2.unsqueeze(1).expand(-1, self.config.max_linked_entity_length, -1)
        entityQuery_head3 = F.relu(self.dropout(self.denseHead3(llm_query)), inplace=True).view([batch_news_num, self.config.entity_embedding_dim])  
        entityQuery_head3 = entityQuery_head3.unsqueeze(1).expand(-1, self.config.max_linked_entity_length, -1)
        entityQuery = [entityQuery_head1, entityQuery_head2, entityQuery_head3]
        # get entity
        batch_linked_entity_emb, batch_linked_entity_mask = self.get_linked_entity_emb(sample_index)
        batch_linked_entity_mask = torch.from_numpy(batch_linked_entity_mask).cuda(non_blocking=True).view([batch_news_num, self.config.max_linked_entity_length])  
        batch_linked_entity_emb = torch.from_numpy(batch_linked_entity_emb).float().cuda(non_blocking=True).view([batch_news_num, self.config.max_linked_entity_length, self.config.entity_embedding_dim])
        batch_linked_entity_emb_sum = self.entity_attention(batch_linked_entity_emb, entityQuery, mask=batch_linked_entity_mask).view([batch_size, news_num, self.config.entity_embedding_dim*self.config.entity_att_head_num])                           # [batch_size, news_num, news_embedding_dim]
        batch_linked_entity_emb_sum = F.normalize(batch_linked_entity_emb_sum, dim=2)
        batch_linked_entity_emb_sum = F.relu(self.dropout(self.dense3(batch_linked_entity_emb_sum)), inplace=True)
        batch_linked_entity_emb_sum = F.relu(self.dropout(self.dense4(batch_linked_entity_emb_sum)), inplace=True) 

        # 6. feature fusion
        news_representation = self.feature_fusion(news_representation, category, subCategory, pretrain_representation, batch_linked_entity_emb_sum)                                       # [batch_size, news_num, news_embedding_dim]
        # news_representation = self.feature_fusion(news_representation, category, subCategory, None, batch_linked_entity_emb_sum)                                       # [batch_size, news_num, news_embedding_dim]
        return news_representation


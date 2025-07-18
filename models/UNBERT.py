# =========================================================================
# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================

from typing import List, Tuple, Dict, Any
import torch
import torch.nn as nn
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from config import Config
from models.modules.unbert.configuration_bert import BertConfig
from models.modules.unbert.unbertModel import UNBertModel


class UNBERT(nn.Module):
    def __init__(self, config: Config):
        super(UNBERT, self).__init__()
        self._pretrained = config.pretrain
        self._level_state = config.level_sate
        self._news_mode = config.news_mode
        self.max_len = config.seq_max_len
        self.model_name = config.model
        self.use_user_embedding = False
        self.config = config

        self._config = BertConfig.from_pretrained(self._pretrained)
        self._model = UNBertModel.from_pretrained(self._pretrained, config=self._config)

        if self._level_state == 'both':
            self._dense = nn.Linear(self._config.hidden_size * 2, 2)
        else:
            self._dense = nn.Linear(self._config.hidden_size, 2)

        if self._news_mode == 'attention':
            assert self.max_len is not None
            self.att = nn.Sequential(
                       nn.Linear(self.max_len * self._config.hidden_size, 128),
                       nn.Sigmoid(),
                       nn.Linear(128, self.max_len)
                       )
        else:
            self.att = None

    def forward(self,
                input_ids: torch.Tensor,
                input_mask: torch.Tensor = None,
                segment_ids: torch.Tensor = None,
                news_segment_ids: torch.Tensor = None,
                sentence_ids: torch.Tensor = None,
                sentence_mask: torch.Tensor = None,
                sentence_segment_ids: torch.Tensor = None
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        output = self._model(input_ids,
                             attention_mask = input_mask,
                             token_type_ids = segment_ids,
                             news_segment_ids = news_segment_ids,
                             sentence_input = (sentence_ids, sentence_mask, sentence_segment_ids),
                             news_mode=self._news_mode,
                             att_mapping=self.att)
        word_hidden = output[0][:, 0, :]
        sentence_hidden = output[1][:, 0, :]
        if self._level_state == 'word':
            hidden = word_hidden
        elif self._level_state == 'news':
            hidden = sentence_hidden
        else:
            hidden = torch.cat([word_hidden, sentence_hidden], -1)
        score = self._dense(hidden).squeeze(-1)
        return score
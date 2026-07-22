import math

from config import Config
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.modules import newsEncoders


class SEIN(nn.Module):
    def __init__(self, config: Config):
        super(SEIN, self).__init__()
        self.config = config
        self.model_name = config.model

        self.news_encoder = newsEncoders.MHSA(config)
        self.news_dim = self.news_encoder.news_embedding_dim
        self.hidden_dim = getattr(config, "sein_hidden_dim", min(self.news_dim, 256))
        self.stage_num = getattr(config, "sein_stage_num", 5)
        self.gcn_layer_num = getattr(config, "sein_gcn_layer_num", 2)
        self.temperature = getattr(config, "sein_temperature", 0.1)
        self.lambda_cl = getattr(config, "sein_consistency_loss_weight", 1e-2)
        self.lambda_sl = getattr(config, "sein_smoothness_loss_weight", 1e-2)

        self.user_embedding = nn.Embedding(config.user_num, self.hidden_dim)
        self.news_id_embedding = nn.Embedding(config.news_num, self.hidden_dim, padding_idx=0)
        self.semantic_proj = nn.Linear(self.news_dim, self.hidden_dim)
        self.stage_lstm = nn.LSTM(self.hidden_dim, self.hidden_dim, batch_first=True)
        self.item_lstm = nn.LSTM(self.hidden_dim, self.hidden_dim, batch_first=True)
        self.position_embedding = nn.Embedding(config.max_history_num, self.hidden_dim)
        self.history_query = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.history_key = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.stage_query = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.stage_key = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.user_gate = nn.Linear(self.hidden_dim * 4, self.hidden_dim)
        self.score_mlp = nn.Sequential(
            nn.Linear(self.hidden_dim * 4, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(self.hidden_dim, 1),
        )

        attn_heads = getattr(config, "sein_attention_heads", 4)
        attn_layers = getattr(config, "sein_attention_layers", 1)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=attn_heads,
            dim_feedforward=self.hidden_dim * 4,
            dropout=config.dropout_rate,
            activation="relu",
            batch_first=True,
        )
        self.long_range_encoder = nn.TransformerEncoder(encoder_layer, num_layers=attn_layers)
        self.dropout = nn.Dropout(config.dropout_rate)

        self.register_buffer("global_graph", torch.empty(0), persistent=False)
        self.register_buffer("seen_user_mask", torch.empty(0, dtype=torch.bool), persistent=False)
        self.register_buffer("seen_news_mask", torch.empty(0, dtype=torch.bool), persistent=False)

    def initialize(self):
        self.news_encoder.initialize()
        nn.init.xavier_uniform_(self.semantic_proj.weight)
        nn.init.zeros_(self.semantic_proj.bias)
        nn.init.normal_(self.user_embedding.weight, std=0.02)
        nn.init.normal_(self.news_id_embedding.weight, std=0.02)
        nn.init.zeros_(self.news_id_embedding.weight[0])
        nn.init.normal_(self.position_embedding.weight, std=0.02)
        for module in [self.history_query, self.history_key, self.stage_query, self.stage_key]:
            nn.init.xavier_uniform_(module.weight)
        nn.init.xavier_uniform_(self.user_gate.weight)
        nn.init.zeros_(self.user_gate.bias)
        for module in self.score_mlp:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        for module in [self.stage_lstm, self.item_lstm]:
            for name, parameter in module.named_parameters():
                if "weight" in name:
                    nn.init.xavier_uniform_(parameter)
                else:
                    nn.init.zeros_(parameter)

    def set_corpus(self, corpus):
        edges = set()
        seen_users = torch.zeros(self.config.user_num, dtype=torch.bool)
        seen_news = torch.zeros(self.config.news_num, dtype=torch.bool)
        user_offset = self.config.user_num
        for behavior in corpus.train_behaviors:
            user_id = int(behavior[0])
            if 0 <= user_id < self.config.user_num:
                seen_users[user_id] = True
            clicked_items = [int(i) for i, m in zip(behavior[1], behavior[2]) if m > 0 and int(i) > 0]
            clicked_items.append(int(behavior[3]))
            if len(behavior) > 4:
                clicked_items.extend(int(i) for i in behavior[4] if int(i) > 0)
            for news_id in clicked_items:
                if 0 < news_id < self.config.news_num:
                    seen_news[news_id] = True
            graph_items = [int(i) for i, m in zip(behavior[1], behavior[2]) if m > 0 and int(i) > 0]
            graph_items.append(int(behavior[3]))
            for news_id in graph_items:
                if 0 <= user_id < self.config.user_num and 0 < news_id < self.config.news_num:
                    edges.add((user_id, user_offset + news_id))
                    edges.add((user_offset + news_id, user_id))

        node_num = self.config.user_num + self.config.news_num
        self.seen_user_mask = seen_users
        self.seen_news_mask = seen_news
        if not edges:
            self.global_graph = torch.sparse_coo_tensor(
                torch.zeros((2, 0), dtype=torch.long),
                torch.zeros(0, dtype=torch.float32),
                (node_num, node_num),
            ).coalesce()
            return

        indices = torch.tensor(list(edges), dtype=torch.long).t()
        degrees = torch.bincount(indices[0], minlength=node_num).float().clamp_min(1.0)
        values = degrees[indices[0]].rsqrt() * degrees[indices[1]].rsqrt()
        self.global_graph = torch.sparse_coo_tensor(indices, values, (node_num, node_num)).coalesce()

    def _global_embeddings(self):
        user_base = self.user_embedding.weight
        news_base = self.news_id_embedding.weight
        if self.seen_user_mask.numel() == self.config.user_num:
            user_base = user_base * self.seen_user_mask.to(user_base.device).unsqueeze(-1).float()
        if self.seen_news_mask.numel() == self.config.news_num:
            news_base = news_base * self.seen_news_mask.to(news_base.device).unsqueeze(-1).float()
        all_embeddings = torch.cat([user_base, news_base], dim=0)
        embeddings = [all_embeddings]

        graph = self.global_graph
        if graph.numel() == 0:
            return user_base, news_base

        graph = graph.to(all_embeddings.device)
        propagated = all_embeddings
        for _ in range(self.gcn_layer_num):
            propagated = torch.sparse.mm(graph, propagated)
            embeddings.append(propagated)

        final = torch.stack(embeddings, dim=0).sum(dim=0)
        return final[: self.config.user_num], final[self.config.user_num :]

    def _news_id_lookup(self, indices):
        embeddings = self.news_id_embedding(indices)
        if self.seen_news_mask.numel() == self.config.news_num:
            valid = self.seen_news_mask.to(indices.device)[indices].unsqueeze(-1).float()
            embeddings = embeddings * valid
        return embeddings

    def _encode_news(self, title_text, title_mask, title_entity, content_text, content_mask, content_entity, category, subCategory):
        semantic = self.news_encoder(
            title_text,
            title_mask,
            title_entity,
            content_text,
            content_mask,
            content_entity,
            category,
            subCategory,
            None,
        )
        return self.semantic_proj(semantic)

    def _stage_representations(self, history_embedding, history_mask, history_stage_ids=None):
        batch_size, history_len, dim = history_embedding.size()
        device = history_embedding.device
        if history_stage_ids is None:
            position_ids = torch.arange(history_len, device=device)
            stage_ids = torch.clamp(position_ids * self.stage_num // max(history_len, 1), max=self.stage_num - 1)
            stage_one_hot = F.one_hot(stage_ids, num_classes=self.stage_num).float()
            stage_mask = history_mask.unsqueeze(-1) * stage_one_hot.unsqueeze(0)
        else:
            history_stage_ids = history_stage_ids.to(device=device, dtype=torch.long)
            valid_stage = (history_stage_ids >= 0) & (history_stage_ids < self.stage_num) & history_mask.bool()
            safe_stage_ids = history_stage_ids.clamp(min=0, max=self.stage_num - 1)
            stage_one_hot = F.one_hot(safe_stage_ids, num_classes=self.stage_num).float()
            stage_mask = valid_stage.unsqueeze(-1).float() * stage_one_hot
        stage_count = stage_mask.sum(dim=1).clamp_min(1.0)
        stage_embedding = torch.einsum("bhd,bht->btd", history_embedding, stage_mask) / stage_count.unsqueeze(-1)
        stage_valid = stage_mask.sum(dim=1) > 0
        return stage_embedding, stage_valid

    def _last_valid_stage(self, stage_sequence, stage_valid):
        stage_indices = torch.arange(stage_valid.size(1), device=stage_valid.device).unsqueeze(0)
        last_stage = stage_indices.expand_as(stage_valid).masked_fill(~stage_valid, -1).max(dim=1).values
        gather_index = last_stage.clamp_min(0).view(-1, 1, 1).expand(-1, 1, stage_sequence.size(-1))
        return stage_sequence.gather(dim=1, index=gather_index).squeeze(1)

    def _long_range_user(self, history_embedding, history_mask):
        batch_size, history_len, _ = history_embedding.size()
        device = history_embedding.device
        safe_mask = history_mask.bool()
        empty_history = ~safe_mask.any(dim=1)
        if empty_history.any():
            safe_mask = safe_mask.clone()
            safe_mask[empty_history, 0] = True

        pos = self.position_embedding(torch.arange(history_len, device=device)).unsqueeze(0)
        sequence = history_embedding + pos
        encoded = self.long_range_encoder(sequence, src_key_padding_mask=~safe_mask)
        encoded = encoded * safe_mask.unsqueeze(-1).float()
        return encoded.sum(dim=1) / safe_mask.sum(dim=1, keepdim=True).float().clamp_min(1.0)

    def _candidate_attention(self, query, keys, values, mask):
        q = query.unsqueeze(2)
        attn = (q * keys.unsqueeze(1)).sum(dim=-1) / math.sqrt(float(self.hidden_dim))
        attn = attn.masked_fill(~mask.unsqueeze(1), -1e9)
        empty = ~mask.any(dim=1)
        if empty.any():
            attn = attn.clone()
            attn[empty] = 0.0
        weights = F.softmax(attn, dim=-1)
        attended = torch.matmul(weights, values)
        if empty.any():
            attended = attended.masked_fill(empty.view(-1, 1, 1), 0.0)
        return attended

    def _auxiliary_loss(self, temporal_user, global_user, stage_sequence, stage_valid):
        temporal_user = F.normalize(temporal_user, dim=-1)
        global_user = F.normalize(global_user, dim=-1)
        contrast_logits = torch.matmul(temporal_user, global_user.t()) / self.temperature
        labels = torch.arange(contrast_logits.size(0), device=contrast_logits.device)
        consistency_loss = F.cross_entropy(contrast_logits, labels)

        if stage_sequence.size(1) <= 1:
            smoothness_loss = stage_sequence.new_tensor(0.0)
        else:
            adjacent_valid = stage_valid[:, 1:] & stage_valid[:, :-1]
            diffs = (stage_sequence[:, 1:] - stage_sequence[:, :-1]).pow(2).sum(dim=-1)
            if adjacent_valid.any():
                smoothness_loss = diffs[adjacent_valid].mean()
            else:
                smoothness_loss = stage_sequence.new_tensor(0.0)

        return self.lambda_cl * consistency_loss + self.lambda_sl * smoothness_loss

    def _batch_indices_to_tensor(self, indices, device):
        if indices is None or torch.is_tensor(indices):
            return indices
        if isinstance(indices, (list, tuple)) and indices and torch.is_tensor(indices[0]):
            return torch.stack(list(indices), dim=1).to(device=device, dtype=torch.long)
        return torch.as_tensor(indices, device=device, dtype=torch.long)

    def _encode_candidate_items(self, candidate_base):
        batch_size, candidate_num, hidden_dim = candidate_base.size()
        flat_candidates = candidate_base.reshape(batch_size * candidate_num, 1, hidden_dim)
        flat_sequence, _ = self.item_lstm(flat_candidates)
        return flat_sequence.reshape(batch_size, candidate_num, hidden_dim)

    def _history_stage_ids_from_extra_args(self, extra_args, history_mask):
        for arg in reversed(extra_args):
            if not torch.is_tensor(arg):
                continue
            if arg.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.long):
                continue
            if arg.dim() == history_mask.dim() and tuple(arg.shape) == tuple(history_mask.shape):
                return arg
        return None

    def _unpack_extra_args(self, extra_args, history_mask):
        if self.config.dataset_name in {"MIND", "gossipcop"}:
            history_index = extra_args[0] if len(extra_args) > 0 else None
            sample_index = extra_args[1] if len(extra_args) > 1 else None
            stage_args = extra_args[2:]
        else:
            history_index = extra_args[2] if len(extra_args) > 2 else None
            sample_index = extra_args[3] if len(extra_args) > 3 else None
            stage_args = extra_args[4:]
        history_stage_ids = self._history_stage_ids_from_extra_args(stage_args, history_mask)
        return history_index, sample_index, history_stage_ids

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
        *extra_args,
    ):
        history_index, sample_index, history_stage_ids = self._unpack_extra_args(
            extra_args, user_history_mask
        )
        history_index = self._batch_indices_to_tensor(history_index, user_ID.device)
        sample_index = self._batch_indices_to_tensor(sample_index, user_ID.device)

        global_users, global_news = self._global_embeddings()
        user_global = global_users[user_ID]

        history_semantic = self._encode_news(
            user_title_text,
            user_title_mask,
            user_title_entity,
            user_content_text,
            user_content_mask,
            user_content_entity,
            user_category,
            user_subCategory,
        )
        history_id = self._news_id_lookup(history_index) if history_index is not None else 0.0
        history_global = global_news[history_index] if history_index is not None else 0.0
        history_embedding = self.dropout(history_semantic + history_id + history_global)

        stage_embedding, stage_valid = self._stage_representations(history_embedding, user_history_mask, history_stage_ids)
        stage_sequence, _ = self.stage_lstm(stage_embedding)
        short_term_user = self._last_valid_stage(stage_sequence, stage_valid)
        long_range_user = self._long_range_user(history_embedding, user_history_mask)
        user_representation = user_global + short_term_user + long_range_user

        candidate_semantic = self._encode_news(
            news_title_text,
            news_title_mask,
            news_title_entity,
            news_content_text,
            news_content_mask,
            news_content_entity,
            news_category,
            news_subCategory,
        )
        if candidate_semantic.dim() == 2:
            candidate_semantic = candidate_semantic.unsqueeze(1)
        if sample_index is not None and sample_index.dim() == 1:
            sample_index = sample_index.unsqueeze(1)
        candidate_id = self._news_id_lookup(sample_index) if sample_index is not None else 0.0
        candidate_global = global_news[sample_index] if sample_index is not None else 0.0
        candidate_base = self.dropout(candidate_semantic + candidate_id + candidate_global)
        candidate_sequence = self._encode_candidate_items(candidate_base)
        candidate_representation = candidate_global + candidate_sequence

        history_keys = self.history_key(history_embedding)
        history_query = self.history_query(candidate_representation)
        history_match = self._candidate_attention(
            history_query,
            history_keys,
            history_embedding,
            user_history_mask.bool(),
        )
        stage_keys = self.stage_key(stage_sequence)
        stage_query = self.stage_query(candidate_representation)
        stage_match = self._candidate_attention(stage_query, stage_keys, stage_sequence, stage_valid)

        base_user = user_representation.unsqueeze(1).expand_as(candidate_representation)
        gate = torch.sigmoid(self.user_gate(torch.cat([base_user, history_match, stage_match, candidate_representation], dim=-1)))
        candidate_user = gate * history_match + (1.0 - gate) * (base_user + stage_match)
        score_feature = torch.cat(
            [
                candidate_user,
                candidate_representation,
                candidate_user * candidate_representation,
                torch.abs(candidate_user - candidate_representation),
            ],
            dim=-1,
        )
        mlp_logits = self.score_mlp(score_feature).squeeze(-1)
        cosine_logits = (
            F.normalize(candidate_user, dim=-1) * F.normalize(candidate_representation, dim=-1)
        ).sum(dim=-1) * math.sqrt(float(self.hidden_dim))
        logits = mlp_logits + cosine_logits
        if logits.size(1) == 1:
            logits = logits.squeeze(1)

        if self.training:
            aux_loss = self._auxiliary_loss(short_term_user, user_global, stage_sequence, stage_valid)
            return logits, aux_loss
        return logits

import json
import os
import re
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from config import Config
from models.modules.click_predictor import DotProduct
from models.modules.layers import Attention


MODEL_NAMES = ("ONCE",)


def _as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def _dtype_from_config(value: str) -> torch.dtype:
    value = str(value).lower()
    if value in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if value in {"fp16", "float16", "half"}:
        return torch.float16
    return torch.float32


class ONCEDIREQwen3NewsEncoder(nn.Module):
    """
    Qwen3 item encoder used by ONCE/DIRE.

    The paper and Legommenders implementation cache lower LLM hidden states
    and only fine-tune the upper layers. This module mirrors that path: it
    builds or loads a hidden-state cache indexed by NewsTorch's news IDs, then
    runs the remaining Qwen3 layers, projects to the recommender hidden size,
    and applies additive attention over the token sequence.
    """

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.news_embedding_dim = getattr(config, "hidden_size", 256)
        self.attention = Attention(self.news_embedding_dim, config.attention_dim)
        self.dropout = nn.Dropout(p=config.dropout_rate)

        self.transformer = None
        self.projection: Optional[nn.Linear] = None
        self.hidden_cache = None
        self.mask_cache = None
        self._projection_initialized = False
        self._network_prepared = False

        self.max_length = int(getattr(config, "once_max_length", 128))
        self.page_size = int(getattr(config, "once_item_page_size", 16))
        self.forward_page_size = int(getattr(config, "once_forward_page_size", 32))
        self.cache_key = getattr(config, "once_cache_key", "qwen3_0.6b")
        self.cache_root = getattr(
            config,
            "once_cache_dir",
            os.path.join(config.data_path, "once_dire_qwen3_naml"),
        )
        self.cache_dtype = np.float16 if str(getattr(config, "once_cache_dtype", "float16")).lower() in {"fp16", "float16", "half"} else np.float32
        self.model_name_or_path = (
            os.environ.get("ONCE_QWEN_MODEL", "")
            or getattr(config, "once_qwen_model_name_or_path", "")
            or "Qwen/Qwen3-0.6B"
        )
        self.hf_token = self._get_hf_token(config)
        self.dtype = _dtype_from_config(getattr(config, "once_dtype", "bfloat16"))
        self.trust_remote_code = _as_bool(getattr(config, "once_trust_remote_code", False))
        self.local_files_only = _as_bool(getattr(config, "once_local_files_only", False))
        self.use_lora = _as_bool(getattr(config, "once_use_lora", True))
        self.lora_r = getattr(config, "once_lora_r", 32)
        self.lora_alpha = getattr(config, "once_lora_alpha", 128)
        self.lora_dropout = float(getattr(config, "once_lora_dropout", 0.1))
        self.lora_target_modules = self._parse_lora_target_modules(
            getattr(config, "once_lora_target_modules", ("q_proj", "v_proj"))
        )
        self.tune_from = self._parse_tune_from(getattr(config, "once_tune_from", -2))

    @staticmethod
    def _parse_tune_from(value):
        if value is None:
            return None
        if isinstance(value, str) and value.strip().lower() in {"", "none", "null"}:
            return None
        return int(value)

    @staticmethod
    def _parse_lora_target_modules(value):
        if value is None:
            return ["q_proj", "v_proj"]
        if isinstance(value, str):
            value = [item.strip() for item in value.split(",")]
        modules = [str(item).strip() for item in value if str(item).strip()]
        return modules or ["q_proj", "v_proj"]

    @staticmethod
    def _get_hf_token(config: Config):
        token = getattr(config, "once_hf_token", None)
        if token:
            return token
        token_env = getattr(config, "once_hf_token_env", "HF_TOKEN")
        return (
            os.environ.get("ONCE_HF_TOKEN")
            or os.environ.get(str(token_env), "")
            or os.environ.get("HF_TOKEN")
            or os.environ.get("HUGGING_FACE_HUB_TOKEN")
            or None
        )

    def _hf_kwargs(self):
        kwargs = {
            "trust_remote_code": self.trust_remote_code,
            "local_files_only": self.local_files_only,
        }
        if self.hf_token:
            kwargs["token"] = self.hf_token
        return kwargs

    def _raise_checkpoint_error(self, error):
        raise RuntimeError(
            f"Cannot load the Qwen3 checkpoint {self.model_name_or_path!r}. "
            "Check network access, or set ONCE_QWEN_MODEL to a local directory "
            "containing safetensors weights and enable once_local_files_only."
        ) from error

    def initialize(self):
        self.attention.initialize()
        self._initialize_projection()

    def _initialize_projection(self):
        if self.projection is None or self._projection_initialized:
            return
        nn.init.xavier_uniform_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)
        self._projection_initialized = True

    def set_corpus(self, corpus):
        if self._network_prepared:
            return
        if self.tune_from is None:
            raise ValueError(
                "ONCE-DIRE-QWEN3-NAML requires once_tune_from to cache frozen Qwen3 layers. "
                "Use once_tune_from: -2 in config/once.yaml."
            )

        self._load_transformer()
        self._normalize_tune_from()
        self._ensure_hidden_cache(corpus)
        self._slice_transformer_layers()
        self._apply_lora()
        self._network_prepared = True

    def _load_transformer(self):
        if self.transformer is not None:
            return
        try:
            from transformers import AutoModel
        except ImportError as exc:
            raise ImportError("transformers is required for ONCE-DIRE-QWEN3-NAML.") from exc

        try:
            self.transformer = AutoModel.from_pretrained(
                self.model_name_or_path,
                torch_dtype=self.dtype,
                use_safetensors=True,
                **self._hf_kwargs(),
            )
        except (OSError, ValueError) as exc:
            self._raise_checkpoint_error(exc)
        model_type = str(getattr(self.transformer.config, "model_type", ""))
        if not model_type.lower().startswith("qwen3"):
            raise ValueError(
                f"ONCE requires a Qwen3 checkpoint, but {self.model_name_or_path!r} "
                f"declares model_type={model_type!r}."
            )
        hidden_size = self.transformer.config.hidden_size
        self.projection = nn.Linear(hidden_size, self.news_embedding_dim)
        self._initialize_projection()

    def _normalize_tune_from(self):
        num_layers = self.transformer.config.num_hidden_layers
        if self.tune_from < 0:
            self.tune_from = num_layers + self.tune_from
        if self.tune_from < 0 or self.tune_from >= num_layers - 1:
            raise ValueError(
                f"once_tune_from must resolve into [0, {num_layers - 2}] so at "
                f"least one Qwen3 layer remains trainable, got {self.tune_from}."
            )

    @property
    def _cache_dir(self):
        return os.path.join(
            self.cache_root,
            _safe_name(str(getattr(self.config, "DATASET_ROOT", "dataset"))),
            _safe_name(str(self.cache_key)),
        )

    @property
    def _hidden_cache_path(self):
        filename = f"hidden_layer_{self.tune_from}_maxlen_{self.max_length}_{self.cache_dtype.__name__}.npy"
        return os.path.join(self._cache_dir, filename)

    @property
    def _mask_cache_path(self):
        filename = f"mask_layer_{self.tune_from}_maxlen_{self.max_length}.npy"
        return os.path.join(self._cache_dir, filename)

    @property
    def _cache_done_path(self):
        filename = f"done_layer_{self.tune_from}_maxlen_{self.max_length}_{self.cache_dtype.__name__}.json"
        return os.path.join(self._cache_dir, filename)

    def _ensure_hidden_cache(self, corpus):
        os.makedirs(self._cache_dir, exist_ok=True)
        if not self._cache_files_match(corpus.news_num):
            self._build_hidden_cache(corpus)

        self.hidden_cache = np.load(self._hidden_cache_path, mmap_mode="r")
        self.mask_cache = np.load(self._mask_cache_path, mmap_mode="r")

    def _cache_files_match(self, news_num: int) -> bool:
        if not (
            os.path.exists(self._hidden_cache_path)
            and os.path.exists(self._mask_cache_path)
            and os.path.exists(self._cache_done_path)
        ):
            return False
        try:
            hidden_cache = np.load(self._hidden_cache_path, mmap_mode="r")
            mask_cache = np.load(self._mask_cache_path, mmap_mode="r")
            with open(self._cache_done_path, "r", encoding="utf-8") as done_f:
                manifest = json.load(done_f)
            return (
                hidden_cache.shape == (news_num, self.max_length, self.transformer.config.hidden_size)
                and mask_cache.shape == (news_num, self.max_length)
                and manifest.get("model") == self.model_name_or_path
                and manifest.get("tune_from") == self.tune_from
                and manifest.get("hidden_state_index") == self.tune_from + 1
            )
        except (OSError, ValueError, json.JSONDecodeError):
            return False

    def _build_hidden_cache(self, corpus):
        try:
            from transformers import AutoTokenizer
            from numpy.lib.format import open_memmap
        except ImportError as exc:
            raise ImportError(
                "transformers is required to build the ONCE Qwen3 hidden-state cache."
            ) from exc

        try:
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_name_or_path,
                **self._hf_kwargs(),
            )
        except (OSError, ValueError) as exc:
            self._raise_checkpoint_error(exc)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
        if tokenizer.pad_token_id is None:
            raise ValueError(
                "The Qwen3 tokenizer needs a pad, eos, or unk token for ONCE cache generation."
            )

        prompts = self._collect_prompts(corpus)
        hidden_size = self.transformer.config.hidden_size
        if os.path.exists(self._cache_done_path):
            os.remove(self._cache_done_path)
        hidden_cache = open_memmap(
            self._hidden_cache_path,
            mode="w+",
            dtype=self.cache_dtype,
            shape=(len(prompts), self.max_length, hidden_size),
        )
        mask_cache = open_memmap(
            self._mask_cache_path,
            mode="w+",
            dtype=np.int8,
            shape=(len(prompts), self.max_length),
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.transformer.to(device=device, dtype=self.dtype)
        self.transformer.eval()
        with torch.no_grad():
            for start in range(0, len(prompts), self.page_size):
                end = min(start + self.page_size, len(prompts))
                encoded = tokenizer(
                    prompts[start:end],
                    add_special_tokens=False,
                    padding="max_length",
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                input_ids = encoded["input_ids"].to(device)
                attention_mask = encoded["attention_mask"].to(device)
                outputs = self.transformer(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    use_cache=False,
                    return_dict=True,
                )
                # hidden_states[0] is the token embedding output. Cache the
                # output of the final frozen layer, then retain all later layers.
                batch_hidden = outputs.hidden_states[self.tune_from + 1].float().cpu()
                batch_nan_mask = torch.isnan(batch_hidden).any(dim=-1)
                if batch_nan_mask.any():
                    batch_hidden[batch_nan_mask] = torch.rand_like(batch_hidden[batch_nan_mask])
                    bad_rows = batch_nan_mask.any(dim=-1)
                    encoded["attention_mask"][bad_rows] = 0
                    encoded["attention_mask"][bad_rows, 0] = 1
                hidden_cache[start:end] = batch_hidden.numpy().astype(self.cache_dtype, copy=False)
                mask_cache[start:end] = encoded["attention_mask"].numpy().astype(np.int8, copy=False)
                hidden_cache.flush()
                mask_cache.flush()
        with open(self._cache_done_path, "w", encoding="utf-8") as done_f:
            json.dump(
                {
                    "news_num": len(prompts),
                    "max_length": self.max_length,
                    "hidden_size": hidden_size,
                    "dtype": self.cache_dtype.__name__,
                    "model": self.model_name_or_path,
                    "tune_from": self.tune_from,
                    "hidden_state_index": self.tune_from + 1,
                    "architecture": "Qwen3",
                },
                done_f,
            )

    def _collect_prompts(self, corpus):
        dataset_name = getattr(self.config, "dataset_name", None)
        if dataset_name == "MIND":
            return self._collect_mind_prompts(corpus)
        if dataset_name == "ebnerd":
            return self._collect_ebnerd_prompts(corpus)
        if dataset_name == "Adressa":
            return self._collect_adressa_prompts(corpus)
        if dataset_name == "gossipcop":
            return self._collect_gossipcop_prompts(corpus)
        raise NotImplementedError(
            f"ONCE-DIRE-QWEN3-NAML does not support dataset_name={dataset_name!r}."
        )

    def _collect_mind_prompts(self, corpus):
        prompts = ["news article:"] * corpus.news_num
        seen = set()
        for root in [self.config.train_root, self.config.dev_root, self.config.test_root]:
            news_path = os.path.join(root, "news.tsv")
            if not os.path.exists(news_path):
                continue
            with open(news_path, "r", encoding="utf-8") as news_f:
                for line in news_f:
                    fields = line.rstrip("\n").split("\t")
                    if len(fields) < 8:
                        continue
                    news_id, category, sub_category, title, abstract = fields[:5]
                    if news_id not in corpus.news_ID_dict:
                        continue
                    index = corpus.news_ID_dict[news_id]
                    if index in seen:
                        continue
                    prompts[index] = self._build_prompt(title, abstract, category, sub_category)
                    seen.add(index)
        return prompts

    def _collect_ebnerd_prompts(self, corpus):
        import pandas as pd

        prompts = ["news article:"] * corpus.news_num
        seen = set()
        for root in [self.config.train_root, self.config.dev_root, self.config.test_root]:
            news_path = os.path.join(root, "news.parquet")
            if not os.path.exists(news_path):
                continue
            df_news = pd.read_parquet(news_path)
            for _, row in df_news.iterrows():
                news_id = self._clean_text(row.get("nid"))
                if news_id not in corpus.news_ID_dict:
                    continue
                index = corpus.news_ID_dict[news_id]
                if index in seen:
                    continue
                prompts[index] = self._build_prompt(
                    self._clean_text(row.get("title")),
                    self._clean_text(row.get("abstract")),
                    self._clean_text(row.get("category")),
                    self._clean_text(row.get("subcategory")),
                )
                seen.add(index)
        return prompts

    def _collect_gossipcop_prompts(self, corpus):
        import csv

        prompts = ["news article:"] * corpus.news_num
        seen = set()
        news_path = os.path.join(getattr(self.config, "root", "."), self.config.DATASET_ROOT, "news.csv")
        if not os.path.exists(news_path):
            return prompts
        with open(news_path, newline="", encoding="utf-8") as news_f:
            for row in csv.DictReader(news_f):
                news_id = self._clean_text(row.get("news_id"))
                if news_id not in corpus.news_ID_dict:
                    continue
                index = corpus.news_ID_dict[news_id]
                if index in seen:
                    continue
                prompts[index] = self._build_prompt(
                    self._clean_text(row.get("title")),
                    self._clean_text(row.get("text")),
                    self._clean_text(row.get("label")),
                    self._clean_text(row.get("source")),
                )
                seen.add(index)
        return prompts

    def _collect_adressa_prompts(self, corpus):
        prompts = ["news article:"] * corpus.news_num
        for news_id, row in getattr(corpus, "news_records", {}).items():
            if news_id not in corpus.news_ID_dict:
                continue
            prompts[corpus.news_ID_dict[news_id]] = self._build_prompt(
                self._clean_text(row.get("title")),
                self._clean_text(row.get("abstract")),
                self._clean_text(row.get("category")),
                self._clean_text(row.get("subcategory")),
            )
        return prompts

    @staticmethod
    def _clean_text(value) -> str:
        if value is None:
            return ""
        try:
            if isinstance(value, float) and np.isnan(value):
                return ""
        except TypeError:
            pass
        text = str(value)
        return "" if text.lower() == "nan" else text.strip()

    @staticmethod
    def _build_prompt(title: str, abstract: str, category: str, sub_category: str) -> str:
        parts = [
            "news article:",
            f"[title] {title or ''}",
            f"[abstract] {abstract or ''}",
            f"[category] {category or ''}",
            f"[subcategory] {sub_category or ''}",
        ]
        return " ".join(part.strip() for part in parts if part is not None)

    def _slice_transformer_layers(self):
        layers = getattr(self.transformer, "layers", None)
        if layers is None:
            raise AttributeError(
                "Expected a Qwen3 AutoModel with a top-level .layers ModuleList."
            )
        self.transformer.layers = nn.ModuleList(list(layers)[self.tune_from + 1:])
        if hasattr(self.transformer, "embed_tokens"):
            self.transformer.embed_tokens = None

    def _apply_lora(self):
        num_layers = self.transformer.config.num_hidden_layers
        if not self.use_lora or self.tune_from >= num_layers - 1:
            return
        try:
            from peft import LoraConfig, get_peft_model
        except ImportError as exc:
            raise ImportError(
                "peft is required when once_use_lora=true. Install peft or set once_use_lora: false."
            ) from exc

        lora_config = LoraConfig(
            inference_mode=False,
            r=int(self.lora_r),
            lora_alpha=int(self.lora_alpha),
            lora_dropout=self.lora_dropout,
            target_modules=self.lora_target_modules,
        )
        self.transformer = get_peft_model(self.transformer, lora_config)

    @staticmethod
    def _batch_indices_to_tensor(indices, batch_size=None) -> torch.Tensor:
        if torch.is_tensor(indices):
            return indices
        if isinstance(indices, np.ndarray):
            return torch.from_numpy(indices).long()
        if isinstance(indices, (list, tuple)):
            if not indices:
                return torch.empty(0, dtype=torch.long)
            if all(torch.is_tensor(item) for item in indices):
                tensors = list(indices)
                if tensors[0].dim() > 0 and (batch_size is None or tensors[0].size(0) == batch_size):
                    return torch.stack(tensors, dim=1).long()
                return torch.stack(tensors, dim=0).long()
        return torch.as_tensor(indices, dtype=torch.long)

    def encode_by_index(self, news_index) -> torch.Tensor:
        if self.hidden_cache is None or self.mask_cache is None or self.transformer is None:
            raise RuntimeError(
                "ONCE-DIRE Qwen3 cache is not prepared. "
                "set_corpus(corpus) must run before training."
            )

        news_index = self._batch_indices_to_tensor(news_index)
        output_shape = list(news_index.shape) + [self.news_embedding_dim]
        flat_index = news_index.reshape(-1).long().cpu()
        device = next(self.parameters()).device
        flat_index_np = flat_index.numpy()
        representations = []
        for start in range(0, len(flat_index_np), self.forward_page_size):
            end = min(start + self.forward_page_size, len(flat_index_np))
            batch_index = flat_index_np[start:end]
            hidden_states_np = np.array(self.hidden_cache[batch_index], copy=True)
            attention_mask_np = np.array(self.mask_cache[batch_index], copy=True)
            hidden_states = torch.from_numpy(hidden_states_np).to(device=device, dtype=self.dtype)
            attention_mask = torch.from_numpy(attention_mask_np).to(device=device, dtype=torch.long)

            outputs = self.transformer(
                inputs_embeds=hidden_states,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
            ).last_hidden_state
            outputs = self.dropout(self.projection(outputs.float()))
            representations.append(self.attention(outputs, attention_mask.float()))
        news_representation = torch.cat(representations, dim=0)
        return news_representation.view(output_shape)


class ONCE_DIRE_QWEN3_NAML(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.news_encoder = ONCEDIREQwen3NewsEncoder(config)
        self.user_attention = Attention(self.news_encoder.news_embedding_dim, config.attention_dim)
        self.click_predictor = DotProduct()
        self.model_name = config.model
        self.batch_size = config.batch_size
        self.config = config

    def initialize(self):
        self.news_encoder.initialize()
        self.user_attention.initialize()

    def set_corpus(self, corpus):
        self.news_encoder.set_corpus(corpus)

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
        history_index=None,
        news_index=None,
    ):
        if history_index is None or news_index is None:
            raise ValueError(
                "ONCE-DIRE-QWEN3-NAML requires history_index and news_index from the dataset."
            )
        batch_size = user_ID.size(0)
        history_index = self.news_encoder._batch_indices_to_tensor(history_index, batch_size)
        news_index = self.news_encoder._batch_indices_to_tensor(news_index, batch_size)
        if news_index.dim() == 1:
            news_index = news_index.unsqueeze(1)

        history_representation = self.news_encoder.encode_by_index(history_index)
        user_representation = self.user_attention(history_representation, user_history_mask)
        news_representation = self.news_encoder.encode_by_index(news_index)
        logits = self.click_predictor(
            user_representation.unsqueeze(dim=1),
            news_representation.permute(0, 2, 1),
        ).squeeze(dim=1)
        return logits

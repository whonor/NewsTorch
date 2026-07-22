import csv
import os
import pickle
import torch
import torch.nn as nn
from torchmetrics import MeanSquaredError, MeanAbsoluteError
from tqdm import tqdm

from dataset_corpus_preprocessing.EBNeRD_corpus_main import EBNeRD_Corpus, Ebnerd_DevTest_Dataset
from dataset_corpus_preprocessing.Adressa_corpus_main import (
    Adressa_Corpus,
    Adressa_DevTest_Dataset,
    Adressa_DevTest_Dataset_IPNR,
)
from dataset_corpus_preprocessing.MIND_corpus_IPNR import MIND_DevTest_Dataset_IPNR
from dataset_corpus_preprocessing.MIND_corpus_main import MIND_Corpus, MIND_DevTest_Dataset
from dataset_corpus_preprocessing.MIND_corpus_SentiDebias import MIND_Corpus_SentiDebias, MIND_DevTest_Dataset_SentiDebias
from dataset_corpus_preprocessing.MIND_corpus_SentiRec import MIND_Corpus_SentiRec, MIND_DevTest_Dataset_SentiRec
from dataset_corpus_preprocessing.Fake_MIND_corpus import Fake_MIND_Corpus, MIND_DevTest_Dataset as Fake_MIND_DevTest_Dataset
from torch.utils.data import DataLoader

from config import Config
import sys, os, os.path
import numpy as np
import json
from sklearn.metrics import roc_auc_score
import time


ONCE_DIRE_MODEL_NAMES = {"ONCE"}
CLICKBAIT_SCORE_ATTRS = (
    "news_clickbait_scores",
    "news_clickbait_score",
    "news_clickbait",
    "clickbait_scores",
    "clickbait_score",
    "clickbait",
    "news_mllm_clickbait_scores",
    "news_mllm_clickbait_score",
    "mllm_clickbait_scores",
    "mllm_clickbait_score",
    "mllm_scores",
)
CLICKBAIT_SCORE_KEYS = (
    "clickbait_score",
    "clickbait",
    "bait_score",
    "mllm_clickbait_score",
    "mllm_score",
    "score",
)
CLICKBAIT_ID_KEYS = ("news_id", "news_ID", "article_id", "article_ID", "nid", "id")
CLICKBAIT_CACHE_STEMS = (
    "clickbait",
    "clickbait_score",
    "clickbait_scores",
    "news_clickbait",
    "news_clickbait_score",
    "news_clickbait_scores",
    "mllm_clickbait",
    "mllm_clickbait_score",
    "mllm_clickbait_scores",
    "mllm_scores",
    "visual_clickbait_scores",
)


def _as_score_vector(score: torch.Tensor, batch_size: int) -> torch.Tensor:
    if score.dim() == 1:
        return score
    if score.dim() == 2 and score.size(1) == 1:
        return score.squeeze(dim=1)
    if score.numel() == batch_size:
        return score.reshape(batch_size)
    raise ValueError(f"Expected one score per sample, got score shape {tuple(score.shape)} for batch size {batch_size}.")


def _get_model_inputs(config, data_batch):
    data_batch = [item.cuda(non_blocking=True) if isinstance(item, torch.Tensor) else item for item in data_batch]

    if config.model == 'MMRec':
        user_ID = data_batch[0]
        user_category = data_batch[1]
        user_subCategory = data_batch[2]
        user_title_text = data_batch[3]
        user_title_mask = data_batch[4]
        user_history_mask = data_batch[9]
        news_category = data_batch[13]
        news_subCategory = data_batch[14]
        news_title_text = data_batch[15]
        news_title_mask = data_batch[16]
        history_image_embedding = data_batch[25]
        candidate_image_embedding = data_batch[26]

        news_feature = {
            "input_ids": news_title_text,
            "attention_mask": news_title_mask,
            "category": news_category,
            "subCategory": news_subCategory,
            "input_imgs": candidate_image_embedding.unsqueeze(1),
            "image_loc": torch.zeros(news_title_text.shape[0], 1, 5).cuda(non_blocking=True),
        }
        history_feature = {
            "input_ids": user_title_text,
            "attention_mask": user_title_mask,
            "category": user_category,
            "subCategory": user_subCategory,
            "input_imgs": history_image_embedding.unsqueeze(2),
            "image_loc": torch.zeros(user_title_text.shape[0], user_title_text.shape[1], 1, 5).cuda(non_blocking=True),
        }
        # MMRec forward signature: forward(self, news_feature, history_feature, user_history_mask, label=None, compute_loss=True)
        # compute_scores_mmrec calls: model(news_feature, history_feature, user_history_mask, None, compute_loss=False)
        return (news_feature, history_feature, user_history_mask, None, False)

    elif config.model == 'SentiRec':
        if config.dataset_name == 'MIND':
             # 0-9: user features, 10-17: news features, 18-19: sentiment, 20-21: indices
             # We need to insert 3 Nones for graph args at index 10
             args = list(data_batch[:10]) + [None, None, None] + list(data_batch[10:18]) + [data_batch[20], data_batch[21], data_batch[18], data_batch[19]]
             return tuple(args)
        else:
             return tuple(data_batch[:21] + [data_batch[23], data_batch[24], data_batch[21], data_batch[22]])

    elif config.model == 'IPNR' or config.model == 'TCCM' or config.model == 'DREAM' or config.model == 'SEIN':
        return data_batch

    elif config.model in ONCE_DIRE_MODEL_NAMES or config.model == 'PNR-LLM':
        news_category = data_batch[13].unsqueeze(dim=1)
        news_subCategory = data_batch[14].unsqueeze(dim=1)
        news_title_text = data_batch[15].unsqueeze(dim=1)
        news_title_mask = data_batch[16].unsqueeze(dim=1)
        news_title_entity = data_batch[17].unsqueeze(dim=1)
        news_content_text = data_batch[18].unsqueeze(dim=1)
        news_content_mask = data_batch[19].unsqueeze(dim=1)
        news_content_entity = data_batch[20].unsqueeze(dim=1)
        candidate_news_index = data_batch[22] if config.dataset_name in ['MIND', 'gossipcop'] else data_batch[24]
        candidate_news_index = candidate_news_index.unsqueeze(dim=1)

        data_batch = list(data_batch)
        data_batch[13] = news_category
        data_batch[14] = news_subCategory
        data_batch[15] = news_title_text
        data_batch[16] = news_title_mask
        data_batch[17] = news_title_entity
        data_batch[18] = news_content_text
        data_batch[19] = news_content_mask
        data_batch[20] = news_content_entity
        if config.dataset_name in ['MIND', 'gossipcop']:
            data_batch[22] = candidate_news_index
            return tuple(data_batch[:21] + [data_batch[21], data_batch[22]])
        data_batch[24] = candidate_news_index
        return tuple(data_batch[:21] + [data_batch[23], data_batch[24]])
        
    else:
        user_ID = data_batch[0]
        news_category = data_batch[13]
        news_subCategory = data_batch[14]
        news_title_text = data_batch[15]
        news_title_mask = data_batch[16]
        news_title_entity = data_batch[17]
        news_content_text = data_batch[18]
        news_content_mask = data_batch[19]
        news_content_entity = data_batch[20]
        if config.dataset_name in ['MIND', 'gossipcop']:
            candidate_news_index = data_batch[22]
        else:
            candidate_news_index = data_batch[24]

        news_category = news_category.unsqueeze(dim=1)
        news_subCategory = news_subCategory.unsqueeze(dim=1)
        news_title_text = news_title_text.unsqueeze(dim=1)
        news_title_mask = news_title_mask.unsqueeze(dim=1)
        news_title_entity = news_title_entity.unsqueeze(dim=1)
        news_content_text = news_content_text.unsqueeze(dim=1)
        news_content_mask = news_content_mask.unsqueeze(dim=1)
        news_content_entity = news_content_entity.unsqueeze(dim=1)
        candidate_news_index = candidate_news_index.unsqueeze(dim=1)

        # Create a copy or a new list to avoid in-place modification of data_batch
        # if it's used elsewhere, though here it seems okay.
        data_batch = list(data_batch)
        data_batch[13] = news_category
        data_batch[14] = news_subCategory
        data_batch[15] = news_title_text
        data_batch[16] = news_title_mask
        data_batch[17] = news_title_entity
        data_batch[18] = news_content_text
        data_batch[19] = news_content_mask
        data_batch[20] = news_content_entity
        if config.dataset_name in ['MIND', 'gossipcop']:
            data_batch[22] = candidate_news_index
        else:
            data_batch[24] = candidate_news_index

        if config.model == "TANR":
             return data_batch[:21]
        elif config.model == "CNE-SUE" or config.model == "DKN" or config.model == "FIM":
             return data_batch[:21]
        elif config.model == "LKPNR":
            if config.dataset_name in {'ebnerd', 'Adressa'}:
                return data_batch[:21] + data_batch[23:25]
            else:
                return data_batch
        elif config.model == "SentiDebias":
            def discretize(scores):
                labels = torch.ones_like(scores, dtype=torch.long)
                labels[scores < -0.05] = 0
                labels[scores > 0.05] = 2
                return labels

            if config.dataset_name == 'MIND':
                # 0-9: user, 10-17: news, 18: sent_hist, 19: sent_cand
                return tuple(list(data_batch[:10]) + [None, None, None] + list(data_batch[10:18]) + [discretize(data_batch[18]), discretize(data_batch[19])])
            elif config.dataset_name in {'ebnerd', 'Adressa'}:
                # Ebnerd: 0-9 user, 10-12 graph, 13-20 news, 21 hist_sent, 22 cand_sent
                return tuple(list(data_batch[:10]) + [None, None, None] + list(data_batch[13:21]) + [discretize(data_batch[21]), discretize(data_batch[22])])
        else:
            return data_batch[:21]

def compute_complexity(model, config, data_batch):
    profile = None
    try:
        from thop import profile
    except ImportError:
        print("thop is not installed. Please install it to compute complexity.")
        model.eval()
    print(f"DEBUG: model type: {type(model)}")
    print(f"DEBUG: total parameters: {sum(p.numel() for p in model.parameters())}")
    inputs = _get_model_inputs(config, data_batch)
    
    # thop requires tuple inputs
    if not isinstance(inputs, tuple) and not isinstance(inputs, list):
         inputs = (inputs,)
    else:
         inputs = tuple(inputs)

    # 1. Clean up attributes to allow thop to register its buffers
    for m in model.modules():
        if hasattr(m, "total_ops"): del m.total_ops
        if hasattr(m, "total_params"): del m.total_params

    def count_zero_ops(m, x, y):
        m.total_ops = torch.DoubleTensor([0])

    custom_ops = {
        nn.Dropout: count_zero_ops,
        nn.Dropout2d: count_zero_ops,
        nn.Dropout3d: count_zero_ops,
        nn.Identity: count_zero_ops,
    }

    flops, params = 0, 0
    if profile is not None:
        try:
            # 2. Run profile. Use return values if successful for best accuracy.
            flops, params = profile(model, inputs=inputs, custom_ops=custom_ops, verbose=False)
        except Exception as e:
            print(f"DEBUG: thop.profile crashed: {e}")
            # import traceback
            # traceback.print_exc()
    
    # 3. Fallback/Manual summation if profile was not run or crashed
    if flops == 0 or params == 0:
        for m in model.modules():
            if len(list(m.children())) == 0: # Sum only leaf modules to avoid double counting
                if hasattr(m, "total_ops"):
                    val = m.total_ops.item() if isinstance(m.total_ops, torch.Tensor) else m.total_ops
                    flops += val
    # 4. Ensure all hooks are removed even if profile crashes
    for m in model.modules():
        if hasattr(m, "_forward_hooks"):
            m._forward_hooks.clear()
        if hasattr(m, "_forward_pre_hooks"):
            m._forward_pre_hooks.clear()

    # Always use a reliable parameter count
    param_list = list(model.parameters())
    if not param_list:
        print("DEBUG: model.parameters() is EMPTY!")
        # Try to find parameters in submodules manually if top-level list is empty for some reason
        total_p = 0
        for m in model.modules():
            total_p += sum(p.numel() for p in m.parameters(recurse=False))
        params = total_p
    else:
        params = sum(p.numel() for p in param_list)
    
    print(f"DEBUG: Calculated params: {params}")
            
    return flops, params

def compute_inference_time(model, config, data_batch, repetitions=100):
    model.eval()
    inputs = _get_model_inputs(config, data_batch)
    
    # Check if inputs is a sequence to unpack
    is_sequence = isinstance(inputs, (tuple, list))
    
    # Warmup
    with torch.no_grad():
        if is_sequence:
            model(*inputs)
        else:
            model(inputs)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        
    start_time = time.time()
    with torch.no_grad():
        for _ in range(repetitions):
            if is_sequence:
                model(*inputs)
            else:
                model(inputs)
                
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    end_time = time.time()
    
    return (end_time - start_time) / repetitions


def MAE_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_score = torch.tensor(y_score)
    y_true = torch.tensor(y_true)
    mae = MeanAbsoluteError()
    score = mae(y_score, y_true)
    return score

def RMSE_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_score = torch.tensor(y_score)
    y_true = torch.tensor(y_true)
    rmse = MeanSquaredError(squared=False)
    score = rmse(y_score, y_true)
    return score

def dce_at_k(clickbait_scores, k=5):
    """
    Discounted Clickbait Exposure: discounted average valid clickbait score
    among the top-k items, weighted by 1 / log2(rank + 1).

    Missing clickbait scores are encoded as values outside [0, 1] in the corpus
    cache; those are ignored so missing annotations do not look like low exposure.
    """
    discounts = 1.0 / np.log2(np.arange(2, k + 2, dtype=np.float32))
    valid_scores = []
    valid_discounts = []
    for score, discount in zip(clickbait_scores[:k], discounts):
        try:
            score = float(score)
        except (TypeError, ValueError):
            continue
        if np.isfinite(score) and 0.0 <= score <= 1.0:
            valid_scores.append(score)
            valid_discounts.append(float(discount))
    if not valid_scores:
        return float("nan")
    return float(np.dot(valid_scores, valid_discounts) / np.sum(valid_discounts))


def _coerce_clickbait_value(value):
    if isinstance(value, dict):
        for key in CLICKBAIT_SCORE_KEYS:
            if key in value:
                return _coerce_clickbait_value(value[key])
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _clickbait_array_from_mapping(raw_scores, corpus):
    news_num = getattr(corpus, "news_num", None)
    news_id_dict = getattr(corpus, "news_ID_dict", None)
    if news_num is None:
        return None

    clickbait_scores = np.full(news_num, np.nan, dtype=np.float32)
    for raw_key, raw_value in raw_scores.items():
        index = None
        if news_id_dict is not None and raw_key in news_id_dict:
            index = news_id_dict[raw_key]
        else:
            try:
                index = int(raw_key)
            except (TypeError, ValueError):
                index = None

        if index is not None and 0 <= index < news_num:
            clickbait_scores[index] = _coerce_clickbait_value(raw_value)
    return clickbait_scores


def _clickbait_array_from_records(raw_scores, corpus):
    news_num = getattr(corpus, "news_num", None)
    news_id_dict = getattr(corpus, "news_ID_dict", None)
    if news_num is None or news_id_dict is None:
        return None

    clickbait_scores = np.full(news_num, np.nan, dtype=np.float32)
    for record in raw_scores:
        if not isinstance(record, dict):
            return None

        news_id = None
        for key in CLICKBAIT_ID_KEYS:
            if key in record:
                news_id = str(record[key])
                break
        if news_id is None or news_id not in news_id_dict:
            continue

        score = float("nan")
        for key in CLICKBAIT_SCORE_KEYS:
            if key in record:
                score = _coerce_clickbait_value(record[key])
                break
        clickbait_scores[news_id_dict[news_id]] = score
    return clickbait_scores


def _normalize_clickbait_scores(raw_scores, corpus):
    if raw_scores is None:
        return None

    news_num = getattr(corpus, "news_num", None)
    if isinstance(raw_scores, torch.Tensor):
        raw_scores = raw_scores.detach().cpu().numpy()

    if isinstance(raw_scores, np.ndarray):
        try:
            scores = raw_scores.astype(np.float32, copy=False).reshape(-1)
        except (TypeError, ValueError):
            scores = np.array([_coerce_clickbait_value(score) for score in raw_scores.reshape(-1)], dtype=np.float32)
        if news_num is None or len(scores) == news_num:
            return scores
        return None

    if isinstance(raw_scores, dict):
        return _clickbait_array_from_mapping(raw_scores, corpus)

    if isinstance(raw_scores, (list, tuple)):
        if raw_scores and isinstance(raw_scores[0], dict):
            return _clickbait_array_from_records(raw_scores, corpus)
        scores = np.array([_coerce_clickbait_value(score) for score in raw_scores], dtype=np.float32)
        if news_num is None or len(scores) == news_num:
            return scores

    return None


def _load_clickbait_csv(path, corpus):
    with open(path, newline="", encoding="utf-8") as score_f:
        return _clickbait_array_from_records(csv.DictReader(score_f), corpus)


def _load_clickbait_parquet(path, corpus):
    import pandas as pd

    raw_scores = pd.read_parquet(path).to_dict("records")
    return _clickbait_array_from_records(raw_scores, corpus)


def _load_clickbait_cache(path, corpus):
    if path.endswith(".json"):
        with open(path, "r", encoding="utf-8") as score_f:
            raw_scores = json.load(score_f)
        return _normalize_clickbait_scores(raw_scores, corpus)
    if path.endswith(".pkl"):
        with open(path, "rb") as score_f:
            raw_scores = pickle.load(score_f)
        return _normalize_clickbait_scores(raw_scores, corpus)
    if path.endswith(".csv"):
        return _load_clickbait_csv(path, corpus)
    if path.endswith(".parquet"):
        return _load_clickbait_parquet(path, corpus)
    return None


def _has_valid_clickbait_scores(clickbait_scores):
    if clickbait_scores is None:
        return False
    scores = np.asarray(clickbait_scores, dtype=np.float32)
    return bool(np.any(np.isfinite(scores) & (0.0 <= scores) & (scores <= 1.0)))


def _candidate_clickbait_cache_paths(config):
    cache_dirs = ["cache"]
    if getattr(config, "dataset_name", None) == "ebnerd":
        cache_dirs.insert(0, os.path.join("cache", "ebnerd"))
    data_root = getattr(config, "DATASET_ROOT", None)
    if data_root:
        cache_dirs.append(os.path.join("cache", data_root))

    suffixes = [getattr(config, "dataset_size", None), data_root]
    suffixes = [suffix for suffix in suffixes if suffix]
    paths = []
    for cache_dir in cache_dirs:
        for stem in CLICKBAIT_CACHE_STEMS:
            for extension in ("json", "pkl", "csv", "parquet"):
                paths.append(os.path.join(cache_dir, f"{stem}.{extension}"))
                for suffix in suffixes:
                    paths.append(os.path.join(cache_dir, f"{stem}-{suffix}.{extension}"))
                    paths.append(os.path.join(cache_dir, f"{stem}_{suffix}.{extension}"))
    return list(dict.fromkeys(paths))


def get_clickbait_scores(config, corpus):
    explicit_path = getattr(config, "clickbait_score_path", "")
    if explicit_path and os.path.exists(explicit_path):
        clickbait_scores = _load_clickbait_cache(explicit_path, corpus)
        if _has_valid_clickbait_scores(clickbait_scores):
            return clickbait_scores

    for attr in CLICKBAIT_SCORE_ATTRS:
        if hasattr(corpus, attr):
            clickbait_scores = _normalize_clickbait_scores(getattr(corpus, attr), corpus)
            if _has_valid_clickbait_scores(clickbait_scores):
                return clickbait_scores

    for path in _candidate_clickbait_cache_paths(config):
        if not os.path.exists(path):
            continue
        clickbait_scores = _load_clickbait_cache(path, corpus)
        if _has_valid_clickbait_scores(clickbait_scores):
            return clickbait_scores

    return None


def _get_candidate_news_indices(corpus, mode):
    behaviors = getattr(corpus, f"{mode}_behaviors", None)
    if behaviors is None:
        return None

    candidate_news_indices = []
    for behavior in behaviors:
        try:
            candidate_news_indices.append(int(behavior[3]))
        except (IndexError, TypeError, ValueError):
            return None
    return candidate_news_indices


def _get_candidate_engagement_values(corpus, mode):
    behaviors = getattr(corpus, f"{mode}_behaviors", None)
    if behaviors is None:
        return None, None

    read_times = []
    scroll_percentages = []
    for behavior in behaviors:
        read_times.append(behavior[6] if len(behavior) > 6 else np.nan)
        scroll_percentages.append(behavior[7] if len(behavior) > 7 else np.nan)
    return read_times, scroll_percentages


def _coerce_finite_float(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def _scroll_confidence(scroll_percentage):
    scroll_percentage = _coerce_finite_float(scroll_percentage)
    if scroll_percentage is None or scroll_percentage < 0.0:
        return None
    if scroll_percentage > 1.0:
        scroll_percentage /= 100.0
    return float(np.clip(scroll_percentage, 0.0, 1.0))


def _read_time_confidence(read_time, read_time_scale):
    read_time = _coerce_finite_float(read_time)
    if read_time is None or read_time <= 0.0 or read_time_scale is None or read_time_scale <= 0.0:
        return None
    confidence = np.log1p(read_time) / np.log1p(read_time_scale)
    return float(np.clip(confidence, 0.0, 1.0))


def _engagement_confidence(read_time, scroll_percentage, read_time_scale):
    confidences = []
    read_confidence = _read_time_confidence(read_time, read_time_scale)
    scroll_confidence = _scroll_confidence(scroll_percentage)
    if read_confidence is not None:
        confidences.append(read_confidence)
    if scroll_confidence is not None:
        confidences.append(scroll_confidence)
    if not confidences:
        return None
    return float(np.mean(confidences))


def _build_sub_scores(indices, scores, candidate_news_indices=None, candidate_read_times=None, candidate_scroll_percentages=None):
    if not indices:
        return []

    sub_scores = [[] for _ in range(indices[-1] + 1)]
    has_candidate_news_indices = (
        candidate_news_indices is not None and len(candidate_news_indices) == len(scores)
    )
    has_candidate_read_times = (
        candidate_read_times is not None and len(candidate_read_times) == len(scores)
    )
    has_candidate_scroll_percentages = (
        candidate_scroll_percentages is not None and len(candidate_scroll_percentages) == len(scores)
    )
    for i, index in enumerate(indices):
        entry = [scores[i], len(sub_scores[index]), None, np.nan, np.nan]
        if has_candidate_news_indices:
            entry[2] = candidate_news_indices[i]
        if has_candidate_read_times:
            entry[3] = candidate_read_times[i]
        if has_candidate_scroll_percentages:
            entry[4] = candidate_scroll_percentages[i]
        sub_scores[index].append(entry)
    return sub_scores


def _compute_dce_from_sub_scores(sub_scores, clickbait_scores, k=5):
    if clickbait_scores is None:
        return float("nan")

    dces = []
    for sub_score in sub_scores:
        if not sub_score:
            continue
        ranked_clickbait_scores = [
            clickbait_scores[entry[2]]
            for entry in sorted(sub_score, key=lambda x: x[0], reverse=True)
            if entry[2] is not None and 0 <= entry[2] < len(clickbait_scores)
        ]
        dces.append(dce_at_k(ranked_clickbait_scores, k))

    valid_dces = [score for score in dces if np.isfinite(score)]
    return float(np.mean(valid_dces)) if valid_dces else float("nan")


def _clicked_read_time_scale(sub_scores, truth_file_path):
    read_times = []

    with open(truth_file_path, "r", encoding="utf-8") as truth_f:
        for line_index, truth_line in enumerate(truth_f):
            impid, labels = parse_line(truth_line)
            if labels == []:
                continue

            try:
                sub_score_index = int(impid) - 1
            except (TypeError, ValueError):
                sub_score_index = line_index

            if sub_score_index < 0 or sub_score_index >= len(sub_scores):
                continue

            sub_score = sub_scores[sub_score_index]
            if not sub_score:
                continue

            for entry in sub_score:
                candidate_position = entry[1]
                if candidate_position >= len(labels) or labels[candidate_position] <= 0:
                    continue
                read_time = _coerce_finite_float(entry[3])
                if read_time is not None and read_time > 0.0:
                    read_times.append(read_time)

    if not read_times:
        return None
    return max(float(np.percentile(read_times, 95)), 1.0)


def _candidate_read_time_scale(sub_scores):
    read_times = []
    for sub_score in sub_scores:
        for entry in sub_score:
            read_time = _coerce_finite_float(entry[3])
            if read_time is not None and read_time > 0.0:
                read_times.append(read_time)

    if not read_times:
        return None
    return max(float(np.percentile(read_times, 95)), 1.0)


def _cba_gain(entry, labels, clickbait_scores, read_time_scale):
    candidate_position = entry[1]
    if candidate_position >= len(labels) or labels[candidate_position] <= 0:
        return 0.0
    if clickbait_scores is None:
        return float("nan")

    news_index = entry[2]
    if news_index is None or news_index < 0 or news_index >= len(clickbait_scores):
        return float("nan")

    clickbait_score = _coerce_finite_float(clickbait_scores[news_index])
    if clickbait_score is None or clickbait_score < 0.0 or clickbait_score > 1.0:
        return float("nan")

    q_i = _engagement_confidence(entry[3], entry[4], read_time_scale)
    if q_i is None:
        return float("nan")
    return float(q_i * (1.0 - clickbait_score))


def _visual_clickbait_label(entry, clickbait_scores, read_time_scale, clickbait_threshold, satisfaction_threshold):
    if clickbait_scores is None:
        return None

    news_index = entry[2]
    if news_index is None or news_index < 0 or news_index >= len(clickbait_scores):
        return None

    clickbait_score = _coerce_finite_float(clickbait_scores[news_index])
    if clickbait_score is None or clickbait_score < 0.0 or clickbait_score > 1.0:
        return None

    q_i = _engagement_confidence(entry[3], entry[4], read_time_scale)
    if q_i is None:
        return None

    return bool(clickbait_score >= clickbait_threshold and q_i <= satisfaction_threshold)


def _compute_cb_hr_from_sub_scores(
    sub_scores,
    clickbait_scores,
    ks=(5, 10),
    clickbait_threshold=0.5,
    satisfaction_threshold=0.5,
):
    """
    Clickbait Hit Rate@K: fraction of impressions whose top-k list contains
    at least one visual clickbait candidate.
    """
    if clickbait_scores is None:
        return {k: float("nan") for k in ks}

    clickbait_threshold = _coerce_finite_float(clickbait_threshold)
    satisfaction_threshold = _coerce_finite_float(satisfaction_threshold)
    if clickbait_threshold is None or satisfaction_threshold is None:
        return {k: float("nan") for k in ks}

    read_time_scale = _candidate_read_time_scale(sub_scores)
    cb_hrs = {k: [] for k in ks}

    for sub_score in sub_scores:
        if not sub_score:
            continue

        label_by_candidate_position = {}
        for entry in sub_score:
            visual_clickbait = _visual_clickbait_label(
                entry,
                clickbait_scores,
                read_time_scale,
                clickbait_threshold,
                satisfaction_threshold,
            )
            if visual_clickbait is not None:
                label_by_candidate_position[entry[1]] = visual_clickbait

        if not label_by_candidate_position:
            continue

        ranked_entries = sorted(sub_score, key=lambda x: x[0], reverse=True)
        for k in ks:
            top_k_entries = ranked_entries[:k]
            cb_hrs[k].append(float(any(
                label_by_candidate_position.get(entry[1], False)
                for entry in top_k_entries
            )))

    return {
        k: float(np.mean(scores)) if scores else float("nan")
        for k, scores in cb_hrs.items()
    }


def _compute_cb_hr_metrics(config, sub_scores, clickbait_scores):
    cb_hrs = _compute_cb_hr_from_sub_scores(
        sub_scores,
        clickbait_scores,
        (5, 10),
        getattr(config, "cbhr_clickbait_threshold", 0.5),
        getattr(config, "cbhr_satisfaction_threshold", 0.5),
    )
    return cb_hrs[5], cb_hrs[10]


def _cba_ndcg_at_k(ranked_gains, k):
    if not ranked_gains:
        return float("nan")

    gains = np.nan_to_num(np.asarray(ranked_gains, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if not np.any(gains > 0.0):
        return float("nan")

    actual_gains = gains[:k]
    discounts = np.log2(np.arange(len(actual_gains), dtype=np.float32) + 2.0)
    actual = float(np.sum(actual_gains / discounts))

    ideal_gains = np.sort(gains)[::-1][:k]
    ideal_discounts = np.log2(np.arange(len(ideal_gains), dtype=np.float32) + 2.0)
    ideal = float(np.sum(ideal_gains / ideal_discounts))
    return actual / ideal if ideal > 0.0 else float("nan")


def _compute_cba_ndcg_from_sub_scores(sub_scores, truth_file_path, clickbait_scores, ks=(5, 10)):
    if clickbait_scores is None:
        return {k: float("nan") for k in ks}

    read_time_scale = _clicked_read_time_scale(sub_scores, truth_file_path)
    cba_ndcgs = {k: [] for k in ks}

    with open(truth_file_path, "r", encoding="utf-8") as truth_f:
        for line_index, truth_line in enumerate(truth_f):
            impid, labels = parse_line(truth_line)
            if labels == []:
                continue

            try:
                sub_score_index = int(impid) - 1
            except (TypeError, ValueError):
                sub_score_index = line_index

            if sub_score_index < 0 or sub_score_index >= len(sub_scores):
                continue

            sub_score = sub_scores[sub_score_index]
            if not sub_score:
                continue

            gain_by_candidate_position = {
                entry[1]: _cba_gain(entry, labels, clickbait_scores, read_time_scale)
                for entry in sub_score
            }
            positive_gains = [
                gain for gain in gain_by_candidate_position.values()
                if np.isfinite(gain) and gain > 0.0
            ]
            if not positive_gains:
                continue

            ranked_entries = sorted(sub_score, key=lambda x: x[0], reverse=True)
            ranked_gains = [gain_by_candidate_position.get(entry[1], 0.0) for entry in ranked_entries]
            for k in ks:
                score = _cba_ndcg_at_k(ranked_gains, k)
                if np.isfinite(score):
                    cba_ndcgs[k].append(score)

    return {
        k: float(np.mean(scores)) if scores else float("nan")
        for k, scores in cba_ndcgs.items()
    }


def recall_at_k(y_true, y_score, k=5):
    """
    Compute Recall@k using numpy for efficiency.
    """
    # Sort indices by scores in descending order
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order)
    # Top-k indices in y_true
    y_true_k = y_true[:k]
    # Calculate recall@k
    return np.sum(y_true_k) / np.sum(y_true)

def hit_at_k(y_true, y_score, k=5):
    """
    Compute Hit@k using numpy for efficiency.
    """
    # Sort indices by scores in descending order
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order)
    # Top-k indices in y_true
    y_true_k = y_true[:k]
    # Check if there's at least one hit in top-k
    return 1.0 if np.any(y_true_k) else 0.0

def precision_at_k(y_true, y_score, k=5):
    """
    Compute Precision@k using numpy for efficiency.
    """
    # Sort indices by scores in descending order
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order)
    # Top-k indices in y_true
    y_true_k = y_true[:k]
    # Calculate precision@k
    return np.sum(y_true_k) / k

def dcg_score(y_true, y_score, k=10):
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])
    gains = 2 ** y_true - 1
    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gains / discounts)


def ndcg_score(y_true, y_score, k=10):
    best = dcg_score(y_true, y_true, k)
    actual = dcg_score(y_true, y_score, k)
    return actual / best


def mrr_score(y_true, y_score):
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order)
    rr_score = y_true / (np.arange(len(y_true)) + 1)
    return np.sum(rr_score) / np.sum(y_true)


def parse_line(l):
    # impid, ranks = l.strip('\n').split()
    impid, ranks = l.strip().split()
    # print("[DEBUG] parse_line got ranks:", ranks)
    try:
        ranks = json.loads(ranks)
    except json.JSONDecodeError:
        # try to fix the error format：如 [1,,4,,5,,3,,2] → extract [1, 4, 5, 3, 2]
        content = ranks.strip('[] ')
        # transform '1, 4, 5, 3, 2' to ['1', '4', '5', '3', '2']
        # remove empty parts and convert to integers
        parts = [p for p in content.split(',') if p.strip().isdigit()]
        ranks_fixed = [int(p.strip()) for p in parts]
        ranks = ranks_fixed

    return impid, ranks


def scoring(truth_f, sub_f):
    aucs = []
    mrrs = []
    ndcg5s = []
    ndcg10s = []

    maes = []
    rmses = []
    recall5s = []
    recall10s = []
    hit5s = []
    hit10s = []
    precision5s = []
    precision10s = []

    line_index = 1
    for lt in truth_f:
        ls = sub_f.readline()
        impid, labels = parse_line(lt)

        # ignore masked impressions
        if labels == []:
            continue

        if ls == '':
            # empty line: filled with 0 ranks
            sub_impid = impid
            sub_ranks = [1] * len(labels)
        else:
            try:
                sub_impid, sub_ranks = parse_line(ls)
            except:
                raise ValueError("line-{}: Invalid Input Format!".format(line_index))

        if sub_impid != impid:
            raise ValueError("line-{}: Inconsistent Impression Id {} and {}".format(
                line_index,
                sub_impid,
                impid
            ))

        lt_len = float(len(labels))

        y_true = np.array(labels, dtype='float32')
        y_score = []
        for rank in sub_ranks:
            score_rslt = 1. / rank
            if score_rslt < 0 or score_rslt > 1:
                raise ValueError("Line-{}: score_rslt should be int from 0 to {}".format(
                    line_index,
                    lt_len
                ))
            y_score.append(score_rslt)

        auc = roc_auc_score(y_true, y_score)
        mrr = mrr_score(y_true, y_score)
        ndcg5 = ndcg_score(y_true, y_score, 5)
        ndcg10 = ndcg_score(y_true, y_score, 10)

        aucs.append(auc)
        mrrs.append(mrr)
        ndcg5s.append(ndcg5)
        ndcg10s.append(ndcg10)

        mae = MAE_score(y_true, y_score)
        rmse = RMSE_score(y_true, y_score)
        recall5 = recall_at_k(y_true, y_score, 5)
        recall10 = recall_at_k(y_true, y_score, 10)
        precision5 = precision_at_k(y_true, y_score, 5)
        precision10 = precision_at_k(y_true, y_score, 10)
        hit5 = hit_at_k(y_true, y_score, 5)
        hit10 = hit_at_k(y_true, y_score, 10)

        maes.append(mae)
        rmses.append(rmse)
        recall5s.append(recall5)
        recall10s.append(recall10)
        hit5s.append(hit5)
        hit10s.append(hit10)
        precision5s.append(precision5)
        precision10s.append(precision10)

        line_index += 1

    return (np.mean(aucs), np.mean(mrrs), np.mean(ndcg5s), np.mean(ndcg10s),
            np.mean(maes), np.mean(rmses), np.mean(recall5s), np.mean(recall10s),
            np.mean(hit5s), np.mean(hit10s), np.mean(precision5s), np.mean(precision10s))


def compute_scores_IPNR(config: Config, model: nn.Module, mind_corpus: MIND_Corpus, batch_size: int, mode: str, result_file: str, dataset: str):
    assert mode in ['dev', 'test'], 'mode must be chosen from \'dev\' or \'test\''
    eval_dataset = (
        Adressa_DevTest_Dataset_IPNR(mind_corpus, mode)
        if config.dataset_name == 'Adressa'
        else MIND_DevTest_Dataset_IPNR(mind_corpus, mode)
    )
    dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    indices = (mind_corpus.dev_indices if mode == 'dev' else mind_corpus.test_indices)
    scores = torch.zeros([len(indices)]).cuda()
    index = 0
    torch.cuda.empty_cache()
    model.eval()
    with torch.no_grad():
        for data_batch in tqdm(dataloader):
            data_batch = [item.cuda(non_blocking=True) if isinstance(item, torch.Tensor) else item for item in data_batch]
            (user_ID, _, _, _, _, _, _, _, _, _, _, _, _,
             news_category, news_subCategory, news_title_text, news_title_mask, _, news_content_text, news_content_mask, _, news_concept_text, news_concept_mask) = data_batch

            batch_size = user_ID.size(0)
            news_category = news_category.unsqueeze(dim=1)
            news_subCategory = news_subCategory.unsqueeze(dim=1)
            news_title_text = news_title_text.unsqueeze(dim=1)
            news_title_mask = news_title_mask.unsqueeze(dim=1)
            news_content_text = news_content_text.unsqueeze(dim=1)
            news_content_mask = news_content_mask.unsqueeze(dim=1)
            news_concept_text = news_concept_text.unsqueeze(dim=1)
            news_concept_mask = news_concept_mask.unsqueeze(dim=1)

            data_batch[13] = news_category
            data_batch[14] = news_subCategory
            data_batch[15] = news_title_text
            data_batch[16] = news_title_mask
            data_batch[18] = news_content_text
            data_batch[19] = news_content_mask
            data_batch[21] = news_concept_text
            data_batch[22] = news_concept_mask

            scores[index: index+batch_size] = model(*data_batch).squeeze(dim=1) # [batch_size]
            index += batch_size
    scores = scores.tolist()
    sub_scores = _build_sub_scores(indices, scores)
    with open(result_file, 'w', encoding='utf-8') as result_f:
        for i, sub_score in enumerate(sub_scores):
            sub_score.sort(key=lambda x: x[0], reverse=True)
            result = [0 for _ in range(len(sub_score))]
            for j in range(len(sub_score)):
                result[sub_score[j][1]] = j + 1
            result_f.write(('' if i == 0 else '\n') + str(i + 1) + ' ' + str(result).replace(' ', ''))
    if dataset != 'submission' or mode != 'test':
        truth_file_path = "./cache/" + mode + '/ref/truth-%s.txt' % config.DATASET_ROOT
        with open(truth_file_path, 'r', encoding='utf-8') as truth_f, open(result_file, 'r', encoding='utf-8') as result_f:
            auc, mrr, ndcg5, ndcg10, mae, rmse, recall5, recall10, hit5, hit10, precision5, precision10 = scoring(truth_f, result_f)
        return auc, mrr, ndcg5, ndcg10, mae, rmse, recall5, recall10, hit5, hit10, precision5, precision10
    else:
        return (None,) * 12


def compute_scores(config: Config, model: nn.Module, corpus, batch_size: int, mode: str, result_file: str, dataset_size: str):
    assert mode in ['dev', 'test'], 'mode must be chosen from \'dev\' or \'test\''
    if config.dataset_name == 'ebnerd':
        if corpus is None:
            if config.model in ['CPRS', 'DREAM']:
                from dataset_corpus_preprocessing.EBNeRD_corpus_CPRS import EBNeRD_Corpus as EBNeRD_Corpus_CPRS
                corpus = EBNeRD_Corpus_CPRS(config)
            elif config.model == 'TCCM':
                from dataset_corpus_preprocessing.EBNeRD_corpus_TCCM import EBNeRD_Corpus as EBNeRD_Corpus_TCCM
                corpus = EBNeRD_Corpus_TCCM(config)
            elif config.model == 'SEIN':
                from dataset_corpus_preprocessing.EBNeRD_corpus_SEIN import EBNeRD_Corpus as EBNeRD_Corpus_SEIN
                corpus = EBNeRD_Corpus_SEIN(config)
            else:
                corpus = EBNeRD_Corpus(config)
        
        if config.model in ['CPRS', 'DREAM']:
            from dataset_corpus_preprocessing.EBNeRD_corpus_CPRS import Ebnerd_DevTest_Dataset as Ebnerd_DevTest_Dataset_CPRS
            devtest_dataset = Ebnerd_DevTest_Dataset_CPRS(corpus, mode)
        elif config.model == 'TCCM':
            from dataset_corpus_preprocessing.EBNeRD_corpus_TCCM import Ebnerd_DevTest_Dataset as Ebnerd_DevTest_Dataset_TCCM
            devtest_dataset = Ebnerd_DevTest_Dataset_TCCM(corpus, mode)
        elif config.model == 'SEIN':
            from dataset_corpus_preprocessing.EBNeRD_corpus_SEIN import Ebnerd_DevTest_Dataset as Ebnerd_DevTest_Dataset_SEIN
            devtest_dataset = Ebnerd_DevTest_Dataset_SEIN(corpus, mode)
        else:
            from dataset_corpus_preprocessing.EBNeRD_corpus_main import Ebnerd_DevTest_Dataset
            devtest_dataset = Ebnerd_DevTest_Dataset(corpus, mode)
    elif config.dataset_name == 'MIND':
        if config.model == 'SentiRec':
            corpus = MIND_Corpus_SentiRec(config)
            devtest_dataset = MIND_DevTest_Dataset_SentiRec(corpus, mode)
        elif config.model == 'SentiDebias':
            corpus = MIND_Corpus_SentiDebias(config)
            devtest_dataset = MIND_DevTest_Dataset_SentiDebias(corpus, mode)
        else:
            corpus = MIND_Corpus(config)
            devtest_dataset = MIND_DevTest_Dataset(corpus, mode)
    elif config.dataset_name == 'gossipcop':
        if corpus is None:
            corpus = Fake_MIND_Corpus(config)
        devtest_dataset = Fake_MIND_DevTest_Dataset(corpus, mode)
    elif config.dataset_name == 'Adressa':
        if corpus is None:
            corpus = Adressa_Corpus(config)
        devtest_dataset = Adressa_DevTest_Dataset(corpus, mode)
    dataloader = DataLoader(devtest_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=batch_size // 16, pin_memory=True)
    indices = (corpus.dev_indices if mode == 'dev' else corpus.test_indices)
    scores = torch.zeros([len(indices)]).cuda()
    index = 0
    model.eval()

    with torch.no_grad():
        for data_batch in tqdm(dataloader):
            data_batch = [item.cuda(non_blocking=True) if isinstance(item, torch.Tensor) else item for item in data_batch]
            user_ID = data_batch[0]
            batch_size = user_ID.size(0)

            if config.model == "SentiRec" and config.dataset_name == 'MIND':
                # Custom handling for SentiRec on MIND to avoid incorrect unsqueezing
                args = list(data_batch[:10]) + [None, None, None] + list(data_batch[10:18]) + [data_batch[20], data_batch[21]]
                kwargs = {
                    'history_sentiment': data_batch[18],
                    'candidate_sentiment': data_batch[19]
                }
                scores[index: index + batch_size] = model(*args, **kwargs)[0]
                index += batch_size
                continue
            
            if config.model == "SentiDebias":
                def discretize(scores):
                    labels = torch.ones_like(scores, dtype=torch.long)
                    labels[scores < -0.05] = 0
                    labels[scores > 0.05] = 2
                    return labels

                if config.dataset_name == 'MIND':
                     # 0-9: user, 10-17: news, 18: sent_hist, 19: sent_cand
                     args = list(data_batch[:10]) + [None, None, None] + list(data_batch[10:18]) + [discretize(data_batch[18]), discretize(data_batch[19])]
                     scores[index: index + batch_size] = model(*args)[1]
                elif config.dataset_name in {'ebnerd', 'Adressa'}:
                     # Ebnerd: 0-9 user, 10-12 graph, 13-20 news, 21 hist_sent, 22 cand_sent
                     args = list(data_batch[:10]) + [None, None, None] + list(data_batch[13:21]) + [discretize(data_batch[21]), discretize(data_batch[22])]
                     scores[index: index + batch_size] = model(*args)[1]
                index += batch_size
                continue

            news_category = data_batch[13]
            news_subCategory = data_batch[14]
            news_title_text = data_batch[15]
            news_title_mask = data_batch[16]
            news_title_entity = data_batch[17]
            news_content_text = data_batch[18]
            news_content_mask = data_batch[19]
            news_content_entity = data_batch[20]
            if config.dataset_name in ['MIND', 'gossipcop']:
                candidate_news_index = data_batch[22]
            else:
                candidate_news_index = data_batch[24]

            batch_size = user_ID.size(0)
            news_category = news_category.unsqueeze(dim=1)
            news_subCategory = news_subCategory.unsqueeze(dim=1)
            news_title_text = news_title_text.unsqueeze(dim=1)
            news_title_mask = news_title_mask.unsqueeze(dim=1)
            news_title_entity = news_title_entity.unsqueeze(dim=1)
            news_content_text = news_content_text.unsqueeze(dim=1)
            news_content_mask = news_content_mask.unsqueeze(dim=1)
            news_content_entity = news_content_entity.unsqueeze(dim=1)
            candidate_news_index = candidate_news_index.unsqueeze(dim=1)

            data_batch[13] = news_category
            data_batch[14] = news_subCategory
            data_batch[15] = news_title_text
            data_batch[16] = news_title_mask
            data_batch[17] = news_title_entity
            data_batch[18] = news_content_text
            data_batch[19] = news_content_mask
            data_batch[20] = news_content_entity
            if config.dataset_name in ['MIND', 'gossipcop']:
                data_batch[22] = candidate_news_index
            else:
                data_batch[24] = candidate_news_index

            if config.model == "TANR":
                score, _ = model(*data_batch[:21])
                scores[index: index + batch_size] = score
            elif config.model == "CNE-SUE" or config.model == "DKN" or config.model == "FIM":
                scores[index: index + batch_size] = model(*data_batch[:21]).squeeze(1)
            elif config.model == "LKPNR":
                if config.dataset_name in {'ebnerd', 'Adressa'}:
                    model_args = data_batch[:21] + data_batch[23:25]
                    scores[index: index + batch_size] = model(*model_args).squeeze(dim=1)
                else:
                    scores[index: index + batch_size] = model(*data_batch).squeeze(dim=1)
            elif config.model == "SentiRec":
                if config.dataset_name == 'MIND':
                    # data_batch has 24 elements. 0-9: user, 10-17: news, 18-19: sent, 20-21: idx, 22-23: img
                    args = list(data_batch[:10]) + [None, None, None] + list(data_batch[10:18]) + [data_batch[20], data_batch[21]]
                    kwargs = {
                        'history_sentiment': data_batch[18],
                        'candidate_sentiment': data_batch[19]
                    }
                else:
                    args = data_batch[:21] + [data_batch[23], data_batch[24]]
                    kwargs = {
                        'history_sentiment': data_batch[21],
                        'candidate_sentiment': data_batch[22]
                    }
                scores[index: index + batch_size] = model(*args, **kwargs)[0]

            elif config.model in ["CPRS", "DREAM"]:
                scores[index: index + batch_size] = model(*data_batch)[0].squeeze(dim=1)
            elif config.model == "TCCM":
                out = model(*data_batch)
                if isinstance(out, tuple):
                    out = out[0]
                scores[index: index + batch_size] = out
            elif config.model == "SEIN":
                out = model(*data_batch)
                if isinstance(out, tuple):
                    out = out[0]
                scores[index: index + batch_size] = out
            elif config.model in ONCE_DIRE_MODEL_NAMES:
                if config.dataset_name in ['MIND', 'gossipcop']:
                    args = data_batch[:21] + [data_batch[21], data_batch[22]]
                else:
                    args = data_batch[:21] + [data_batch[23], data_batch[24]]
                scores[index: index + batch_size] = _as_score_vector(model(*args), batch_size)
            elif config.model == "PNR-LLM":
                if config.dataset_name in ['MIND', 'gossipcop']:
                    args = data_batch[:21] + [data_batch[21], data_batch[22]]
                else:
                    args = data_batch[:21] + [data_batch[23], data_batch[24]]
                scores[index: index + batch_size] = _as_score_vector(model(*args), batch_size)
            else:
                scores[index: index + batch_size] = model(*data_batch[:21])
            index += batch_size
    scores = scores.tolist()
    sub_scores = _build_sub_scores(indices, scores)
    with open(result_file, 'w', encoding='utf-8') as result_f:
        for i, sub_score in enumerate(sub_scores):
            sub_score.sort(key=lambda x: x[0], reverse=True)
            result = [0 for _ in range(len(sub_score))]
            for j in range(len(sub_score)):
                result[sub_score[j][1]] = j + 1
            result_f.write(('' if i == 0 else '\n') + str(i + 1) + ' ' + str(result).replace(' ', ''))
    if dataset_size != 'submission' or mode != 'test':
        truth_file_path = config.data_path + '/' + mode + '/ref/truth-%s.txt' % config.DATASET_ROOT
        with open(truth_file_path, 'r', encoding='utf-8') as truth_f, open(result_file, 'r', encoding='utf-8') as result_f:
            auc, mrr, ndcg5, ndcg10, mae, rmse, recall5, recall10, hit5, hit10, precision5, precision10 = scoring(truth_f, result_f)
        return auc, mrr, ndcg5, ndcg10, mae, rmse, recall5, recall10, hit5, hit10, precision5, precision10
    else:
        return (None,) * 12


def compute_scores_mmrec(config: Config, model: nn.Module, corpus, batch_size: int, mode: str, result_file: str, dataset_size: str):
    assert mode in ['dev', 'test'], 'mode must be chosen from \'dev\' or \'test\''
    if config.dataset_name == 'ebnerd':
        corpus = EBNeRD_Corpus(config)
        dataset = Ebnerd_DevTest_Dataset(corpus, mode)
    elif config.dataset_name == 'Adressa':
        if corpus is None:
            corpus = Adressa_Corpus(config)
        dataset = Adressa_DevTest_Dataset(corpus, mode)
    elif config.dataset_name == 'MIND':
        corpus = MIND_Corpus(config)
        dataset = MIND_DevTest_Dataset(corpus, mode)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                            num_workers=batch_size // 16, pin_memory=True)
    indices = (corpus.dev_indices if mode == 'dev' else corpus.test_indices)
    scores = torch.zeros([len(indices)]).cuda()
    index = 0
    model.eval()

    with torch.no_grad():
        for data_batch in tqdm(dataloader):
            data_batch = [item.cuda(non_blocking=True) if isinstance(item, torch.Tensor) else item for item in data_batch]
            user_ID = data_batch[0]
            user_category = data_batch[1]
            user_subCategory = data_batch[2]
            user_title_text = data_batch[3]
            user_title_mask = data_batch[4]
            user_history_mask = data_batch[9]
            news_category = data_batch[13]
            news_subCategory = data_batch[14]
            news_title_text = data_batch[15]
            news_title_mask = data_batch[16]
            history_image_embedding = data_batch[25]
            candidate_image_embedding = data_batch[26]

            news_feature = {
                "input_ids": news_title_text,
                "attention_mask": news_title_mask,
                "category": news_category,
                "subCategory": news_subCategory,
                "input_imgs": candidate_image_embedding.unsqueeze(1),
                "image_loc": torch.zeros(news_title_text.shape[0], 1, 5).cuda(non_blocking=True),
            }
            history_feature = {
                "input_ids": user_title_text,
                "attention_mask": user_title_mask,
                "category": user_category,
                "subCategory": user_subCategory,
                "input_imgs": history_image_embedding.unsqueeze(2),
                "image_loc": torch.zeros(user_title_text.shape[0], user_title_text.shape[1], 1, 5).cuda(non_blocking=True),
            }
            batch_size = user_ID.size(0)
            
            score = model(news_feature, history_feature, user_history_mask, None, compute_loss=False)
            scores[index: index + batch_size] = score.squeeze(1)

            index += batch_size
    scores = scores.tolist()
    sub_scores = _build_sub_scores(indices, scores)
    with open(result_file, 'w', encoding='utf-8') as result_f:
        for i, sub_score in enumerate(sub_scores):
            sub_score.sort(key=lambda x: x[0], reverse=True)
            result = [0 for _ in range(len(sub_score))]
            for j in range(len(sub_score)):
                result[sub_score[j][1]] = j + 1
            result_f.write(('' if i == 0 else '\n') + str(i + 1) + ' ' + str(result).replace(' ', ''))
    if dataset_size != 'submission' or mode != 'test':
        truth_file_path = config.data_path + '/' + mode + '/ref/truth-%s.txt' % config.DATASET_ROOT
        with open(truth_file_path, 'r', encoding='utf-8') as truth_f, open(result_file, 'r', encoding='utf-8') as result_f:
            auc, mrr, ndcg5, ndcg10, mae, rmse, recall5, recall10, hit5, hit10, precision5, precision10 = scoring(truth_f, result_f)
        return auc, mrr, ndcg5, ndcg10, mae, rmse, recall5, recall10, hit5, hit10, precision5, precision10
    else:
        return (None,) * 12


def get_run_index(result_dir: str):
    assert os.path.exists(result_dir), 'result directory does not exist'
    max_index = 0
    for result_file in os.listdir(result_dir):
        if result_file.strip()[0] == '#' and result_file.strip()[-4:] == '-dev':
            index = int(result_file.strip()[1:-4])
            max_index = max(index, max_index)
    with open(result_dir + '/#' + str(max_index + 1) + '-dev', 'w', encoding='utf-8') as result_f:
        pass
    return max_index + 1


class AvgMetric:
    def __init__(self, auc, mrr, ndcg5, ndcg10):
        self.auc = auc
        self.mrr = mrr
        self.ndcg5 = ndcg5
        self.ndcg10 = ndcg10
        self.avg = (self.auc + self.mrr + (self.ndcg5 + self.ndcg10) / 2) / 3

    def __gt__(self, value):
        return self.avg > value.avg

    def __ge__(self, value):
        return self.avg >= value.avg

    def __lt__(self, value):
        return self.avg < value.avg

    def __le__(self, value):
        return self.avg <= value.avg

    def __str__(self):
        return '%.4f\nAUC = %.4f\nMRR = %.4f\nnDCG@5 = %.4f\nnDCG@10 = %.4f' % (self.avg, self.auc, self.mrr, self.ndcg5, self.ndcg10)

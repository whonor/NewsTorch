import math
import os
import pickle

import numpy as np
import pandas as pd

from dataset_corpus_preprocessing.EBNeRD_corpus_main import (
    EBNeRD_Corpus as BaseEBNeRD_Corpus,
    Ebnerd_DevTest_Dataset as BaseEbnerd_DevTest_Dataset,
    Ebnerd_Train_Dataset as BaseEbnerd_Train_Dataset,
)
from dataset_corpus_preprocessing.ebnerd_behavior_utils import as_list


def _safe_timestamp(value):
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp):
        return None
    return timestamp


def _align_history_timestamps(history, full_articles, full_timestamps):
    history_keys = [str(item).strip() for item in as_list(history)]
    article_keys = [str(item).strip() for item in as_list(full_articles)]
    timestamps = as_list(full_timestamps)

    if not history_keys:
        return []
    if len(article_keys) != len(timestamps):
        usable_len = min(len(article_keys), len(timestamps))
        article_keys = article_keys[:usable_len]
        timestamps = timestamps[:usable_len]

    history_len = len(history_keys)
    if article_keys[-history_len:] == history_keys:
        return timestamps[-history_len:]

    # Fallback for duplicated article ids or differently truncated prepared data:
    # match the requested history as a subsequence from the end of the raw history.
    aligned = [None] * history_len
    source_idx = len(article_keys) - 1
    for target_idx in range(history_len - 1, -1, -1):
        target = history_keys[target_idx]
        while source_idx >= 0 and article_keys[source_idx] != target:
            source_idx -= 1
        if source_idx < 0:
            break
        aligned[target_idx] = timestamps[source_idx]
        source_idx -= 1

    if all(value is not None for value in aligned):
        return aligned
    return timestamps[-history_len:]


def _stage_ids_from_timestamps(timestamps, anchor_time, max_history_num, stage_num, window_days):
    selected = as_list(timestamps)[-max_history_num:]
    anchor = _safe_timestamp(anchor_time)
    if anchor is None:
        valid_times = [_safe_timestamp(value) for value in selected]
        valid_times = [value for value in valid_times if value is not None]
        anchor = max(valid_times) if valid_times else None

    stage_ids = []
    for value in selected:
        timestamp = _safe_timestamp(value)
        if timestamp is None or anchor is None:
            stage_ids.append(-1)
            continue

        age_seconds = max((anchor - timestamp).total_seconds(), 0.0)
        window_index = int(math.floor(age_seconds / (window_days * 86400.0)))
        stage_ids.append(max(stage_num - 1 - min(window_index, stage_num - 1), 0))

    return stage_ids + [-1] * (max_history_num - len(stage_ids))


class EBNeRD_Corpus(BaseEBNeRD_Corpus):
    """EB-NeRD corpus variant that exposes timestamp-window stage ids for SEIN."""

    def __init__(self, config):
        super().__init__(config)
        self.sein_stage_num = getattr(config, "sein_stage_num", 5)
        self.sein_temporal_window_days = getattr(config, "sein_temporal_window_days", 7)
        if self.sein_temporal_window_days <= 0:
            raise ValueError("sein_temporal_window_days must be positive")
        self._attach_sein_history_stage_ids()

    def _stage_cache_file(self):
        dataset_root = str(self.config.DATASET_ROOT).replace(os.sep, "_").replace("/", "_")
        window = str(self.sein_temporal_window_days).replace(".", "p")
        return os.path.join(
            "cache",
            "ebnerd",
            f"sein_history_stages-{dataset_root}-{self.config.dataset_size}-"
            f"h{self.max_history_num}-s{self.sein_stage_num}-w{window}d.pkl",
        )

    def _attach_sein_history_stage_ids(self):
        cache_file = self._stage_cache_file()
        if os.path.exists(cache_file):
            with open(cache_file, "rb") as f:
                stage_data = pickle.load(f)
            if not self._is_valid_stage_cache(stage_data):
                stage_data = self._build_stage_cache()
                with open(cache_file, "wb") as f:
                    pickle.dump(stage_data, f)
        else:
            stage_data = self._build_stage_cache()
            os.makedirs(os.path.dirname(cache_file), exist_ok=True)
            with open(cache_file, "wb") as f:
                pickle.dump(stage_data, f)

        self._append_stage_ids(self.train_behaviors, stage_data["train"])
        self._append_stage_ids(self.dev_behaviors, stage_data["dev"])
        self._append_stage_ids(self.test_behaviors, stage_data["test"])

    def _build_stage_cache(self):
        return {
            "train": self._build_split_stage_ids(self.config.train_root),
            "dev": self._build_split_stage_ids(self.config.dev_root),
            "test": self._build_split_stage_ids(self.config.test_root),
        }

    def _is_valid_stage_cache(self, stage_data):
        if not isinstance(stage_data, dict):
            return False

        expected_lengths = {
            "train": self._split_behavior_count(self.config.train_root),
            "dev": self._split_behavior_count(self.config.dev_root),
            "test": self._split_behavior_count(self.config.test_root),
        }
        for split, expected_len in expected_lengths.items():
            if split not in stage_data or len(stage_data[split]) != expected_len:
                return False
        return True

    @staticmethod
    def _split_behavior_count(split_root):
        return len(pd.read_parquet(os.path.join(split_root, "behaviors.parquet")))

    def _build_split_stage_ids(self, split_root):
        behaviors = pd.read_parquet(os.path.join(split_root, "behaviors.parquet"))
        histories = pd.read_parquet(os.path.join(split_root, "history.parquet"))
        history_by_user = {
            str(row["user_id"]): (row["article_id_fixed"], row["impression_time_fixed"])
            for _, row in histories.iterrows()
        }

        split_stage_ids = []
        for _, row in behaviors.iterrows():
            user_history = history_by_user.get(str(row["uid"]))
            if user_history is None:
                timestamps = []
            else:
                full_articles, full_timestamps = user_history
                timestamps = _align_history_timestamps(row["history"], full_articles, full_timestamps)

            split_stage_ids.append(
                _stage_ids_from_timestamps(
                    timestamps,
                    row.get("time"),
                    self.max_history_num,
                    self.sein_stage_num,
                    self.sein_temporal_window_days,
                )
            )
        return split_stage_ids

    @staticmethod
    def _append_stage_ids(behaviors, split_stage_ids):
        for behavior in behaviors:
            behavior_index_value = behavior[5] if isinstance(behavior[4], (list, tuple, np.ndarray)) else behavior[4]
            behavior_index = int(behavior_index_value)
            if behavior_index >= len(split_stage_ids):
                raise ValueError(
                    "SEIN history stage cache is incompatible with the prepared behaviors: "
                    f"behavior_index={behavior_index}, stage_count={len(split_stage_ids)}. "
                    "Regenerate the EB-NeRD split or remove the stale SEIN stage cache."
                )
            behavior.append(split_stage_ids[behavior_index])


class Ebnerd_Train_Dataset(BaseEbnerd_Train_Dataset):
    def __getitem__(self, index):
        item = super().__getitem__(index)
        stage_ids = np.array(self.train_behaviors[index][-1], dtype=np.int64)
        return item + (stage_ids,)


class Ebnerd_DevTest_Dataset(BaseEbnerd_DevTest_Dataset):
    def __getitem__(self, index):
        item = super().__getitem__(index)
        stage_ids = np.array(self.behaviors[index][-1], dtype=np.int64)
        return item + (stage_ids,)

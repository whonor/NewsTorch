import numpy as np
import pandas as pd


def as_list(value):
    if isinstance(value, (list, tuple, np.ndarray, pd.Series)):
        return list(value)
    if value is None:
        return []
    try:
        if pd.isna(value):
            return []
    except (TypeError, ValueError):
        pass
    return [value]


def as_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def pad_history_read_time(value, max_history_num):
    values = [as_float(item, 0.0) for item in as_list(value)]
    values = values[-max_history_num:]
    return values + [0.0] * (max_history_num - len(values))


def candidate_read_time(value, position=None):
    values = as_list(value)
    if position is not None and position < len(values):
        return as_float(values[position], np.nan)
    if len(values) == 1:
        return as_float(values[0], np.nan)
    return np.nan


def candidate_scroll_percentage(value, position=None):
    return candidate_read_time(value, position)

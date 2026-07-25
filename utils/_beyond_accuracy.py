"""Beyond-accuracy metrics adapted from NewsRecLib's aspect metrics.

Reference implementation:
https://github.com/andreeaiana/newsreclib/blob/main/newsreclib/metrics/functional.py
"""

from collections import OrderedDict
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np


__all__ = [
    "aspect_diversity",
    "aspect_personalization",
    "category_metrics_from_corpus",
    "diversity",
    "generalized_jaccard",
    "personalization",
]


def _validate_inputs(
    scores: Sequence[float], aspects: Sequence[int], num_classes: int
) -> Tuple[np.ndarray, np.ndarray]:
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    aspects = np.asarray(aspects, dtype=np.int64).reshape(-1)
    if scores.shape != aspects.shape:
        raise ValueError("scores and aspects must have the same shape")
    if num_classes <= 0:
        raise ValueError("num_classes must be a positive integer")
    if aspects.size and (np.any(aspects < 0) or np.any(aspects >= num_classes)):
        raise ValueError("aspect labels must be in [0, num_classes)")
    return scores, aspects


def _resolve_top_k(top_k: Optional[int], size: int) -> int:
    if top_k is None:
        return size
    if not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("top_k must be a positive integer or None")
    return min(top_k, size)


def aspect_diversity(
    scores: Sequence[float],
    aspects: Sequence[int],
    num_classes: int,
    top_k: Optional[int] = None,
) -> float:
    """Return normalized entropy of aspect labels in the top-k ranking.

    This follows NewsRecLib's ``diversity`` functional metric. A score of one
    denotes a uniform distribution over every aspect class; zero denotes a
    single-aspect ranking.
    """

    scores, aspects = _validate_inputs(scores, aspects, num_classes)
    top_k = _resolve_top_k(top_k, scores.size)
    if top_k == 0:
        return float("nan")
    if num_classes == 1:
        return 0.0

    ranked_aspects = aspects[np.argsort(-scores, kind="stable")[:top_k]]
    counts = np.bincount(ranked_aspects, minlength=num_classes).astype(np.float64)
    probabilities = counts / counts.sum()
    nonzero_probabilities = probabilities[probabilities > 0.0]
    entropy = -np.sum(nonzero_probabilities * np.log(nonzero_probabilities))
    return float(entropy / np.log(float(num_classes)))


def generalized_jaccard(predicted: Sequence[float], target: Sequence[float]) -> float:
    """Return NewsRecLib's generalized Jaccard similarity."""

    predicted = np.asarray(predicted, dtype=np.float64).reshape(-1)
    target = np.asarray(target, dtype=np.float64).reshape(-1)
    if predicted.shape != target.shape:
        raise ValueError("predicted and target distributions must have the same shape")
    denominator = np.maximum(predicted, target).sum()
    if denominator == 0.0:
        return float("nan")
    return float(np.minimum(predicted, target).sum() / denominator)


def aspect_personalization(
    scores: Sequence[float],
    candidate_aspects: Sequence[int],
    clicked_aspects: Sequence[int],
    num_classes: int,
    top_k: Optional[int] = None,
) -> float:
    """Compare top-k and clicked-history aspect counts with generalized Jaccard."""

    scores, candidate_aspects = _validate_inputs(
        scores, candidate_aspects, num_classes
    )
    clicked_aspects = np.asarray(clicked_aspects, dtype=np.int64).reshape(-1)
    if clicked_aspects.size and (
        np.any(clicked_aspects < 0) or np.any(clicked_aspects >= num_classes)
    ):
        raise ValueError("clicked aspect labels must be in [0, num_classes)")

    top_k = _resolve_top_k(top_k, scores.size)
    if top_k == 0:
        return float("nan")
    ranked_aspects = candidate_aspects[
        np.argsort(-scores, kind="stable")[:top_k]
    ]
    predicted_counts = np.bincount(
        ranked_aspects, minlength=num_classes
    ).astype(np.float64)
    clicked_counts = np.bincount(
        clicked_aspects, minlength=num_classes
    ).astype(np.float64)
    return generalized_jaccard(predicted_counts, clicked_counts)


def diversity(
    preds: Sequence[float],
    target: Sequence[int],
    num_classes: int,
    top_k: Optional[int] = None,
) -> float:
    """NewsRecLib-compatible functional name for aspect diversity."""

    return aspect_diversity(preds, target, num_classes, top_k)


def personalization(
    preds: Sequence[float],
    predicted_aspects: Sequence[int],
    target_aspects: Sequence[int],
    num_classes: int,
    top_k: Optional[int] = None,
) -> float:
    """NewsRecLib-compatible functional name for aspect personalization."""

    return aspect_personalization(
        preds, predicted_aspects, target_aspects, num_classes, top_k
    )


def _mean_or_nan(values: Iterable[float]) -> float:
    finite_values = [float(value) for value in values if np.isfinite(value)]
    return float(np.mean(finite_values)) if finite_values else float("nan")


def category_metrics_from_corpus(
    corpus,
    mode: str,
    impression_indices: Sequence[int],
    scores: Sequence[float],
    top_ks: Sequence[int] = (5, 10),
    aspect: str = "category",
) -> Tuple[float, ...]:
    """Compute NewsRecLib-style aspect metrics over corpus impressions.

    The returned order is all diversity values followed by all personalization
    values, each in ``top_ks`` order. ``aspect`` selects either corpus category
    or subcategory metadata.
    """

    if mode not in {"dev", "test"}:
        raise ValueError("mode must be 'dev' or 'test'")
    aspect_attributes = {
        "category": ("news_category", "category_dict"),
        "subcategory": ("news_subCategory", "subCategory_dict"),
    }
    if aspect not in aspect_attributes:
        raise ValueError("aspect must be 'category' or 'subcategory'")
    top_ks = tuple(int(k) for k in top_ks)
    if not top_ks or any(k <= 0 for k in top_ks):
        raise ValueError("top_ks must contain positive integers")

    news_aspect_attr, aspect_dict_attr = aspect_attributes[aspect]
    behaviors = getattr(corpus, f"{mode}_behaviors", None)
    news_aspects = getattr(corpus, news_aspect_attr, None)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    impression_indices = np.asarray(impression_indices).reshape(-1)
    empty_result = (float("nan"),) * (2 * len(top_ks))
    if behaviors is None or news_aspects is None:
        return empty_result
    if len(behaviors) != scores.size or scores.size != impression_indices.size:
        return empty_result

    news_aspects = np.asarray(news_aspects, dtype=np.int64).reshape(-1)
    aspect_dict = getattr(corpus, aspect_dict_attr, None)
    num_classes = len(aspect_dict) if aspect_dict is not None else 0
    if news_aspects.size:
        num_classes = max(num_classes, int(news_aspects.max()) + 1)
    if num_classes <= 0:
        return empty_result

    grouped = OrderedDict()
    for behavior, impression_index, score in zip(
        behaviors, impression_indices, scores
    ):
        try:
            candidate_index = int(behavior[3])
            impression_index = int(impression_index)
        except (IndexError, TypeError, ValueError):
            return empty_result
        if candidate_index < 0 or candidate_index >= news_aspects.size:
            return empty_result
        group = grouped.setdefault(
            impression_index,
            {"scores": [], "candidate_aspects": [], "clicked_aspects": None},
        )
        group["scores"].append(float(score))
        group["candidate_aspects"].append(int(news_aspects[candidate_index]))

        if group["clicked_aspects"] is None:
            try:
                history_indices = np.asarray(behavior[1], dtype=np.int64).reshape(-1)
                history_mask = np.asarray(behavior[2]).reshape(-1).astype(bool)
            except (IndexError, TypeError, ValueError):
                return empty_result
            valid_size = min(history_indices.size, history_mask.size)
            history_indices = history_indices[:valid_size][history_mask[:valid_size]]
            history_indices = history_indices[
                (history_indices >= 0) & (history_indices < news_aspects.size)
            ]
            group["clicked_aspects"] = news_aspects[history_indices]

    diversities = {k: [] for k in top_ks}
    personalizations = {k: [] for k in top_ks}
    for group in grouped.values():
        for k in top_ks:
            diversities[k].append(
                aspect_diversity(
                    group["scores"], group["candidate_aspects"], num_classes, k
                )
            )
            personalizations[k].append(
                aspect_personalization(
                    group["scores"],
                    group["candidate_aspects"],
                    group["clicked_aspects"],
                    num_classes,
                    k,
                )
            )

    return tuple(_mean_or_nan(diversities[k]) for k in top_ks) + tuple(
        _mean_or_nan(personalizations[k]) for k in top_ks
    )

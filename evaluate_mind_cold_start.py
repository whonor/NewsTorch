import argparse
import copy
import csv
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Set, Tuple

import numpy as np
import torch

from config import Config
from dataset_corpus_preprocessing.MIND_corpus_main import MIND_Corpus
from general_runner import get_model_classes
from utils._evaluation import compute_scores


UNSUPPORTED_MODELS = {
    "IPNR",
    "SentiDebias",
    "SentiRec",
    "MMRec",
    "CNE-SUE",
    "CNRCL",
}


@dataclass
class RawImpression:
    user_id: str
    history: List[str]
    candidates: List[Tuple[str, int]]


def parse_candidate(candidate: str) -> Tuple[str, int]:
    news_id, label = candidate.rsplit("-", 1)
    return news_id, int(label)


def read_news_ids(path: str) -> Set[str]:
    news_ids = set()
    with open(path, "r", encoding="utf-8") as news_f:
        for line in news_f:
            news_ids.add(line.split("\t", 1)[0])
    return news_ids


def read_train_evidence(train_root: str) -> Tuple[Set[str], Set[str]]:
    train_users = set()
    interacted_articles = set()
    with open(os.path.join(train_root, "behaviors.tsv"), "r", encoding="utf-8") as behaviors_f:
        for line in behaviors_f:
            _, user_id, _, history, impressions = line.rstrip("\n").split("\t")
            train_users.add(user_id)
            if history:
                interacted_articles.update(history.split())
            for impression in impressions.split():
                news_id, _ = parse_candidate(impression)
                interacted_articles.add(news_id)
    return train_users, interacted_articles


def read_raw_impressions(behaviors_path: str) -> List[RawImpression]:
    impressions = []
    with open(behaviors_path, "r", encoding="utf-8") as behaviors_f:
        for line in behaviors_f:
            _, user_id, _, history, candidates = line.rstrip("\n").split("\t")
            impressions.append(
                RawImpression(
                    user_id=user_id,
                    history=history.split() if history else [],
                    candidates=[parse_candidate(candidate) for candidate in candidates.split()],
                )
            )
    return impressions


def write_truth_file(path: str, raw_impressions: Sequence[RawImpression], selected_indices: Sequence[int]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as truth_f:
        for output_index, raw_index in enumerate(selected_indices):
            labels = [label for _, label in raw_impressions[raw_index].candidates]
            truth_f.write(
                ("" if output_index == 0 else "\n")
                + str(output_index + 1)
                + " "
                + str(labels).replace(" ", "")
            )


def pad_history(corpus: MIND_Corpus, history_ids: Iterable[str]) -> Tuple[List[int], np.ndarray]:
    history = [corpus.news_ID_dict[news_id] for news_id in history_ids if news_id in corpus.news_ID_dict]
    history = history[: corpus.max_history_num]
    padding_num = max(0, corpus.max_history_num - len(history))
    history_index = history + [0] * padding_num
    history_mask = np.zeros([corpus.max_history_num], dtype=np.float32)
    history_mask[: len(history)] = 1.0
    return history_index, history_mask


def build_candidate_rows_by_impression(corpus: MIND_Corpus):
    rows_by_impression = defaultdict(list)
    for row in corpus.test_behaviors:
        rows_by_impression[row[4]].append(row)
    return rows_by_impression


def build_subset_corpus(
    base_corpus: MIND_Corpus,
    raw_impressions: Sequence[RawImpression],
    selected_indices: Sequence[int],
    history_mode: str,
    shots: int = 0,
) -> MIND_Corpus:
    rows_by_impression = build_candidate_rows_by_impression(base_corpus)
    subset = copy.copy(base_corpus)
    subset.test_behaviors = []
    subset.test_indices = []

    for output_index, raw_index in enumerate(selected_indices):
        raw_impression = raw_impressions[raw_index]
        if history_mode == "zero":
            history_index, history_mask = pad_history(base_corpus, [])
        elif history_mode == "fewshot":
            history_index, history_mask = pad_history(base_corpus, raw_impression.history[:shots])
        elif history_mode == "original":
            history_index = history_mask = None
        else:
            raise ValueError(f"Unknown history mode: {history_mode}")

        for row in rows_by_impression[raw_index]:
            new_row = list(row)
            if history_mode != "original":
                new_row[1] = history_index
                new_row[2] = history_mask
            new_row[4] = output_index
            subset.test_behaviors.append(new_row)
            subset.test_indices.append(output_index)

    return subset


def unique_candidate_articles(raw_impressions: Sequence[RawImpression], selected_indices: Sequence[int]) -> Set[str]:
    return {
        news_id
        for raw_index in selected_indices
        for news_id, _ in raw_impressions[raw_index].candidates
    }


def unique_users(raw_impressions: Sequence[RawImpression], selected_indices: Sequence[int]) -> Set[str]:
    return {raw_impressions[raw_index].user_id for raw_index in selected_indices}


def positive_cold_articles(
    raw_impressions: Sequence[RawImpression],
    selected_indices: Sequence[int],
    cold_articles: Set[str],
) -> Set[str]:
    return {
        news_id
        for raw_index in selected_indices
        for news_id, label in raw_impressions[raw_index].candidates
        if label == 1 and news_id in cold_articles
    }


def evaluate_subset(
    config: Config,
    model: torch.nn.Module,
    base_corpus: MIND_Corpus,
    raw_impressions: Sequence[RawImpression],
    selected_indices: Sequence[int],
    output_dir: str,
    scenario: str,
    history_mode: str,
    shots: int = 0,
):
    subset_corpus = build_subset_corpus(
        base_corpus=base_corpus,
        raw_impressions=raw_impressions,
        selected_indices=selected_indices,
        history_mode=history_mode,
        shots=shots,
    )
    truth_path = os.path.join(output_dir, f"{scenario}.truth.txt")
    result_path = os.path.join(output_dir, f"{scenario}.ranks.txt")
    write_truth_file(truth_path, raw_impressions, selected_indices)
    metrics = compute_scores(
        config,
        model,
        subset_corpus,
        config.batch_size,
        "test",
        result_path,
        config.dataset_size,
        truth_file_path=truth_path,
    )
    return metrics[:4]


def load_model(config: Config, corpus: MIND_Corpus):
    model_classes = get_model_classes()
    if config.model not in model_classes:
        raise ValueError(f"Unknown model: {config.model}")
    if config.model in UNSUPPORTED_MODELS:
        unsupported = ", ".join(sorted(UNSUPPORTED_MODELS))
        raise ValueError(
            f"Cold-start masking currently supports base MIND candidate-ranking models only. "
            f"Unsupported models: {unsupported}."
        )

    model = model_classes[config.model](config)
    if hasattr(model, "set_corpus"):
        model.set_corpus(corpus)

    checkpoint = torch.load(config.test_model_path, map_location=torch.device("cpu"))
    model_state_dict = checkpoint[config.model]
    if list(model_state_dict.keys())[0].startswith("module."):
        model_state_dict = {key[7:]: value for key, value in model_state_dict.items()}
    model.load_state_dict(model_state_dict)
    model.cuda()
    model.eval()
    return model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="", help="Directory for cold-start ranks, truth files, and CSV results.")
    parser.add_argument("--shots", nargs="+", type=int, default=[1, 2, 3], help="Few-shot history lengths.")
    return parser.parse_known_args()[0]


def main():
    cold_args = parse_args()
    config = Config()
    if config.dataset_name != "MIND" or config.DATASET_ROOT != "MIND-small" or config.dataset_size != "small":
        raise ValueError(
            "This evaluator is intentionally MIND-small only. Use "
            "--dataset_name=MIND --DATASET_ROOT=MIND-small --dataset_size=small."
        )
    if not config.test_model_path:
        raise ValueError("--test_model_path is required.")

    output_dir = cold_args.output_dir or os.path.join("cache", "cold_start", "MIND-small", config.model)
    os.makedirs(output_dir, exist_ok=True)

    train_news_ids = read_news_ids(os.path.join(config.train_root, "news.tsv"))
    train_users, train_interacted_articles = read_train_evidence(config.train_root)
    raw_impressions = read_raw_impressions(os.path.join(config.test_root, "behaviors.tsv"))
    cold_articles = {
        news_id
        for impression in raw_impressions
        for news_id, _ in impression.candidates
        if news_id not in train_news_ids and news_id not in train_interacted_articles
    }

    new_user_indices = [
        index for index, impression in enumerate(raw_impressions) if impression.user_id not in train_users
    ]
    new_article_indices = [
        index
        for index, impression in enumerate(raw_impressions)
        if any(label == 1 and news_id in cold_articles for news_id, label in impression.candidates)
    ]

    base_corpus = MIND_Corpus(config)
    model = load_model(config, base_corpus)

    rows = []
    scenarios = [("new_user_zero", new_user_indices, "zero", 0)]
    scenarios.extend(
        (f"new_user_{shot}_shot", new_user_indices, "fewshot", shot)
        for shot in cold_args.shots
    )
    scenarios.append(("new_article", new_article_indices, "original", 0))

    for scenario, selected_indices, history_mode, shots in scenarios:
        if not selected_indices:
            print(f"Skipping {scenario}: no matching impressions.")
            continue
        auc, mrr, ndcg5, ndcg10 = evaluate_subset(
            config=config,
            model=model,
            base_corpus=base_corpus,
            raw_impressions=raw_impressions,
            selected_indices=selected_indices,
            output_dir=output_dir,
            scenario=scenario,
            history_mode=history_mode,
            shots=shots,
        )
        row = {
            "scenario": scenario,
            "shots": shots,
            "AUC": auc,
            "MRR": mrr,
            "nDCG@5": ndcg5,
            "nDCG@10": ndcg10,
            "users": len(unique_users(raw_impressions, selected_indices)),
            "articles": len(unique_candidate_articles(raw_impressions, selected_indices)),
            "cold_articles": len(positive_cold_articles(raw_impressions, selected_indices, cold_articles)),
            "impressions": len(selected_indices),
        }
        rows.append(row)
        print(
            f"{scenario}: AUC={auc:.4f}, MRR={mrr:.4f}, "
            f"nDCG@5={ndcg5:.4f}, nDCG@10={ndcg10:.4f}, "
            f"users={row['users']}, articles={row['articles']}, "
            f"cold_articles={row['cold_articles']}, impressions={row['impressions']}"
        )

    csv_path = os.path.join(output_dir, "cold_start_results.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as csv_f:
        fieldnames = [
            "scenario",
            "shots",
            "AUC",
            "MRR",
            "nDCG@5",
            "nDCG@10",
            "users",
            "articles",
            "cold_articles",
            "impressions",
        ]
        writer = csv.DictWriter(csv_f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote cold-start results to {csv_path}")


if __name__ == "__main__":
    main()

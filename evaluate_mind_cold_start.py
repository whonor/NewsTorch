import argparse
import copy
import csv
import gc
import os
import sys
import traceback
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

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


@dataclass(frozen=True)
class ModelRun:
    model: str
    seed: int
    run_name: str
    checkpoint: str


RESULT_FIELDS = [
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
    "model",
    "seed",
    "run",
    "checkpoint",
]


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
    rows_by_impression=None,
) -> MIND_Corpus:
    if rows_by_impression is None:
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
    rows_by_impression=None,
):
    subset_corpus = build_subset_corpus(
        base_corpus=base_corpus,
        raw_impressions=raw_impressions,
        selected_indices=selected_indices,
        history_mode=history_mode,
        shots=shots,
        rows_by_impression=rows_by_impression,
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
    parser.add_argument(
        "--output_dir",
        default="",
        help="Directory for cold-start ranks, truth files, and CSV results.",
    )
    parser.add_argument("--shots", nargs="+", type=int, default=[1, 2, 3], help="Few-shot history lengths.")
    parser.add_argument(
        "--all_models",
        action="store_true",
        help=(
            "Evaluate one available checkpoint for every supported model. This is also "
            "the default when neither --model nor --test_model_path is supplied."
        ),
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Evaluate one available checkpoint for each listed model.",
    )
    parser.add_argument(
        "--checkpoint_root",
        default=os.path.join("cache", "best_models", "MIND-small"),
        help="Root containing MODEL/#N/MODEL checkpoints.",
    )
    parser.add_argument(
        "--checkpoint_seed",
        "--seed",
        dest="checkpoint_seed",
        type=int,
        default=None,
        help=(
            "In batch mode, start searching at this seed (#N is treated as seed N-1) "
            "and use the next available checkpoint. By default, start at seed 0."
        ),
    )
    return parser.parse_known_args()[0]


def checkpoint_seed(run_name: str) -> Optional[int]:
    if not run_name.startswith("#") or not run_name[1:].isdigit():
        return None
    run_number = int(run_name[1:])
    return run_number - 1 if run_number > 0 else None


def find_model_run(checkpoint_root: str, model: str, seed: Optional[int] = None) -> Optional[ModelRun]:
    model_dir = os.path.join(checkpoint_root, model)
    if not os.path.isdir(model_dir):
        return None

    starting_seed = 0 if seed is None else seed
    runs = []
    for run_name in os.listdir(model_dir):
        run_seed = checkpoint_seed(run_name)
        if run_seed is None or run_seed < starting_seed:
            continue
        checkpoint = os.path.join(model_dir, run_name, model)
        if os.path.isfile(checkpoint):
            runs.append(ModelRun(model, run_seed, run_name, checkpoint))

    if not runs:
        return None
    return min(runs, key=lambda run: run.seed)


def discover_model_runs(
    checkpoint_root: str,
    requested_models: Optional[Sequence[str]],
    seed: Optional[int],
) -> Tuple[List[ModelRun], List[Dict[str, object]]]:
    model_classes = get_model_classes()
    if requested_models is None:
        if not os.path.isdir(checkpoint_root):
            raise ValueError(f"Checkpoint root does not exist: {checkpoint_root}")
        requested_models = sorted(
            name
            for name in os.listdir(checkpoint_root)
            if os.path.isdir(os.path.join(checkpoint_root, name))
        )

    runs = []
    skipped = []
    seen = set()
    for model in requested_models:
        if model in seen:
            continue
        seen.add(model)
        if model not in model_classes:
            skipped.append({"model": model, "reason": "unknown model"})
            continue
        if model in UNSUPPORTED_MODELS:
            skipped.append({"model": model, "reason": "unsupported cold-start corpus"})
            continue

        run = find_model_run(checkpoint_root, model, seed)
        if run is None:
            starting_seed = 0 if seed is None else seed
            skipped.append(
                {
                    "model": model,
                    "reason": f"no checkpoint at or after seed {starting_seed}",
                }
            )
            continue
        runs.append(run)
    return runs, skipped


def config_for_run(model_run: ModelRun) -> Config:
    original_argv = sys.argv
    sys.argv = original_argv + [
        "--dataset_name=MIND",
        "--DATASET_ROOT=MIND-small",
        "--dataset_size=small",
        f"--model={model_run.model}",
        f"--seed={model_run.seed}",
        f"--test_model_path={model_run.checkpoint}",
        "--mode=test",
    ]
    try:
        return Config()
    finally:
        sys.argv = original_argv


def validate_config(config: Config) -> None:
    if config.dataset_name != "MIND" or config.DATASET_ROOT != "MIND-small" or config.dataset_size != "small":
        raise ValueError(
            "This evaluator is intentionally MIND-small only. Use "
            "--dataset_name=MIND --DATASET_ROOT=MIND-small --dataset_size=small."
        )
    if not config.test_model_path:
        raise ValueError("--test_model_path is required.")


def read_cold_start_inputs(config: Config):
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
    return raw_impressions, cold_articles, new_user_indices, new_article_indices


def corpus_cache_key(config: Config) -> Tuple[object, ...]:
    return (
        config.DATASET_ROOT,
        config.dataset_size,
        config.word_threshold,
        config.tokenizer,
        config.max_history_num,
        config.max_title_length,
        config.max_abstract_length,
        config.word_embedding_dim,
        config.entity_embedding_dim,
        config.context_embedding_dim,
    )


def bind_corpus_dimensions(config: Config, corpus: MIND_Corpus) -> None:
    config.user_num = len(corpus.user_ID_dict)
    config.news_num = corpus.news_num
    config.category_num = len(corpus.category_dict)
    config.subCategory_num = len(corpus.subCategory_dict)
    config.vocabulary_size = len(corpus.word_dict)
    config.entity_size = len(corpus.entity_dict)


def get_base_corpus(
    config: Config,
    corpus_cache: Dict[Tuple[object, ...], MIND_Corpus],
) -> MIND_Corpus:
    cache_key = corpus_cache_key(config)
    if cache_key not in corpus_cache:
        corpus_cache[cache_key] = MIND_Corpus(config)
    else:
        bind_corpus_dimensions(config, corpus_cache[cache_key])
    corpus_cache[cache_key].model = config.model
    return corpus_cache[cache_key]


def write_csv(path: str, rows: Sequence[Dict[str, object]], fieldnames: Sequence[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as csv_f:
        writer = csv.DictWriter(csv_f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def evaluate_model_run(
    config: Config,
    model_run: ModelRun,
    base_corpus: MIND_Corpus,
    raw_impressions: Sequence[RawImpression],
    cold_articles: Set[str],
    new_user_indices: Sequence[int],
    new_article_indices: Sequence[int],
    shots_to_evaluate: Sequence[int],
    output_dir: str,
    rows_by_impression,
) -> List[Dict[str, object]]:
    os.makedirs(output_dir, exist_ok=True)
    model = None
    try:
        model = load_model(config, base_corpus)
        rows = []
        scenarios = [("new_user_zero", new_user_indices, "zero", 0)]
        scenarios.extend(
            (f"new_user_{shot}_shot", new_user_indices, "fewshot", shot)
            for shot in shots_to_evaluate
        )
        scenarios.append(("new_article", new_article_indices, "original", 0))

        for scenario, selected_indices, history_mode, shots in scenarios:
            if not selected_indices:
                print(f"Skipping {config.model}/{scenario}: no matching impressions.")
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
                rows_by_impression=rows_by_impression,
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
                "model": config.model,
                "seed": model_run.seed,
                "run": model_run.run_name,
                "checkpoint": model_run.checkpoint,
            }
            rows.append(row)
            print(
                f"{config.model}/{scenario}: AUC={auc:.4f}, MRR={mrr:.4f}, "
                f"nDCG@5={ndcg5:.4f}, nDCG@10={ndcg10:.4f}, "
                f"users={row['users']}, articles={row['articles']}, "
                f"cold_articles={row['cold_articles']}, impressions={row['impressions']}"
            )
        write_csv(os.path.join(output_dir, "cold_start_results.csv"), rows, RESULT_FIELDS)
        return rows
    finally:
        if model is not None:
            del model
        gc.collect()
        torch.cuda.empty_cache()


def explicit_model_run(config: Config) -> ModelRun:
    run_name = os.path.basename(os.path.dirname(config.test_model_path))
    return ModelRun(
        model=config.model,
        seed=config.seed,
        run_name=run_name,
        checkpoint=config.test_model_path,
    )


def main():
    cold_args = parse_args()
    if cold_args.all_models and cold_args.models is not None:
        raise ValueError("Use either --all_models or --models, not both.")

    explicit_single_model = any(
        argument == option or argument.startswith(option + "=")
        for argument in sys.argv[1:]
        for option in ("--model", "--test_model_path")
    )
    batch_mode = (
        cold_args.all_models
        or cold_args.models is not None
        or not explicit_single_model
    )
    if batch_mode:
        requested_models = None if cold_args.all_models else cold_args.models
        model_runs, skipped = discover_model_runs(
            cold_args.checkpoint_root,
            requested_models,
            cold_args.checkpoint_seed,
        )
        if not model_runs:
            reasons = "; ".join(f"{item['model']}: {item['reason']}" for item in skipped)
            raise ValueError(f"No evaluable model checkpoints found. {reasons}")
        output_root = cold_args.output_dir or os.path.join("cache", "cold_start", "MIND-small")
        print("Selected one checkpoint per model:")
        for model_run in model_runs:
            print(
                f"  {model_run.model}: seed={model_run.seed}, "
                f"run={model_run.run_name}, checkpoint={model_run.checkpoint}"
            )
        for item in skipped:
            print(f"  Skipping {item['model']}: {item['reason']}")
    else:
        config = Config()
        validate_config(config)
        model_runs = [explicit_model_run(config)]
        skipped = []
        output_root = cold_args.output_dir or os.path.join(
            "cache", "cold_start", "MIND-small", config.model
        )

    os.makedirs(output_root, exist_ok=True)
    all_rows = []
    failures = []
    corpus_cache = {}
    rows_by_impression_cache = {}
    cold_start_inputs = None
    input_roots = None

    for model_run in model_runs:
        try:
            current_config = (
                config
                if not batch_mode
                else config_for_run(model_run)
            )
            validate_config(current_config)
            current_roots = (current_config.train_root, current_config.test_root)
            if input_roots is not None and current_roots != input_roots:
                raise ValueError("All batch models must use the same train and test roots.")
            if cold_start_inputs is None:
                cold_start_inputs = read_cold_start_inputs(current_config)
                input_roots = current_roots

            raw_impressions, cold_articles, new_user_indices, new_article_indices = cold_start_inputs
            base_corpus = get_base_corpus(current_config, corpus_cache)
            corpus_identity = id(base_corpus)
            if corpus_identity not in rows_by_impression_cache:
                rows_by_impression_cache[corpus_identity] = build_candidate_rows_by_impression(base_corpus)
            model_output_dir = (
                os.path.join(output_root, model_run.model)
                if batch_mode
                else output_root
            )
            rows = evaluate_model_run(
                config=current_config,
                model_run=model_run,
                base_corpus=base_corpus,
                raw_impressions=raw_impressions,
                cold_articles=cold_articles,
                new_user_indices=new_user_indices,
                new_article_indices=new_article_indices,
                shots_to_evaluate=cold_args.shots,
                output_dir=model_output_dir,
                rows_by_impression=rows_by_impression_cache[corpus_identity],
            )
            all_rows.extend(rows)
            if batch_mode:
                write_csv(
                    os.path.join(output_root, "cold_start_results.csv"),
                    all_rows,
                    RESULT_FIELDS,
                )
        except Exception as error:
            if not batch_mode:
                raise
            traceback.print_exc()
            failures.append(
                {
                    "model": model_run.model,
                    "seed": model_run.seed,
                    "run": model_run.run_name,
                    "checkpoint": model_run.checkpoint,
                    "error": str(error),
                }
            )

    results_path = os.path.join(output_root, "cold_start_results.csv")
    write_csv(results_path, all_rows, RESULT_FIELDS)
    if batch_mode:
        write_csv(
            os.path.join(output_root, "skipped_models.csv"),
            skipped,
            ["model", "reason"],
        )
        write_csv(
            os.path.join(output_root, "failed_models.csv"),
            failures,
            ["model", "seed", "run", "checkpoint", "error"],
        )
    print(
        f"Wrote {len(all_rows)} cold-start result rows from "
        f"{len(model_runs) - len(failures)} model(s) to {results_path}"
    )
    if skipped or failures:
        print(f"Skipped {len(skipped)} model(s); {len(failures)} model(s) failed.")


if __name__ == "__main__":
    main()

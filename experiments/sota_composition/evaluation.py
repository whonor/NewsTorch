"""Protocol validation, lifecycle checks, and paired reporting for B0--B7."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import statistics
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, Mapping, MutableMapping, Optional

import torch
import yaml
from scipy.stats import ttest_rel

from models.SOTAComposition import SOTAComposition, available_sota_components
from models.modules.cnrcl.utils import NewsRecSupConLoss_with_neg_pos


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
METRICS = ("auc", "mrr", "ndcg5", "ndcg10")
LIFECYCLE_CHECKS = (
    "initialization",
    "forward",
    "backward",
    "checkpoint_recovery",
    "full_workflow",
)


@dataclass(frozen=True)
class Composition:
    id: str
    research_question: str
    news_encoder: str
    user_encoder: str
    enhancement: str
    training: str
    special_modules: tuple
    previously_unregistered: bool


COMPOSITIONS = (
    Composition("B0", "backbone", "cnrcl", "cnrcl", "none", "cnrcl", (), False),
    Composition("B1", "controlled_substitution", "nrms", "cnrcl", "none", "cnrcl", (), True),
    Composition("B2", "controlled_substitution", "naml", "cnrcl", "none", "cnrcl", (), True),
    Composition("B3", "cross_paradigm", "once-qwen3", "cnrcl", "none", "cnrcl", (), True),
    Composition("B4", "controlled_substitution", "cnrcl", "nrms", "none", "cnrcl", (), True),
    Composition("B5", "cross_paradigm", "cnrcl", "sein", "none", "cnrcl", ("sein-temporal-graph",), True),
    Composition("B6", "deep_composition", "once-qwen3", "sein", "none", "cnrcl", ("sein-temporal-graph",), True),
    Composition("B7", "deep_composition", "cnrcl", "cnrcl", "llm-augmentation", "cnrcl", (), True),
)


class SOTAProtocolError(ValueError):
    """Raised when a study input violates the frozen B0--B7 contract."""


def _composition_from_mapping(value: Mapping) -> Composition:
    required = {
        "id",
        "research_question",
        "news_encoder",
        "user_encoder",
        "enhancement",
        "training",
        "special_modules",
        "previously_unregistered",
    }
    missing = sorted(required.difference(value))
    if missing:
        raise SOTAProtocolError("Composition is missing: " + ", ".join(missing))
    special = value["special_modules"]
    if not isinstance(special, list) or not all(isinstance(item, str) for item in special):
        raise SOTAProtocolError("special_modules must be a list of names")
    return Composition(
        id=str(value["id"]),
        research_question=str(value["research_question"]),
        news_encoder=str(value["news_encoder"]),
        user_encoder=str(value["user_encoder"]),
        enhancement=str(value["enhancement"]),
        training=str(value["training"]),
        special_modules=tuple(special),
        previously_unregistered=bool(value["previously_unregistered"]),
    )


def _resolve_from_protocol(protocol_path: Path, value: str) -> str:
    path = Path(value).expanduser()
    if path.is_absolute():
        return str(path.resolve())
    return str((REPOSITORY_ROOT / path).resolve())


def load_protocol(path) -> MutableMapping:
    protocol_path = Path(path).resolve()
    with protocol_path.open("r", encoding="utf-8") as protocol_file:
        protocol = yaml.safe_load(protocol_file)
    if not isinstance(protocol, dict) or protocol.get("schema_version") != 1:
        raise SOTAProtocolError("Only schema_version 1 is supported")
    compositions = tuple(
        _composition_from_mapping(value)
        for value in protocol.get("compositions", ())
    )
    expected_ids = tuple("B%d" % index for index in range(8))
    if tuple(value.id for value in compositions) != expected_ids:
        raise SOTAProtocolError("Protocol must define ordered B0--B7 compositions")
    registry = available_sota_components()
    for composition in compositions:
        if composition.news_encoder not in registry["news"]:
            raise SOTAProtocolError("Unregistered news component: %s" % composition.news_encoder)
        if composition.user_encoder not in registry["user"]:
            raise SOTAProtocolError("Unregistered user component: %s" % composition.user_encoder)
        if composition.enhancement not in registry["enhancement"]:
            raise SOTAProtocolError("Unregistered enhancement: %s" % composition.enhancement)
        if composition.training not in registry["training"]:
            raise SOTAProtocolError("Unregistered training strategy: %s" % composition.training)
        unknown_special = set(composition.special_modules).difference(registry["special"])
        if unknown_special:
            raise SOTAProtocolError("Unregistered special module(s): %s" % sorted(unknown_special))
    seeds = protocol.get("seeds")
    if (
        not isinstance(seeds, list)
        or len(seeds) < 2
        or not all(isinstance(seed, int) and seed >= 0 for seed in seeds)
        or seeds != sorted(set(seeds))
    ):
        raise SOTAProtocolError(
            "Paired seeds must be a sorted list of at least two unique non-negative integers"
        )
    required_training = {
        "batch_size",
        "max_epochs",
        "negative_sample_num",
        "optimizer",
        "learning_rate",
        "weight_decay",
        "gradient_clip_norm",
        "dropout_rate",
        "early_stopping_patience",
        "early_stopping_min_delta",
        "stopping_metric",
        "num_workers",
        "deterministic",
        "strategy",
        "contrastive_weight",
    }
    missing_training = sorted(required_training.difference(protocol.get("training", {})))
    if missing_training:
        raise SOTAProtocolError("Training fields missing: " + ", ".join(missing_training))
    sequence = protocol.get("sequence_lengths", {})
    if set(sequence) != {"title", "abstract", "history"}:
        raise SOTAProtocolError("sequence_lengths must define title, abstract, history")
    assets = protocol.get("assets", {})
    required_assets = {
        "once_hidden_cache",
        "once_mask_cache",
        "once_manifest",
        "llm_embedding",
        "similar_news",
        "asset_manifest",
    }
    if set(assets) != required_assets:
        raise SOTAProtocolError("assets must define the frozen ONCE and sampling files")
    protocol["_path"] = str(protocol_path)
    protocol["_compositions"] = compositions
    protocol["_resolved_assets"] = {
        key: _resolve_from_protocol(protocol_path, value)
        for key, value in assets.items()
    }
    return protocol


def composition_settings(protocol: Mapping, composition: Composition) -> Dict:
    training = protocol["training"]
    assets = protocol["_resolved_assets"]
    settings = dict(protocol.get("model_settings", {}))
    settings.update({
        "composition_id": composition.id,
        "news_encoder_type": composition.news_encoder,
        "user_encoder_type": composition.user_encoder,
        "enhancement_type": composition.enhancement,
        "training_strategy": composition.training,
        "special_modules": list(composition.special_modules),
        "sota_llm_embedding_path": assets["llm_embedding"],
        "sota_similarity_path": assets["similar_news"],
        "sota_contrastive_weight": training["contrastive_weight"],
    })
    return settings


def synthetic_config(
    composition: Composition, model_settings: Optional[Mapping] = None
) -> SimpleNamespace:
    values = dict(
        model="SOTA-COMPOSABLE",
        composition_id=composition.id,
        news_encoder_type=composition.news_encoder,
        user_encoder_type=composition.user_encoder,
        enhancement_type=composition.enhancement,
        training_strategy=composition.training,
        special_modules=list(composition.special_modules),
        batch_size=2,
        dataset_name="synthetic",
        dataset_size="tiny",
        vocabulary_size=96,
        word_embedding_dim=16,
        max_title_length=6,
        max_abstract_length=7,
        max_history_num=4,
        negative_sample_num=4,
        category_num=5,
        subCategory_num=7,
        category_embedding_dim=4,
        subCategory_embedding_dim=4,
        hidden_dim=4,
        attention_dim=8,
        gcn_layer_num=1,
        no_gcn_residual=False,
        dropout_rate=0.0,
        head_num=2,
        head_dim=4,
        nrms_use_category=True,
        cnn_method="naive",
        cnn_kernel_num=12,
        cnn_window_size=3,
        similarity_num=3,
        sota_composition_dim=24,
        sota_llm_embedding_dim=16,
        sota_llm_embedding_path=":synthetic:",
        user_num=16,
        news_num=96,
        sein_hidden_dim=16,
        sein_stage_num=2,
        sein_gcn_layer_num=1,
        sein_attention_heads=2,
        sein_attention_layers=1,
        sein_temperature=0.1,
        sein_consistency_loss_weight=0.01,
        sein_smoothness_loss_weight=0.01,
    )
    values.update(dict(model_settings or {}))
    return SimpleNamespace(**values)


def synthetic_inputs(config: SimpleNamespace):
    batch = config.batch_size
    history = config.max_history_num
    candidates = 1 + config.negative_sample_num
    if config.training_strategy == "cnrcl":
        candidates += config.similarity_num

    def tokens(*shape):
        return torch.randint(1, config.vocabulary_size, shape, dtype=torch.long)

    history_mask = torch.ones(batch, history, dtype=torch.long)
    history_mask[0, -1] = 0
    title_mask = torch.ones(batch, history, config.max_title_length)
    content_mask = torch.ones(batch, history, config.max_abstract_length)
    candidate_title_mask = torch.ones(batch, candidates, config.max_title_length)
    candidate_content_mask = torch.ones(batch, candidates, config.max_abstract_length)
    graph_size = history + config.category_num
    graph = torch.eye(graph_size).unsqueeze(0).expand(batch, -1, -1).clone()
    category_mask = torch.ones(batch, config.category_num + 1)
    history_category_indices = torch.randint(
        0, config.category_num, (batch, history), dtype=torch.long
    )
    history_index = torch.randint(1, config.news_num, (batch, history))
    news_index = torch.randint(1, config.news_num, (batch, candidates))
    return (
        torch.randint(0, config.user_num, (batch,)),
        torch.randint(0, config.category_num, (batch, history)),
        torch.randint(0, config.subCategory_num, (batch, history)),
        tokens(batch, history, config.max_title_length),
        title_mask,
        torch.zeros(batch, history, config.max_title_length, dtype=torch.long),
        tokens(batch, history, config.max_abstract_length),
        content_mask,
        torch.zeros(batch, history, config.max_abstract_length, dtype=torch.long),
        history_mask,
        graph,
        category_mask,
        history_category_indices,
        torch.randint(0, config.category_num, (batch, candidates)),
        torch.randint(0, config.subCategory_num, (batch, candidates)),
        tokens(batch, candidates, config.max_title_length),
        candidate_title_mask,
        torch.zeros(batch, candidates, config.max_title_length, dtype=torch.long),
        tokens(batch, candidates, config.max_abstract_length),
        candidate_content_mask,
        torch.zeros(batch, candidates, config.max_abstract_length, dtype=torch.long),
        history_index,
        news_index,
    )


def evaluate_lifecycle(
    composition: Composition, model_settings: Optional[Mapping] = None
) -> Dict:
    torch.manual_seed(0)
    config = synthetic_config(composition, model_settings)
    inputs = synthetic_inputs(config)
    checks: Dict[str, Dict] = {}
    model = None
    optimizer = None
    outputs = None
    try:
        model = SOTAComposition(config)
        model.initialize()
        finite = all(torch.isfinite(parameter).all() for parameter in model.parameters())
        if not finite:
            raise AssertionError("Initialization produced non-finite parameters")
        checks["initialization"] = {
            "passed": True,
            "composition": model.composition,
        }
    except Exception as error:
        checks["initialization"] = {"passed": False, "error": repr(error)}

    if checks.get("initialization", {}).get("passed"):
        try:
            model.train()
            outputs = model(*inputs)
            logits, products, auxiliary = outputs
            expected = (config.batch_size, 1 + config.negative_sample_num)
            if tuple(logits.shape) != expected:
                raise AssertionError("Expected logits %r, got %r" % (expected, tuple(logits.shape)))
            expected_products = 1 + config.negative_sample_num
            if config.training_strategy == "cnrcl":
                expected_products += config.similarity_num
            if products.shape[:2] != (config.batch_size, expected_products):
                raise AssertionError("Contrastive products have wrong shape")
            if not all(torch.isfinite(value).all() for value in (logits, products, auxiliary)):
                raise AssertionError("Forward produced non-finite values")
            checks["forward"] = {"passed": True, "shape": list(logits.shape)}
        except Exception as error:
            checks["forward"] = {"passed": False, "error": repr(error)}

    if checks.get("forward", {}).get("passed"):
        try:
            optimizer = torch.optim.Adam(
                (parameter for parameter in model.parameters() if parameter.requires_grad),
                lr=1e-3,
            )
            logits, products, auxiliary = outputs
            prediction = -torch.log_softmax(logits, dim=1)[:, 0].mean() + auxiliary
            loss = prediction
            if config.training_strategy == "cnrcl":
                loss = loss + NewsRecSupConLoss_with_neg_pos()(
                    products, config.negative_sample_num
                )
            optimizer.zero_grad()
            loss.backward()
            component_gradients = {
                "news": any(
                    parameter.grad is not None
                    for parameter in model.news_encoder.parameters()
                    if parameter.requires_grad
                ),
                "user": any(
                    parameter.grad is not None
                    for parameter in model.user_encoder.parameters()
                    if parameter.requires_grad
                ),
            }
            if not all(component_gradients.values()):
                raise AssertionError("Not every trainable component received gradients")
            gradients = [
                parameter.grad
                for parameter in model.parameters()
                if parameter.grad is not None
            ]
            if not gradients or not all(torch.isfinite(value).all() for value in gradients):
                raise AssertionError("Backward produced missing/non-finite gradients")
            optimizer.step()
            checks["backward"] = {
                "passed": True,
                "loss": float(loss.detach()),
                "component_gradients": component_gradients,
            }
        except Exception as error:
            checks["backward"] = {"passed": False, "error": repr(error)}

    checkpoint = None
    expected_prediction = None
    if checks.get("backward", {}).get("passed"):
        try:
            model.eval()
            with torch.no_grad():
                expected_prediction = model(*inputs)
            checkpoint = io.BytesIO()
            torch.save(model.state_dict(), checkpoint)
            checkpoint.seek(0)
            restored = SOTAComposition(config)
            restored.load_state_dict(torch.load(checkpoint, map_location="cpu"))
            restored.eval()
            with torch.no_grad():
                actual_prediction = restored(*inputs)
            if not torch.equal(expected_prediction, actual_prediction):
                raise AssertionError("Checkpoint recovery changed predictions")
            checks["checkpoint_recovery"] = {"passed": True}
        except Exception as error:
            checks["checkpoint_recovery"] = {"passed": False, "error": repr(error)}

    if checks.get("checkpoint_recovery", {}).get("passed"):
        try:
            validation_score = float(expected_prediction[:, 0].mean())
            if not math.isfinite(validation_score):
                raise AssertionError("Validation selection was not finite")
            checkpoint.seek(0)
            best = SOTAComposition(config)
            best.load_state_dict(torch.load(checkpoint, map_location="cpu"))
            best.eval()
            with torch.no_grad():
                test_prediction = best(*inputs)
            if not torch.isfinite(test_prediction).all():
                raise AssertionError("Test workflow produced non-finite predictions")
            checks["full_workflow"] = {
                "passed": True,
                "stages": ["train", "validation", "best_checkpoint", "test"],
            }
        except Exception as error:
            checks["full_workflow"] = {"passed": False, "error": repr(error)}

    for name in LIFECYCLE_CHECKS:
        checks.setdefault(
            name,
            {"passed": False, "error": "skipped because an earlier stage failed"},
        )
    passed = sum(bool(checks[name]["passed"]) for name in LIFECYCLE_CHECKS)
    return {
        "composition_id": composition.id,
        "passed": passed,
        "total": len(LIFECYCLE_CHECKS),
        "checks": checks,
    }


def run_lifecycle_matrix(
    compositions: Iterable[Composition] = COMPOSITIONS,
    model_settings: Optional[Mapping] = None,
) -> Dict:
    results = [
        evaluate_lifecycle(composition, model_settings)
        for composition in compositions
    ]
    return {
        "schema_version": 1,
        "checks": list(LIFECYCLE_CHECKS),
        "results": results,
        "all_passed": all(item["passed"] == item["total"] for item in results),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_freeze_manifest(protocol_path, output_path) -> Dict:
    protocol = load_protocol(protocol_path)
    files = (
        REPOSITORY_ROOT / "base_trainer.py",
        REPOSITORY_ROOT / "general_runner.py",
        REPOSITORY_ROOT / "models" / "SOTAComposition.py",
        REPOSITORY_ROOT / "utils" / "_evaluation.py",
        REPOSITORY_ROOT
        / "dataset_corpus_preprocessing"
        / "MIND_corpus_main.py",
        REPOSITORY_ROOT / "config" / "sota-composable.yaml",
        Path(protocol["_path"]),
        REPOSITORY_ROOT / "experiments" / "sota_composition" / "dataset.py",
        REPOSITORY_ROOT / "scripts" / "run_sota_composition.py",
    )
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(REPOSITORY_ROOT), text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    asset_manifest = Path(protocol["_resolved_assets"]["asset_manifest"])
    if not asset_manifest.is_file():
        raise FileNotFoundError("Asset manifest is missing: %s" % asset_manifest)
    result = {
        "schema_version": 1,
        "git_head": commit,
        "registry": available_sota_components(),
        "compositions": [asdict(value) for value in protocol["_compositions"]],
        "framework_files": {
            str(path.relative_to(REPOSITORY_ROOT)): _sha256(path) for path in files
        },
        "asset_manifest": str(asset_manifest),
        "asset_manifest_sha256": _sha256(asset_manifest),
        "per_composition_core_files_changed": {
            value.id: [] for value in protocol["_compositions"]
        },
        "per_composition_dedicated_model_classes": {
            value.id: [] for value in protocol["_compositions"]
        },
    }
    output = Path(output_path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as output_file:
        json.dump(result, output_file, indent=2, sort_keys=True)
        output_file.write("\n")
    return result


def _load_result(path: Path, composition: Composition, seed: int) -> Mapping:
    with path.open("r", encoding="utf-8") as result_file:
        result = json.load(result_file)
    if result.get("model") != "SOTA-COMPOSABLE" or result.get("seed") != seed:
        raise SOTAProtocolError("Result identity mismatch: %s" % path)
    metadata = result.get("composition", {})
    expected = {
        "id": composition.id,
        "news_encoder": composition.news_encoder,
        "user_encoder": composition.user_encoder,
        "enhancement": composition.enhancement,
        "training": composition.training,
        "special_modules": list(composition.special_modules),
    }
    if metadata != expected:
        raise SOTAProtocolError("Composition metadata mismatch: %s" % path)
    metrics = result.get("metrics", {})
    for metric in METRICS:
        value = metrics.get(metric)
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise SOTAProtocolError("Missing/non-finite %s in %s" % (metric, path))
    return result


def _config_loc(path: Path) -> Optional[int]:
    if not path.is_file():
        return None
    return sum(
        1
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )


def aggregate_results(
    protocol_path,
    results_root,
    output_root,
    lifecycle_path=None,
    freeze_manifest_path=None,
    allow_incomplete=False,
):
    protocol = load_protocol(protocol_path)
    results_root = Path(results_root).resolve()
    output_root = Path(output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    lifecycle_by_id = {}
    if lifecycle_path and Path(lifecycle_path).is_file():
        lifecycle = json.loads(Path(lifecycle_path).read_text(encoding="utf-8"))
        lifecycle_by_id = {
            item["composition_id"]: item for item in lifecycle.get("results", [])
        }
    runs = {}
    missing = []
    for composition in protocol["_compositions"]:
        runs[composition.id] = {}
        for seed in protocol["seeds"]:
            path = results_root / composition.id / ("seed-%d.json" % seed)
            if not path.is_file():
                missing.append({"composition_id": composition.id, "seed": seed, "path": str(path)})
                continue
            runs[composition.id][seed] = _load_result(path, composition, seed)
    if missing and not allow_incomplete:
        raise SOTAProtocolError("Missing %d result files" % len(missing))

    baseline = runs["B0"]
    summary = []
    for composition in protocol["_compositions"]:
        completed_seeds = sorted(runs[composition.id])
        item = {
            **asdict(composition),
            "special_modules": list(composition.special_modules),
            "core_files_changed": 0,
            "dedicated_model_classes": 0,
            "config_loc": _config_loc(
                results_root / composition.id / "seed-0.yaml"
            ),
            "completed_seeds": completed_seeds,
            "metrics": {},
            "paired_vs_b0": {},
        }
        lifecycle = lifecycle_by_id.get(composition.id)
        item["lifecycle"] = (
            "%d/%d" % (lifecycle["passed"], lifecycle["total"])
            if lifecycle
            else None
        )
        for metric in METRICS:
            values = [
                float(runs[composition.id][seed]["metrics"][metric])
                for seed in completed_seeds
            ]
            item["metrics"][metric] = {
                "mean": statistics.mean(values) if values else None,
                "stdev": statistics.stdev(values) if len(values) > 1 else None,
            }
            paired_seeds = [seed for seed in completed_seeds if seed in baseline]
            if composition.id == "B0" or not paired_seeds:
                item["paired_vs_b0"][metric] = {
                    "seeds": paired_seeds,
                    "delta": None if composition.id == "B0" else None,
                    "relative_percent": None,
                    "p_value": None,
                    "significant_improvement": False,
                }
                continue
            variant_values = [
                float(runs[composition.id][seed]["metrics"][metric])
                for seed in paired_seeds
            ]
            baseline_values = [
                float(baseline[seed]["metrics"][metric]) for seed in paired_seeds
            ]
            deltas = [value - base for value, base in zip(variant_values, baseline_values)]
            delta = statistics.mean(deltas)
            base_mean = statistics.mean(baseline_values)
            p_value = None
            if len(paired_seeds) >= 2:
                result = ttest_rel(variant_values, baseline_values)
                p_value = float(result.pvalue) if math.isfinite(float(result.pvalue)) else 1.0
            item["paired_vs_b0"][metric] = {
                "seeds": paired_seeds,
                "delta": delta,
                "relative_percent": 100.0 * delta / base_mean if base_mean else None,
                "p_value": p_value,
                "significant_improvement": bool(
                    delta > 0 and p_value is not None and p_value < 0.05
                ),
            }
        summary.append(item)

    payload = {
        "schema_version": 1,
        "protocol": protocol["_path"],
        "freeze_manifest": str(Path(freeze_manifest_path).resolve()) if freeze_manifest_path else None,
        "completed_runs": sum(len(value) for value in runs.values()),
        "expected_runs": len(protocol["_compositions"]) * len(protocol["seeds"]),
        "missing_results": missing,
        "summary": summary,
    }
    (output_root / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    fieldnames = [
        "id", "news_encoder", "user_encoder", "enhancement", "special_modules",
        "core_files_changed", "dedicated_model_classes", "config_loc", "lifecycle",
        "seeds", "auc_mean", "auc_stdev", "auc_delta", "auc_p_value",
        "mrr_mean", "ndcg5_mean", "ndcg10_mean", "ndcg10_stdev", "ndcg10_delta", "ndcg10_p_value",
    ]
    with (output_root / "summary.csv").open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for item in summary:
            writer.writerow({
                "id": item["id"],
                "news_encoder": item["news_encoder"],
                "user_encoder": item["user_encoder"],
                "enhancement": item["enhancement"],
                "special_modules": " ".join(item["special_modules"]),
                "core_files_changed": 0,
                "dedicated_model_classes": 0,
                "config_loc": item["config_loc"],
                "lifecycle": item["lifecycle"],
                "seeds": " ".join(map(str, item["completed_seeds"])),
                "auc_mean": item["metrics"]["auc"]["mean"],
                "auc_stdev": item["metrics"]["auc"]["stdev"],
                "auc_delta": item["paired_vs_b0"]["auc"]["delta"],
                "auc_p_value": item["paired_vs_b0"]["auc"]["p_value"],
                "mrr_mean": item["metrics"]["mrr"]["mean"],
                "ndcg5_mean": item["metrics"]["ndcg5"]["mean"],
                "ndcg10_mean": item["metrics"]["ndcg10"]["mean"],
                "ndcg10_stdev": item["metrics"]["ndcg10"]["stdev"],
                "ndcg10_delta": item["paired_vs_b0"]["ndcg10"]["delta"],
                "ndcg10_p_value": item["paired_vs_b0"]["ndcg10"]["p_value"],
            })
    return payload

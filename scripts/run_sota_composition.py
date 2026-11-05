#!/usr/bin/env python3
"""Prepare, validate, execute, and report the strong B0--B7 experiment."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from experiments.sota_composition.evaluation import (  # noqa: E402
    COMPOSITIONS,
    aggregate_results,
    build_freeze_manifest,
    composition_settings,
    load_protocol,
    run_lifecycle_matrix,
)


DEFAULT_PROTOCOL = REPOSITORY_ROOT / "experiments" / "sota_composition" / "protocol.yaml"


def _write_json(path: Path, value) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as output_file:
        json.dump(value, output_file, indent=2, sort_keys=True)
        output_file.write("\n")
    temporary.replace(path)


def _select(values, allowed, label):
    if not values:
        return list(allowed)
    unknown = sorted(set(values).difference(allowed))
    if unknown:
        raise ValueError("Unknown %s: %s" % (label, ", ".join(map(str, unknown))))
    return [value for value in allowed if value in values]


def _validate_assets(protocol):
    assets = protocol["_resolved_assets"]
    missing = [path for path in assets.values() if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing frozen experiment assets:\n  " + "\n  ".join(missing)
        )
    manifest = json.loads(Path(assets["asset_manifest"]).read_text(encoding="utf-8"))
    llm = np.load(assets["llm_embedding"], mmap_mode="r", allow_pickle=False)
    similar = np.load(assets["similar_news"], mmap_mode="r", allow_pickle=False)
    if list(llm.shape) != manifest["llm_embedding"]["shape"]:
        raise ValueError("LLM embedding shape disagrees with asset manifest")
    if list(similar.shape) != manifest["similar_news"]["shape"]:
        raise ValueError("Similar-news shape disagrees with asset manifest")
    return manifest


def _jobs(protocol, results_root: Path, selected_ids=None, selected_seeds=None):
    _validate_assets(protocol)
    compositions = {item.id: item for item in protocol["_compositions"]}
    ids = _select(selected_ids, list(compositions), "composition IDs")
    seeds = _select(selected_seeds, protocol["seeds"], "seeds")
    dataset = protocol["dataset"]
    training = protocol["training"]
    sequence = protocol["sequence_lengths"]
    study_key = str(protocol.get("study", "sota-composition")).replace("_", "-")
    dataset_root = Path(dataset["root"])
    if not dataset_root.is_absolute():
        dataset_root = REPOSITORY_ROOT / dataset_root
    jobs = []
    for composition_id in ids:
        composition = compositions[composition_id]
        for seed in seeds:
            job_root = results_root.resolve() / composition.id
            result_path = job_root / ("seed-%d.json" % seed)
            settings_path = job_root / ("seed-%d.yaml" % seed)
            settings = composition_settings(protocol, composition)
            command = [
                sys.executable,
                str(REPOSITORY_ROOT / "general_runner.py"),
                "--model", "SOTA-COMPOSABLE",
                "--mode", "train",
                "--DATASET_ROOT", str(dataset_root.resolve()),
                "--dataset_name", dataset["name"],
                "--dataset_size", dataset["size"],
                "--dataset_cache_key", "%s-%s-seed-%d" % (study_key, composition.id, seed),
                "--seed", str(seed),
                "--epoch", str(training["max_epochs"]),
                "--batch_size", str(training["batch_size"]),
                "--negative_sample_num", str(training["negative_sample_num"]),
                "--max_title_length", str(sequence["title"]),
                "--max_abstract_length", str(sequence["abstract"]),
                "--max_history_num", str(sequence["history"]),
                "--optimizer", training["optimizer"],
                "--lr", str(training["learning_rate"]),
                "--weight_decay", str(training["weight_decay"]),
                "--gradient_clip_norm", str(training["gradient_clip_norm"]),
                "--dropout_rate", str(training["dropout_rate"]),
                "--early_stopping_epoch", str(training["early_stopping_patience"]),
                "--early_stopping_min_delta", str(training["early_stopping_min_delta"]),
                "--dev_criterion", training["stopping_metric"],
                "--num_workers", str(training["num_workers"]),
                "--config_file", str(settings_path),
                "--metrics_output", str(result_path),
            ]
            if training["deterministic"]:
                command.append("--deterministic")
            jobs.append({
                "composition": composition,
                "seed": seed,
                "result_path": result_path,
                "settings_path": settings_path,
                "settings": settings,
                "command": command,
            })
    return jobs


def command_assets(args):
    command = [
        sys.executable,
        str(REPOSITORY_ROOT / "scripts" / "build_sota_composition_assets.py"),
        "--output-root", args.output_root,
    ]
    if args.force:
        command.append("--force")
    subprocess.run(command, cwd=str(REPOSITORY_ROOT), check=True)


def command_validate(args):
    protocol = load_protocol(args.protocol)
    manifest = _validate_assets(protocol)
    print(
        "validated frozen assets: llm=%s similar=%s"
        % (manifest["llm_embedding"]["shape"], manifest["similar_news"]["shape"])
    )


def command_freeze(args):
    result = build_freeze_manifest(args.protocol, args.output)
    print(
        "froze %d components at git %s to %s"
        % (len(result["compositions"]), result["git_head"], args.output)
    )


def command_lifecycle(args):
    protocol = load_protocol(args.protocol)
    result = run_lifecycle_matrix(
        protocol["_compositions"], protocol.get("model_settings")
    )
    _write_json(Path(args.output), result)
    for item in result["results"]:
        print("%s: %d/%d" % (item["composition_id"], item["passed"], item["total"]))
    if not result["all_passed"]:
        raise SystemExit(1)


def command_jobs(args):
    protocol = load_protocol(args.protocol)
    for job in _jobs(protocol, Path(args.results_root), args.ids, args.seeds):
        print(
            "%s seed=%d\n  %s"
            % (job["composition"].id, job["seed"], shlex.join(job["command"]))
        )


def command_run(args):
    protocol = load_protocol(args.protocol)
    for job in _jobs(protocol, Path(args.results_root), args.ids, args.seeds):
        if job["result_path"].is_file() and not args.force:
            print("preserving completed result: %s" % job["result_path"])
            continue
        print(
            "running %s seed=%d" % (job["composition"].id, job["seed"]),
            flush=True,
        )
        if args.dry_run:
            print(shlex.join(job["command"]))
            continue
        job["settings_path"].parent.mkdir(parents=True, exist_ok=True)
        with job["settings_path"].open("w", encoding="utf-8") as settings_file:
            yaml.safe_dump(job["settings"], settings_file, sort_keys=True)
        subprocess.run(job["command"], cwd=str(REPOSITORY_ROOT), check=True)


def command_report(args):
    report = aggregate_results(
        args.protocol,
        args.results_root,
        args.output_root,
        lifecycle_path=args.lifecycle,
        freeze_manifest_path=args.freeze_manifest,
        allow_incomplete=args.allow_incomplete,
    )
    print(
        "reported %d/%d completed runs to %s"
        % (report["completed_runs"], report["expected_runs"], args.output_root)
    )


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    assets = subparsers.add_parser("assets", help="build frozen Qwen/similarity assets")
    assets.add_argument("--output-root", default="artifacts/sota_composition/assets")
    assets.add_argument("--force", action="store_true")
    assets.set_defaults(function=command_assets)

    validate = subparsers.add_parser("validate", help="validate frozen assets")
    validate.add_argument("--protocol", default=str(DEFAULT_PROTOCOL))
    validate.set_defaults(function=command_validate)

    freeze = subparsers.add_parser("freeze", help="record registry/source freeze")
    freeze.add_argument("--protocol", default=str(DEFAULT_PROTOCOL))
    freeze.add_argument(
        "--output", default="artifacts/sota_composition/freeze_manifest.json"
    )
    freeze.set_defaults(function=command_freeze)

    lifecycle = subparsers.add_parser("lifecycle", help="run five lifecycle gates")
    lifecycle.add_argument("--protocol", default=str(DEFAULT_PROTOCOL))
    lifecycle.add_argument(
        "--output", default="artifacts/sota_composition/lifecycle.json"
    )
    lifecycle.set_defaults(function=command_lifecycle)

    for name, function, help_text in (
        ("jobs", command_jobs, "print the paired 40-run matrix"),
        ("run", command_run, "execute selected runs"),
    ):
        command = subparsers.add_parser(name, help=help_text)
        command.add_argument("--protocol", default=str(DEFAULT_PROTOCOL))
        command.add_argument(
            "--results-root", default="artifacts/sota_composition/results"
        )
        command.add_argument("--ids", nargs="+", choices=[item.id for item in COMPOSITIONS])
        command.add_argument("--seeds", nargs="+", type=int)
        if name == "run":
            command.add_argument("--dry-run", action="store_true")
            command.add_argument("--force", action="store_true")
        command.set_defaults(function=function)

    report = subparsers.add_parser("report", help="aggregate metrics and paired tests")
    report.add_argument("--protocol", default=str(DEFAULT_PROTOCOL))
    report.add_argument(
        "--results-root", default="artifacts/sota_composition/results"
    )
    report.add_argument(
        "--lifecycle", default="artifacts/sota_composition/lifecycle.json"
    )
    report.add_argument(
        "--freeze-manifest", default="artifacts/sota_composition/freeze_manifest.json"
    )
    report.add_argument(
        "--output-root", default="artifacts/sota_composition/report"
    )
    report.add_argument("--allow-incomplete", action="store_true")
    report.set_defaults(function=command_report)
    return parser


def main():
    args = build_parser().parse_args()
    args.function(args)


if __name__ == "__main__":
    main()

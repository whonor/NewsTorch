import argparse
import csv
import json
import random
import shutil
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path


SPLIT_COLUMNS = [
    "impression_id",
    "user_id",
    "created_at",
    "history",
    "candidates",
    "clicked",
    "target_news_id",
    "target_label",
]


def read_rows(path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_rows(path, rows, fieldnames):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_truth(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row_id, row in enumerate(rows):
            labels = [int(label) for label in row["clicked"].split()]
            f.write(("" if row_id == 0 else "\n") + str(row_id + 1) + " " + str(labels).replace(" ", ""))


def label_targets(rows, split_size):
    counts = Counter(row["target_label"] for row in rows)
    total = len(rows)
    fake_target = round(split_size * counts["fake"] / total)
    return {"fake": fake_target, "real": split_size - fake_target}


def sample_label_rows(pool, targets, rng):
    selected = []
    selected_ids = set()
    for label, count in targets.items():
        candidates = [row for row in pool if row["target_label"] == label and row["_row_id"] not in selected_ids]
        if len(candidates) < count:
            raise ValueError(f"Not enough {label} rows: need {count}, have {len(candidates)}")
        rng.shuffle(candidates)
        chosen = candidates[:count]
        selected.extend(chosen)
        selected_ids.update(row["_row_id"] for row in chosen)
    return selected, selected_ids


def rebuild_splits(dataset_dir, seed, train_size, val_size, test_size, min_train_rows):
    behaviors_path = dataset_dir / "behaviors.csv"
    rows = read_rows(behaviors_path)
    rng = random.Random(seed)

    if train_size is None or val_size is None or test_size is None:
        split_counts = Counter(row["split"] for row in rows)
        train_size = train_size or split_counts["train"]
        val_size = val_size or split_counts["val"]
        test_size = test_size or split_counts["test"]

    if train_size + val_size + test_size != len(rows):
        raise ValueError(
            f"Split sizes must sum to {len(rows)}, got {train_size + val_size + test_size}"
        )

    for row_id, row in enumerate(rows):
        row["_row_id"] = row_id

    by_user = defaultdict(list)
    for row in rows:
        by_user[row["user_id"]].append(row)
    for user_rows in by_user.values():
        user_rows.sort(key=lambda row: (row["created_at"], int(row["impression_id"])))

    protected_train_ids = set()
    for user_rows in by_user.values():
        protected_train_ids.update(row["_row_id"] for row in user_rows[:min_train_rows])

    eligible = [row for row in rows if row["_row_id"] not in protected_train_ids]
    val_targets = label_targets(rows, val_size)
    test_targets = label_targets(rows, test_size)

    test_rows, test_ids = sample_label_rows(eligible, test_targets, rng)
    remaining = [row for row in eligible if row["_row_id"] not in test_ids]
    val_rows, val_ids = sample_label_rows(remaining, val_targets, rng)

    split_by_id = {row["_row_id"]: "train" for row in rows}
    split_by_id.update({row["_row_id"]: "val" for row in val_rows})
    split_by_id.update({row["_row_id"]: "test" for row in test_rows})

    output = {"train": [], "val": [], "test": []}
    behavior_rows = []
    for row in sorted(rows, key=lambda r: int(r["impression_id"])):
        split = split_by_id[row["_row_id"]]
        clean = {key: row[key] for key in SPLIT_COLUMNS}
        output[split].append(clean)
        behavior_rows.append({**clean, "split": split})

    return output, behavior_rows


def summarize(output):
    summary = {}
    for split, rows in output.items():
        users = {row["user_id"] for row in rows}
        labels = Counter(row["target_label"] for row in rows)
        summary[split] = {
            "rows": len(rows),
            "users": len(users),
            "labels": dict(labels),
            "fake_ratio": labels["fake"] / len(rows) if rows else 0,
        }
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", default="gossipcop")
    parser.add_argument("--seed", type=int, default=20260504)
    parser.add_argument("--train-size", type=int)
    parser.add_argument("--val-size", type=int)
    parser.add_argument("--test-size", type=int)
    parser.add_argument("--min-train-rows", type=int, default=1)
    parser.add_argument("--no-backup", action="store_true")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    output, behavior_rows = rebuild_splits(
        dataset_dir,
        args.seed,
        args.train_size,
        args.val_size,
        args.test_size,
        args.min_train_rows,
    )

    if not args.no_backup:
        backup_dir = dataset_dir / "split_backups" / datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir.mkdir(parents=True, exist_ok=True)
        for name in ["behaviors.csv", "train.csv", "val.csv", "test.csv"]:
            shutil.copy2(dataset_dir / name, backup_dir / name)

    write_rows(dataset_dir / "train.csv", output["train"], SPLIT_COLUMNS)
    write_rows(dataset_dir / "val.csv", output["val"], SPLIT_COLUMNS)
    write_rows(dataset_dir / "test.csv", output["test"], SPLIT_COLUMNS)
    write_rows(dataset_dir / "behaviors.csv", behavior_rows, SPLIT_COLUMNS + ["split"])

    dataset_root = dataset_dir.name
    write_truth(Path("cache/dev/ref") / f"truth-{dataset_root}.txt", output["val"])
    write_truth(Path("cache/test/ref") / f"truth-{dataset_root}.txt", output["test"])

    print(json.dumps(summarize(output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

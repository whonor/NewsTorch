"""Prepare the licensed Adressa one-week click log for NewsTorch.

The public Adressa release stores one JSON object per click (some mirrors gzip
the daily files).  It does not contain impression negatives, so this adapter
constructs deterministic sampled candidate sets and records that fact in the
output manifest.
"""

import argparse
import datetime as dt
import gzip
import hashlib
import json
import os
import random
from collections import defaultdict


PREPARED_FILES = tuple(
    os.path.join(split, name)
    for split in ("train", "dev", "test")
    for name in ("news.jsonl", "behaviors.jsonl")
)


def _clean(value):
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return value
    return str(value).strip()


def _first(record, *keys, default=None):
    for key in keys:
        value = record.get(key)
        if value is not None and value != "":
            return value
    return default


def _timestamp(value):
    if isinstance(value, (int, float)):
        value = float(value)
        if value > 10**12:
            value /= 1000.0
        return value
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return _timestamp(float(text))
    except ValueError:
        pass
    text = text.replace("Z", "+00:00")
    try:
        parsed = dt.datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    return parsed.timestamp()


def _as_strings(value):
    if value is None:
        return []
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return []
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError:
            decoded = None
        if isinstance(decoded, (list, dict)):
            return _as_strings(decoded)
        return [part.strip() for part in value.replace(";", ",").split(",") if part.strip()]
    if isinstance(value, dict):
        identifier = _first(value, "id", "WikidataId", "name", "title", "item", "value")
        return [str(identifier).strip()] if identifier not in (None, "") else []
    if isinstance(value, (list, tuple, set)):
        result = []
        for item in value:
            result.extend(_as_strings(item))
        return result
    return [str(value).strip()]


def _profile_text(profile):
    values = _as_strings(profile)
    return " ".join(dict.fromkeys(value for value in values if value))


def _iter_json(path):
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as source:
        first = source.read(1)
        source.seek(0)
        if first == "[":
            payload = json.load(source)
            for record in payload:
                if isinstance(record, dict):
                    yield record
            return
        if first == "{":
            try:
                payload = json.load(source)
            except json.JSONDecodeError:
                source.seek(0)
            else:
                if isinstance(payload, dict):
                    records = _first(payload, "events", "records", "data")
                    if isinstance(records, list):
                        for record in records:
                            if isinstance(record, dict):
                                yield record
                    else:
                        yield payload
                    return
        for line_number, line in enumerate(source, start=1):
            line = line.strip().rstrip(",")
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON in {path}:{line_number}: {error}") from error
            if isinstance(record, dict):
                yield record


def _raw_files(raw_dir):
    files = []
    for directory, _, names in os.walk(raw_dir):
        for name in names:
            path = os.path.join(directory, name)
            if name.startswith(".") or name in {"README", "LICENSE"}:
                continue
            if name.endswith((".json", ".jsonl", ".ndjson", ".gz")) or "." not in name:
                files.append(path)
    return sorted(files)


def _event_and_article(record):
    news_id = _clean(_first(record, "documentId", "document_id", "newsId", "news_id", "id"))
    user_id = _clean(_first(record, "userId", "user_id", "uid"))
    timestamp = _timestamp(_first(record, "time", "timestamp", "eventTime", "event_time"))
    if not news_id or not user_id or timestamp is None:
        return None, None

    keywords = _as_strings(_first(record, "keywords", "keyword"))
    entities = _as_strings(_first(record, "namedEntities", "named_entities", "entities"))
    category = _clean(_first(record, "category", "section", default="unknown")) or "unknown"
    subcategory = _clean(_first(record, "subcategory", "subCategory"))
    if not subcategory:
        subcategory = keywords[0] if keywords else category
    title = _clean(_first(record, "title", "headline", default=news_id)) or news_id
    abstract = _clean(_first(record, "abstract", "description", "lead"))
    if not abstract:
        abstract = _profile_text(_first(record, "profile", default=keywords))

    article = {
        "nid": news_id,
        "title": str(title),
        "abstract": str(abstract),
        "category": str(category),
        "subcategory": str(subcategory),
        "keywords": keywords,
        "entities": entities,
        "publish_time": _timestamp(_first(record, "publishTime", "publish_time")),
        "url": _clean(_first(record, "canonicalUrl", "url")),
    }
    event = {
        "uid": user_id,
        "nid": news_id,
        "timestamp": timestamp,
        "read_time": max(0.0, float(_first(record, "activeTime", "active_time", default=0.0) or 0.0)),
    }
    return event, article


def _stable_rng(seed, *parts):
    digest = hashlib.sha256("|".join(map(str, (seed,) + parts)).encode("utf-8")).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def _candidate_set(event, history, news_ids, negative_num, seed):
    excluded = set(history)
    excluded.add(event["nid"])
    pool = [news_id for news_id in news_ids if news_id not in excluded]
    if not pool:
        return None
    rng = _stable_rng(seed, event["uid"], event["nid"], event["timestamp"])
    negatives = rng.sample(pool, min(negative_num, len(pool)))
    candidates = [event["nid"]] + negatives
    labels = [1] + [0] * len(negatives)
    order = list(range(len(candidates)))
    rng.shuffle(order)
    return [candidates[index] for index in order], [labels[index] for index in order]


def _write_jsonl(path, records):
    with open(path, "w", encoding="utf-8") as output:
        for record in records:
            output.write(json.dumps(record, ensure_ascii=False) + "\n")


def is_prepared(dataset_root):
    return all(os.path.exists(os.path.join(dataset_root, path)) for path in PREPARED_FILES)


def prepare_adressa_1week(dataset_root="Adressa-1week", raw_dir=None, negative_num=20,
                          max_history_num=50, min_history=1, seed=0, force=False):
    """Convert raw Adressa events into NewsTorch news and behavior files."""
    dataset_root = os.path.abspath(dataset_root)
    raw_dir = os.path.abspath(raw_dir or os.path.join(dataset_root, "raw"))
    if is_prepared(dataset_root) and not force:
        return os.path.join(dataset_root, "manifest.json")
    if not os.path.isdir(raw_dir):
        raise FileNotFoundError(
            f"Adressa raw directory not found: {raw_dir}. Place the licensed one-week "
            "JSON/JSONL files there, then rerun the preparation command."
        )
    files = _raw_files(raw_dir)
    if not files:
        raise FileNotFoundError(f"No JSON, JSONL, gzip, or extensionless daily files found in {raw_dir}")

    events = []
    articles = {}
    for path in files:
        for record in _iter_json(path):
            event, article = _event_and_article(record)
            if event is None:
                continue
            events.append(event)
            previous = articles.get(article["nid"], {})
            articles[article["nid"]] = {
                key: value if value not in (None, "", []) else previous.get(key, value)
                for key, value in article.items()
            }
    if not events:
        raise ValueError(f"No click records with userId, documentId, and time were found in {raw_dir}")
    events.sort(key=lambda item: (item["timestamp"], item["uid"], item["nid"]))
    first_seen = {}
    for event in events:
        first_seen[event["nid"]] = min(first_seen.get(event["nid"], event["timestamp"]), event["timestamp"])
    days = sorted({dt.datetime.fromtimestamp(item["timestamp"], dt.timezone.utc).date() for item in events})
    if len(days) < 7:
        raise ValueError(f"Adressa-1week preparation requires seven UTC dates, found {len(days)}: {days}")
    selected_days = days[:7]
    day_index = {day: index for index, day in enumerate(selected_days)}
    events = [
        event for event in events
        if dt.datetime.fromtimestamp(event["timestamp"], dt.timezone.utc).date() in day_index
    ]
    seventh_day_events = [
        event for event in events
        if day_index[dt.datetime.fromtimestamp(event["timestamp"], dt.timezone.utc).date()] == 6
    ]
    test_boundary = seventh_day_events[len(seventh_day_events) // 2]["timestamp"] if seventh_day_events else float("inf")

    histories = defaultdict(list)
    history_times = defaultdict(list)
    splits = {"train": [], "dev": [], "test": []}
    news_ids = sorted(articles)
    skipped_short_history = 0
    for event in events:
        day = dt.datetime.fromtimestamp(event["timestamp"], dt.timezone.utc).date()
        index = day_index[day]
        history = histories[event["uid"]]
        read_times = history_times[event["uid"]]
        split = None
        if index == 5:
            split = "train"
        elif index == 6:
            split = "dev" if event["timestamp"] < test_boundary else "test"
        if split is not None:
            if len(history) >= min_history:
                available_news = [
                    news_id for news_id in news_ids
                    if first_seen.get(news_id, event["timestamp"]) <= event["timestamp"]
                ]
                candidates_and_labels = _candidate_set(event, history, available_news, negative_num, seed)
                if candidates_and_labels is not None:
                    candidates, labels = candidates_and_labels
                    candidate_read_time = [event["read_time"] if label == 1 else 0.0 for label in labels]
                    splits[split].append({
                        "uid": event["uid"],
                        "timestamp": event["timestamp"],
                        "history": history[-max_history_num:],
                        "history_read_time": read_times[-max_history_num:],
                        "candidates": candidates,
                        "labels": labels,
                        "next_read_time": event["read_time"],
                        "candidate_read_time": candidate_read_time,
                    })
            else:
                skipped_short_history += 1
        history.append(event["nid"])
        read_times.append(event["read_time"])

    empty = [split for split, records in splits.items() if not records]
    if empty:
        raise ValueError(
            f"Prepared split(s) {empty} are empty. Lower --min-history or inspect the raw date coverage."
        )
    news_records = [articles[news_id] for news_id in news_ids]
    for split, records in splits.items():
        split_root = os.path.join(dataset_root, split)
        os.makedirs(split_root, exist_ok=True)
        _write_jsonl(os.path.join(split_root, "news.jsonl"), news_records)
        _write_jsonl(os.path.join(split_root, "behaviors.jsonl"), records)

    manifest = {
        "dataset": "Adressa-1week",
        "raw_dir": raw_dir,
        "raw_files": len(files),
        "date_range_utc": [str(selected_days[0]), str(selected_days[-1])],
        "split_policy": "days 1-5 history; day 6 train; first/second half of day 7 dev/test",
        "negative_sampling": "deterministic random articles observed by the event time, excluding the positive and user history",
        "negative_num": int(negative_num),
        "max_history_num": int(max_history_num),
        "min_history": int(min_history),
        "seed": int(seed),
        "articles": len(news_records),
        "events": len(events),
        "skipped_short_history": skipped_short_history,
        "behaviors": {split: len(records) for split, records in splits.items()},
    }
    manifest_path = os.path.join(dataset_root, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as output:
        json.dump(manifest, output, ensure_ascii=False, indent=2)
    return manifest_path


def main():
    parser = argparse.ArgumentParser(description="Prepare the licensed Adressa one-week click log.")
    parser.add_argument("--dataset-root", default="Adressa-1week")
    parser.add_argument("--raw-dir", default="")
    parser.add_argument("--negative-num", type=int, default=20)
    parser.add_argument("--max-history-num", type=int, default=50)
    parser.add_argument("--min-history", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    manifest = prepare_adressa_1week(
        dataset_root=args.dataset_root,
        raw_dir=args.raw_dir or None,
        negative_num=args.negative_num,
        max_history_num=args.max_history_num,
        min_history=args.min_history,
        seed=args.seed,
        force=args.force,
    )
    print(f"Prepared Adressa-1week: {manifest}")


if __name__ == "__main__":
    main()

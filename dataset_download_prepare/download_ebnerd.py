#!/usr/bin/env python3
import os
import sys
from pathlib import Path
import polars as pl
import datetime as dt
import pandas as pd
from zipfile import ZipFile

import sys
from pathlib import Path

# Add path into sys.path
ROOT = str(Path(__file__).resolve().parents[1])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from utils._behaviors import ebnerd_from_path, sampling_strategy_wu2019, create_binary_labels_column


def download_file(url: str, dest: Path, chunk_size: int = 1024):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"already existed，skip download: {dest}")
        return
    resp = requests.get(url, stream=True)
    resp.raise_for_status()
    total = int(resp.headers.get('content-length', 0))
    with open(dest, 'wb') as f, tqdm(
        total=total, unit='iB', unit_scale=True, desc=dest.name
    ) as bar:
        for chunk in resp.iter_content(chunk_size=chunk_size):
            f.write(chunk)
            bar.update(len(chunk))
    print(f"download complete: {dest}")

def extract_zip(zip_path: Path, extract_to: Path):
    if extract_to.exists() and any(extract_to.iterdir()):
        print(f"existed and no empty，skip extracted: {extract_to}")
        return
    print(f"Extracting {zip_path} → {extract_to}")
    extract_to.mkdir(parents=True, exist_ok=True)
    with ZipFile(zip_path, 'r') as z:
        z.extractall(extract_to)
    print(f"extract complete: {extract_to}")

def process_dataset(base_dir: Path, splits: dict):
    download_dir = base_dir / 'download'
    for split, url in splits.items():
        filename = Path(url).name
        zip_path = download_dir / filename
        download_file(url, zip_path)
        extract_zip(zip_path, download_dir / split)


def to_tsv(df: pd.DataFrame, fpath: str) -> None:
    """Stores a dataframe in `.tsv` format."""
    df.to_csv(fpath, sep="\t", index=False, header=False)


def _load_news(source_file_path, dst_dir):
    '''
    Articles:
    ['article_id', 'title', 'subtitle', 'last_modified_time', 'premium', 'body', 
    'published_time', 'image_ids', 'article_type', 'url', 'ner_clusters', 'entity_groups', 
    'topics', 'category', 'subcategory', 'category_str', 'total_inviews', 'total_pageviews', 
    'total_read_time', 'sentiment_score', 'sentiment_label']

    '''
    """load news"""
    parsed_news_file = os.path.join(dst_dir, "news.parquet")
    article_file = os.path.join(source_file_path, "articles.parquet")
    print("News not parsed. Loading and parsing raw data.")

    df_articles = pl.read_parquet(article_file,
                                  columns=[
                                      "article_id", # i32
                                      "category", # list[i16]
                                      "subcategory", # list[i16]
                                      "title", # str
                                      "subtitle", # str
                                      "topics", # list[str]
                                      'body', # str
                                      'image_ids', # list[i64]
                                      'article_type', # str
                                      'ner_clusters', # list[str]
                                      'entity_groups', # list[str]
                                      'category_str', # str
                                      "total_inviews", # i32
                                      "total_pageviews", # i32
                                      "total_read_time", # f32
                                      "sentiment_score", # f32
                                      "sentiment_label" # str
                                  ])

    news = df_articles.rename({
        "article_id": "nid",
        "subtitle": "abstract"
    })

    news = news.filter(pl.col("nid").is_not_null())
    news = news.unique(subset=["nid"])

    news.write_parquet(parsed_news_file)
    # to_tsv(news, parsed_news_file)

    return news


def _load_behaviors(source_file_path, dst_dir, split="train"):
    """load the user behaviors"""
    source_file_path = os.path.join(
        source_file_path, split
    )
    parsed_bhv_file = os.path.join(
        dst_dir, split, "behaviors_.parquet"
    )
    print("User behaviors not parsed. Loading and parsing raw data.")
    '''
    Histories:
    ['user_id', 'impression_time_fixed', 'scroll_percentage_fixed', 'article_id_fixed', 
    'read_time_fixed']
    Behaviors:
    ['impression_id', 'article_id', 'impression_time', 'read_time', 'scroll_percentage', 
    'device_type', 'article_ids_inview', 'article_ids_clicked', 'user_id', 'is_sso_user', 
    'gender', 'postcode', 'age', 'is_subscriber', 'session_id', 'next_read_time', 
    'next_scroll_percentage']
    '''
    # load behaviors
    print("Parsing behaviors.")
    PATH = Path(source_file_path)
    df_behaviors = (
        ebnerd_from_path(
            PATH,
            history_size=20,
            padding=0,
        )
        .sample(fraction=1.0, shuffle=True, seed=42)
        .select([
            "impression_time",
            "article_id_fixed",
            "read_time_fixed",
            "article_ids_inview",
            "article_ids_clicked",
            "impression_id",
            "user_id",
            "next_read_time"
        ])
        # .pipe(
        #     sampling_strategy_wu2019,
        #     npratio=4,
        #     shuffle=True,
        #     with_replacement=True,
        #     seed=42,
        # )
        .pipe(create_binary_labels_column)
    )
    '''
                    .pipe(
            sampling_strategy_wu2019,
            npratio=self.neg_num,
            shuffle=True,
            with_replacement=True,
            seed=42,
        )
    '''
    # column_names = ["impression_id", "user_id", "impression_time", "article_id_fixed", "article_ids_inview",
    #                 "article_ids_clicked", "labels"]
    new_names = ["impid", "uid", "time", "history", "history_read_time", "impressions", "labels", "next_read_time"]
    behaviors = df_behaviors.rename({"impression_id": "impid",
                                          "user_id": "uid",
                                          "impression_time": "time",
                                          "article_id_fixed": "history",
                                          "read_time_fixed": "history_read_time",
                                          "article_ids_inview": "impressions"})
    behaviors = behaviors[new_names]

    """
    === behaviors 时间范围分析 ===
    起始时间: 2023-05-18 07:00:03
    结束时间: 2023-05-25 06:59:52
    总跨度: 6 days, 23:59:49
    """
    # last_dt = behaviors["time"].max() - dt.timedelta(days=1)

    # behaviors["time"] = pd.to_datetime(behaviors["time"], format="%m/%d/%Y %I:%M:%S %p")

    # Apply the conversion to the 'history' column
    behaviors = behaviors.with_columns([
        pl.col("impressions").alias("candidates")
    ]).drop("impressions")

    cnt_bhv = len(behaviors)
    behaviors = behaviors.filter(pl.col("history").list.len() > 0)
    dropped_bhv = cnt_bhv - len(behaviors)
    print(
        f"Removed {dropped_bhv} ({dropped_bhv / cnt_bhv}%) behaviors without user history"
    )

    # to_tsv(behaviors, parsed_bhv_file)
    behaviors.write_parquet(parsed_bhv_file)

    return behaviors

def main():
    datasets = {
        'ebnerd_demo': 'https://ebnerd-dataset.s3.eu-west-1.amazonaws.com/ebnerd_demo.zip',
        'ebnerd_small': 'https://ebnerd-dataset.s3.eu-west-1.amazonaws.com/ebnerd_small.zip',
        # 'ebnerd_large': 'https://ebnerd-dataset.s3.eu-west-1.amazonaws.com/ebnerd_large.zip',
        # 'Ekstra_Bladet_image_embeddings': 'https://ebnerd-dataset.s3.eu-west-1.amazonaws.com/artifacts/Ekstra_Bladet_image_embeddings.zip'
    }

    root = Path(__file__).resolve().parent.parent

    for name, url in datasets.items():
        print(f"\n=== Processing dataset: {name} ===")
        ds_dir = root / name
        ds_dir.mkdir(parents=True, exist_ok=True)
        process_dataset(ds_dir, {name: url})

    print("\nAll downloads and extractions completed.")

    for name, url in datasets.items():
        print("\n=== Preparing EB-NeRD %s news data. ===" % name)
        _load_news(source_file_path=str(root / name / "download" / name),
                   dst_dir=str(root / name / "download" / name))
        print("\n=== Preparing EB-NeRD %s users behaviour data. ===" % name)
        _load_behaviors(source_file_path=str(root / name / "download" / name),
                        dst_dir=str(root / name / "download" / name), split="train")
        _load_behaviors(source_file_path=str(root / name / "download" / name),
                        dst_dir=str(root / name / "download" / name), split="validation")

    print("\nAll news and behaviors data process completed.")


def clean_data(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as infile, \
            open(output_path, 'w', encoding='utf-8') as outfile:
        line_num = 0
        skipped = 0

        for line in infile:
            line_num += 1
            line = line.strip()
            parts = line.split('\t')
            if len(parts) == 6:
                outfile.write(line + '\n')
            else:
                skipped += 1
                print(f"Line {line_num} skipped (has {len(parts)} fields): {line}")

        print(f"\nCleaning complete: {line_num - skipped} lines kept, {skipped} lines skipped.")

if __name__ == '__main__':
    try:
        import requests
        from tqdm import tqdm
    except ImportError:
        print("Please install dependencies：pip install requests tqdm", file=sys.stderr)
        sys.exit(1)
    main()



import json
import os
import re
import tarfile
from ast import literal_eval
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from random import shuffle
from typing import Any, Dict, List, Optional, Tuple

import datetime as dt
import numpy as np
import pandas as pd
import requests
import torch.nn as nn
import polars as pl
from sympy.printing.tree import print_node

from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from dataset_corpus_preprocessing.download_utils import download_path, maybe_download, extract_file

from ebrec.utils._behaviors import ebnerd_from_path, sampling_strategy_wu2019, create_binary_labels_column

def to_tsv(df: pd.DataFrame, fpath: str) -> None:
    """Stores a dataframe in `.tsv` format."""
    df.to_csv(fpath, sep="\t", index=False)


def _load_news(dst_dir):
    """加载新闻数据"""
    parsed_news_file = os.path.join(dst_dir, "news.tsv")
    article_file = os.path.join(dst_dir, "articles.parquet")
    print("News not parsed. Loading and parsing raw data.")
    '''
    Articles:
    ['article_id', 'title', 'subtitle', 'last_modified_time', 'premium', 'body', 
    'published_time', 'image_ids', 'article_type', 'url', 'ner_clusters', 'entity_groups', 
    'topics', 'category', 'subcategory', 'category_str', 'total_inviews', 'total_pageviews', 
    'total_read_time', 'sentiment_score', 'sentiment_label']

    '''
    df_articles = pl.read_parquet(article_file,
                                  columns=[
                                      "article_id",
                                      "category",
                                      "subcategory",
                                      "title",
                                      "subtitle",
                                      "body",
                                      "entity_groups",
                                  ])

    news = df_articles.to_pandas()

    news = news.rename(columns={
        "article_id": "nid",
        "subtitle": "abstract"
    })
    news = news.set_index("nid", drop=True)
    to_tsv(news, parsed_news_file)

    return news


def _load_behaviors(source_file_path, dst_dir):
    """加载用户行为数据"""
    file_prefix = ""
    parsed_bhv_file = os.path.join(
        dst_dir + "behaviors.tsv"
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
            "article_ids_inview",
            "article_ids_clicked",
            "impression_id",
            "user_id"
        ])
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
    column_names = ["impression_id", "user_id", "impression_time", "article_id_fixed", "article_ids_inview",
                    "article_ids_clicked", "labels"]
    new_names = ["impid", "uid", "time", "history", "impressions", "labels"]
    behaviors = df_behaviors.to_pandas()
    behaviors = behaviors.rename(columns={"impression_id": "impid",
                                          "user_id": "uid",
                                          "impression_time": "time",
                                          "article_id_fixed": "history",
                                          "article_ids_inview": "impressions"})
    behaviors = behaviors[new_names]

    """
    === behaviors 时间范围分析 ===
    起始时间: 2023-05-18 07:00:03
    结束时间: 2023-05-25 06:59:52
    总跨度: 6 days, 23:59:49
    """
    last_dt = behaviors["time"].max() - dt.timedelta(days=1)

    # behaviors["time"] = pd.to_datetime(behaviors["time"], format="%m/%d/%Y %I:%M:%S %p")

    # Apply the conversion to the 'history' column
    behaviors["candidates"] = behaviors["impressions"]
    behaviors = behaviors.drop(columns=["impressions"])

    behaviors["history"] = behaviors["history"].apply(lambda x: [y for y in x.tolist()])
    behaviors["candidates"] = behaviors["candidates"].apply(lambda x: [y for y in x.tolist()])

    cnt_bhv = len(behaviors)
    behaviors = behaviors[behaviors["history"].apply(len) > 0]
    dropped_bhv = cnt_bhv - len(behaviors)
    print(
        f"Removed {dropped_bhv} ({dropped_bhv / cnt_bhv}%) behaviors without user history"
    )

    behaviors = behaviors.reset_index(drop=True)
    to_tsv(behaviors, parsed_bhv_file)

    return behaviors


def main():



if __name__ == "__main__":
    main()











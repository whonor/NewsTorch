# =========================================================================
# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================

from typing import List, Tuple, Dict, Any

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import multiprocessing
from sklearn.metrics import roc_auc_score, log_loss, accuracy_score
from torchmetrics import AUROC
from torchmetrics.retrieval import RetrievalMRR, RetrievalNormalizedDCG
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm
from time import gmtime, strftime
import argparse
import pandas as pd
import numpy as np
import os

from config import Config
from evaluate import scoring

def func_auc(grouped_df):
    if sum(grouped_df["label"]) == 0 or sum(grouped_df["label"]) == len(grouped_df["label"]):
        return 1.0
    auc_metric = AUROC(task='binary', num_classes=2)
    labels = torch.tensor(grouped_df["label"].values, dtype=torch.float32)
    scores = torch.tensor(grouped_df["score"].values, dtype=torch.float32)
    auc = auc_metric(labels, scores)
    return auc

def func_mrr(grouped_df):
    if sum(grouped_df["label"]) == 0 or sum(grouped_df["label"]) == len(grouped_df["label"]):
        return 1.0
    mrr_metric = RetrievalMRR()
    labels = torch.tensor(grouped_df["label"].values, dtype=torch.float32)
    scores = torch.tensor(grouped_df["score"].values, dtype=torch.float32)
    mrr = mrr_metric(labels, scores)
    return mrr

def func_ndcg5(grouped_df):
    if sum(grouped_df["label"]) == 0 or sum(grouped_df["label"]) == len(grouped_df["label"]):
        return 1.0
    ndcg5_metric = RetrievalNormalizedDCG(top_k=5)
    labels = torch.tensor(grouped_df["label"].values, dtype=torch.float32)
    scores = torch.tensor(grouped_df["score"].values, dtype=torch.float32)
    ndcg5 = ndcg5_metric(labels, scores, 5)
    return ndcg5

def func_ndcg10(grouped_df):
    if sum(grouped_df["label"]) == 0 or sum(grouped_df["label"]) == len(grouped_df["label"]):
        return 1.0
    ndcg10_metric = RetrievalNormalizedDCG(top_k=10)
    labels = torch.tensor(grouped_df["label"].values, dtype=torch.float32)
    scores = torch.tensor(grouped_df["score"].values, dtype=torch.float32)
    ndcg10 = ndcg10_metric(labels, scores, 10)
    return ndcg10

def dev(model, dev_loader, device, out_path, is_epoch=False):
    impression_ids = []
    labels = []
    scores = []
    batch_iterator = tqdm(dev_loader, disable=False)
    for step, dev_batch in enumerate(batch_iterator):
        if not is_epoch and step >= 5000:
            break
        impression_id, label = dev_batch['impression_id'], dev_batch['label']
        with torch.no_grad():
            batch_score = model(dev_batch['input_ids'].to(device), 
                                dev_batch['input_mask'].to(device), 
                                dev_batch['segment_ids'].to(device),
                                dev_batch['news_segment_ids'].to(device),
                                dev_batch['sentence_ids'].to(device),
                                dev_batch['sentence_mask'].to(device),
                                dev_batch['sentence_segment_ids'].to(device))
            batch_score = batch_score.softmax(dim=-1)[:, 1].squeeze(-1)
            batch_score = batch_score.detach().cpu().tolist()
            if not isinstance(batch_score, list):
                batch_score = [batch_score]
            impression_ids.extend(impression_id)
            labels.extend(label.tolist())
            scores.extend(batch_score)

    score_path = os.path.join(out_path, "dev_score.tsv")
    EVAL_DF = pd.DataFrame()
    EVAL_DF["impression_id"] = impression_ids
    EVAL_DF["label"] = labels
    EVAL_DF["score"] = scores
    EVAL_DF.to_csv(score_path, sep="\t", index=False)
    groups_iter = EVAL_DF.groupby("impression_id")
    imp, df_groups = zip(*groups_iter)
    pool = multiprocessing.Pool()
    result_auc = pool.map(func_auc, df_groups)
    result_mrr = pool.map(func_mrr, df_groups)
    result_ndcg5 = pool.map(func_ndcg5, df_groups)
    result_ndcg10 = pool.map(func_ndcg10, df_groups)
    pool.close()
    pool.join()
    auc = np.mean(result_auc)
    mrr = np.mean(result_mrr)
    ndcg5 = np.mean(result_ndcg5)
    ndcg10 = np.mean(result_ndcg10)

    return auc, mrr, ndcg5, ndcg10

def rank_func(x):
    scores = x["score"].tolist()
    tmp = [(i, s) for i, s in enumerate(scores)]
    tmp = sorted(tmp, key=lambda y: y[-1], reverse=True)
    rank = [(i+1, t[0]) for i, t in enumerate(tmp)]
    rank = [str(r[0]) for r in sorted(rank, key=lambda y: y[-1])]
    rank = "[" + ",".join(rank) + "]"
    return {"imp": x["impression_id"].tolist()[0], "rank": rank}

def test(model, test_loader, device, out_path):
    score_path = os.path.join(out_path, "test_score.tsv")
    outfile = os.path.join(out_path, "prediction.txt")
    impression_ids = []
    scores = []
    batch_iterator = tqdm(test_loader, disable=False)
    for step, test_batch in enumerate(batch_iterator):
        impression_id = test_batch['impression_id']
        with torch.no_grad():
            batch_score = model(test_batch['input_ids'].to(device), 
                                test_batch['input_mask'].to(device), 
                                test_batch['segment_ids'].to(device),
                                test_batch['news_segment_ids'].to(device),
                                test_batch['sentence_ids'].to(device),
                                test_batch['sentence_mask'].to(device),
                                test_batch['sentence_segment_ids'].to(device))
            batch_score = batch_score.softmax(dim=-1)[:, 1].squeeze(-1)
            batch_score = batch_score.detach().cpu().tolist()
            if not isinstance(batch_score, list):
                batch_score = [batch_score]
            impression_ids.extend(impression_id)
            scores.extend(batch_score)

    EVAL_DF = pd.DataFrame()
    EVAL_DF["impression_id"] = impression_ids
    EVAL_DF["score"] = scores
    EVAL_DF.to_csv(score_path, sep="\t", index=False)
    groups_iter = EVAL_DF.groupby("impression_id")
    imp, df_groups = zip(*groups_iter)
    pool = multiprocessing.Pool()
    result = pool.map(rank_func, df_groups)
    pool.close()
    pool.join()
    imps = [r["imp"] for r in result]
    ranks = [r["rank"] for r in result]
    with open(outfile, "w") as fout:
        out = [str(imp) + " " + rank for imp, rank in zip(imps, ranks)]
        fout.write("\n".join(out))
    return

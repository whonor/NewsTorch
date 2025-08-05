#!/usr/bin/env python
import sys, os, os.path
import numpy as np
import json
from sklearn.metrics import roc_auc_score

import torch
from torchmetrics.classification import AUROC
from torchmetrics.retrieval import RetrievalNormalizedDCG, RetrievalMRR


def dcg_score(y_true, y_score, k=10):
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])
    gains = 2 ** y_true - 1
    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gains / discounts)


def ndcg_score(y_true, y_score, k=10):
    best = dcg_score(y_true, y_true, k)
    actual = dcg_score(y_true, y_score, k)
    return actual / best


def mrr_score(y_true, y_score):
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order)
    rr_score = y_true / (np.arange(len(y_true)) + 1)
    return np.sum(rr_score) / np.sum(y_true)


def parse_line(l):
    # impid, ranks = l.strip('\n').split()
    impid, ranks = l.strip().split()
    print("[DEBUG] parse_line got ranks:", ranks)
    try:
        ranks = json.loads(ranks)
    except json.JSONDecodeError:
        # try to fix the error format：如 [1,,4,,5,,3,,2] → extract [1, 4, 5, 3, 2]
        content = ranks.strip('[] ')
        # transform '1, 4, 5, 3, 2' to ['1', '4', '5', '3', '2']
        # remove empty parts and convert to integers
        parts = [p for p in content.split(',') if p.strip().isdigit()]
        ranks_fixed = [int(p.strip()) for p in parts]
        ranks = ranks_fixed
        # try to fix this issue
        ranks_fixed = '[' + ','.join(ranks.strip('[] ').split()) + ']'
        ranks = json.loads(ranks_fixed)

    return impid, ranks


def scoring(truth_f, sub_f):
    aucs = []
    mrrs = []
    ndcg5s = []
    ndcg10s = []

    line_index = 1
    for lt in truth_f:
        ls = sub_f.readline()
        impid, labels = parse_line(lt)

        # ignore masked impressions
        if labels == []:
            continue

        if ls == '':
            # empty line: filled with 0 ranks
            sub_impid = impid
            sub_ranks = [1] * len(labels)
        else:
            try:
                sub_impid, sub_ranks = parse_line(ls)
            except:
                raise ValueError("line-{}: Invalid Input Format!".format(line_index))

        if sub_impid != impid:
            raise ValueError("line-{}: Inconsistent Impression Id {} and {}".format(
                line_index,
                sub_impid,
                impid
            ))

        lt_len = float(len(labels))

        y_true = np.array(labels, dtype='float32')
        y_score = []
        for rank in sub_ranks:
            score_rslt = 1. / rank
            if score_rslt < 0 or score_rslt > 1:
                raise ValueError("Line-{}: score_rslt should be int from 0 to {}".format(
                    line_index,
                    lt_len
                ))
            y_score.append(score_rslt)

        auc = roc_auc_score(y_true, y_score)
        mrr = mrr_score(y_true, y_score)
        ndcg5 = ndcg_score(y_true, y_score, 5)
        ndcg10 = ndcg_score(y_true, y_score, 10)

        aucs.append(auc)
        mrrs.append(mrr)
        ndcg5s.append(ndcg5)
        ndcg10s.append(ndcg10)

        line_index += 1

    return np.mean(aucs), np.mean(mrrs), np.mean(ndcg5s), np.mean(ndcg10s)
        


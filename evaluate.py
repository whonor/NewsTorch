#!/usr/bin/env python
import sys, os, os.path
import numpy as np
import json
from sklearn.metrics import roc_auc_score

import torch
from torchmetrics.classification import AUROC
from torchmetrics.retrieval import RetrievalNormalizedDCG, RetrievalMRR

def parse_line(l):
    impid, ranks = l.strip('\n').split()
    ranks = json.loads(ranks)
    return impid, ranks

def scoring(truth_f, sub_f):
    auc_metric = AUROC(task='binary', num_classes=2)
    mrr_metric = RetrievalMRR()
    ndcg5_metric = RetrievalNormalizedDCG(top_k=5)
    ndcg10_metric = RetrievalNormalizedDCG(top_k=10)

    line_index = 1
    for lt in truth_f:
        ls = sub_f.readline()
        impid, labels = parse_line(lt)

        # Ignore masked impressions
        if labels == []:
            continue

        if ls == '':
            # Empty line: filled with 0 ranks
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

        y_true = torch.tensor(labels, dtype=torch.float32)
        y_score = torch.tensor([1. / rank for rank in sub_ranks], dtype=torch.float32)

        # Update metrics
        auc_metric.update(y_score, y_true.int())
        mrr_metric.update(y_score, y_true.int())
        ndcg5_metric.update(y_score, y_true.int())
        ndcg10_metric.update(y_score, y_true.int())

        line_index += 1

    # Compute final metrics
    auc = auc_metric.compute().item()
    mrr = mrr_metric.compute().item()
    ndcg5 = ndcg5_metric.compute().item()
    ndcg10 = ndcg10_metric.compute().item()

    return auc, mrr, ndcg5, ndcg10
        

if __name__ == '__main__':
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]

    submit_dir = os.path.join(input_dir, 'res') 
    truth_dir = os.path.join(input_dir, 'ref')

    if not os.path.isdir(submit_dir):
        print("%s doesn't exist" % submit_dir)

    if os.path.isdir(submit_dir) and os.path.isdir(truth_dir):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        output_filename = os.path.join(output_dir, 'scores.txt')              
        output_file = open(output_filename, 'w')

        truth_file = open(os.path.join(truth_dir, "truth.txt"), 'r')
        submission_answer_file = open(os.path.join(submit_dir, "prediction.txt"), 'r')
        
        auc, mrr, ndcg, ndcg10 = scoring(truth_file, submission_answer_file)

        output_file.write("AUC:{:.4f}\nMRR:{:.4f}\nnDCG@5:{:.4f}\nnDCG@10:{:.4f}".format(auc, mrr, ndcg, ndcg10))
        output_file.close()

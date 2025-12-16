import os
import torch
import torch.nn as nn
from torchmetrics import MeanSquaredError, MeanAbsoluteError
from tqdm import tqdm

from dataset_corpus_preprocessing.EBNeRD_corpus_main import EBNeRD_Corpus, Ebnerd_DevTest_Dataset
from dataset_corpus_preprocessing.MIND_corpus_IPNR import MIND_DevTest_Dataset_IPNR
from dataset_corpus_preprocessing.MIND_corpus_main import MIND_Corpus, MIND_DevTest_Dataset
from torch.utils.data import DataLoader

from config import Config
import sys, os, os.path
import numpy as np
import json
from sklearn.metrics import roc_auc_score


def MAE_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_score = torch.tensor(y_score)
    y_true = torch.tensor(y_true)
    mae = MeanAbsoluteError()
    score = mae(y_score, y_true)
    return score

def RMSE_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_score = torch.tensor(y_score)
    y_true = torch.tensor(y_true)
    rmse = MeanSquaredError(squared=False)
    score = rmse(y_score, y_true)
    return score

def recall_at_k(y_true, y_score, k=5):
    """
    Compute Recall@k using numpy for efficiency.
    """
    # Sort indices by scores in descending order
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order)
    # Top-k indices in y_true
    y_true_k = y_true[:k]
    # Calculate recall@k
    return np.sum(y_true_k) / np.sum(y_true)

def hit_at_k(y_true, y_score, k=5):
    """
    Compute Hit@k using numpy for efficiency.
    """
    # Sort indices by scores in descending order
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order)
    # Top-k indices in y_true
    y_true_k = y_true[:k]
    # Check if there's at least one hit in top-k
    return 1.0 if np.any(y_true_k) else 0.0

def precision_at_k(y_true, y_score, k=5):
    """
    Compute Precision@k using numpy for efficiency.
    """
    # Sort indices by scores in descending order
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order)
    # Top-k indices in y_true
    y_true_k = y_true[:k]
    # Calculate precision@k
    return np.sum(y_true_k) / k

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
    # print("[DEBUG] parse_line got ranks:", ranks)
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

    maes = []
    rmses = []
    recall5s = []
    recall10s = []
    hit5s = []
    hit10s = []
    precision5s = []
    precision10s = []

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

        mae = MAE_score(y_true, y_score)
        rmse = RMSE_score(y_true, y_score)
        recall5 = recall_at_k(y_true, y_score, 5)
        recall10 = recall_at_k(y_true, y_score, 10)
        precision5 = precision_at_k(y_true, y_score, 5)
        precision10 = precision_at_k(y_true, y_score, 10)
        hit5 = hit_at_k(y_true, y_score, 5)
        hit10 = hit_at_k(y_true, y_score, 10)

        maes.append(mae)
        rmses.append(rmse)
        recall5s.append(recall5)
        recall10s.append(recall10)
        hit5s.append(hit5)
        hit10s.append(hit10)
        precision5s.append(precision5)
        precision10s.append(precision10)

        line_index += 1

    return (np.mean(aucs), np.mean(mrrs), np.mean(ndcg5s), np.mean(ndcg10s),
            np.mean(maes), np.mean(rmses), np.mean(recall5s), np.mean(recall10s),
            np.mean(hit5s), np.mean(hit10s), np.mean(precision5s), np.mean(precision10s))


def compute_scores_IPNR(config: Config, model: nn.Module, mind_corpus: MIND_Corpus, batch_size: int, mode: str, result_file: str, dataset: str):
    assert mode in ['dev', 'test'], 'mode must be chosen from \'dev\' or \'test\''
    dataloader = DataLoader(MIND_DevTest_Dataset_IPNR(mind_corpus, mode), batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    indices = (mind_corpus.dev_indices if mode == 'dev' else mind_corpus.test_indices)
    scores = torch.zeros([len(indices)]).cuda()
    index = 0
    torch.cuda.empty_cache()
    model.eval()
    with torch.no_grad():
        for (user_ID, user_category, user_subCategory, user_title_text, user_title_mask, user_title_entity, user_content_text, user_content_mask, user_content_entity, user_history_mask, user_history_graph, user_concept_text, user_concept_mask, \
             news_category, news_subCategory, news_title_text, news_title_mask, news_title_entity, news_content_text, news_content_mask, news_content_entity, news_concept_text, news_concept_mask) in tqdm(dataloader):
            user_ID = user_ID.cuda(non_blocking=True)
            user_category = user_category.cuda(non_blocking=True)
            user_subCategory = user_subCategory.cuda(non_blocking=True)
            user_title_text = user_title_text.cuda(non_blocking=True)
            user_title_mask = user_title_mask.cuda(non_blocking=True)
            user_title_entity = user_title_entity.cuda(non_blocking=True)
            user_content_text = user_content_text.cuda(non_blocking=True)
            user_content_mask = user_content_mask.cuda(non_blocking=True)
            user_content_entity = user_content_entity.cuda(non_blocking=True)
            user_history_mask = user_history_mask.cuda(non_blocking=True)
            user_history_graph = user_history_graph.cuda(non_blocking=True)
            # user_history_category_mask = user_history_category_mask.cuda(non_blocking=True)
            # user_history_category_indices = user_history_category_indices.cuda(non_blocking=True)
            user_concept_text = user_concept_text.cuda(non_blocking=True)
            user_concept_mask = user_concept_mask.cuda(non_blocking=True)

            news_category = news_category.cuda(non_blocking=True)
            news_subCategory = news_subCategory.cuda(non_blocking=True)
            news_title_text = news_title_text.cuda(non_blocking=True)
            news_title_mask = news_title_mask.cuda(non_blocking=True)
            news_title_entity = news_title_entity.cuda(non_blocking=True)
            news_content_text = news_content_text.cuda(non_blocking=True)
            news_content_mask = news_content_mask.cuda(non_blocking=True)
            news_content_entity = news_content_entity.cuda(non_blocking=True)
            news_concept_text = news_concept_text.cuda(non_blocking=True)
            news_concept_mask = news_concept_mask.cuda(non_blocking=True)

            batch_size = user_ID.size(0)
            news_category = news_category.unsqueeze(dim=1)
            news_subCategory = news_subCategory.unsqueeze(dim=1)
            news_title_text = news_title_text.unsqueeze(dim=1)
            news_title_mask = news_title_mask.unsqueeze(dim=1)
            news_content_text = news_content_text.unsqueeze(dim=1)
            news_content_mask = news_content_mask.unsqueeze(dim=1)
            news_concept_text = news_concept_text.unsqueeze(dim=1)
            news_concept_mask = news_concept_mask.unsqueeze(dim=1)

            scores[index: index+batch_size] = model(user_ID, user_category, user_subCategory, user_title_text, user_title_mask, user_title_entity, user_content_text, user_content_mask, user_content_entity, user_history_mask, user_history_graph, user_concept_text, user_concept_mask, \
                                                    news_category, news_subCategory, news_title_text, news_title_mask, news_title_entity, news_content_text, news_content_mask, news_content_entity, news_concept_text, news_concept_mask).squeeze(dim=1) # [batch_size]
            index += batch_size
    scores = scores.tolist()
    sub_scores = [[] for _ in range(indices[-1] + 1)]
    for i, index in enumerate(indices):
        sub_scores[index].append([scores[i], len(sub_scores[index])])
    with open(result_file, 'w', encoding='utf-8') as result_f:
        for i, sub_score in enumerate(sub_scores):
            sub_score.sort(key=lambda x: x[0], reverse=True)
            result = [0 for _ in range(len(sub_score))]
            for j in range(len(sub_score)):
                result[sub_score[j][1]] = j + 1
            result_f.write(('' if i == 0 else '\n') + str(i + 1) + ' ' + str(result).replace(' ', ''))
    if dataset != 'submission' or mode != 'test':
        with open("./cache/" + mode + '/ref/truth-%s.txt' % config.DATASET_ROOT, 'r', encoding='utf-8') as truth_f, open(result_file, 'r', encoding='utf-8') as result_f:
            auc, mrr, ndcg5, ndcg10, mae, rmse, recall5, recall10, hit5, hit10, precision5, precision10 = scoring(truth_f, result_f)
        return auc, mrr, ndcg5, ndcg10, mae, rmse, recall5, recall10, hit5, hit10, precision5, precision10
    else:
        return None, None, None, None, None, None, None, None, None, None, None, None


def compute_scores(config: Config, model: nn.Module, corpus, batch_size: int, mode: str, result_file: str, dataset: str):
    assert mode in ['dev', 'test'], 'mode must be chosen from \'dev\' or \'test\''
    if config.dataset_name == 'ebnerd':
        corpus = EBNeRD_Corpus(config)
        dataset = Ebnerd_DevTest_Dataset(corpus, mode)
    elif config.dataset_name == 'MIND':
        corpus = MIND_Corpus(config)
        dataset = MIND_DevTest_Dataset(corpus, mode)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                            num_workers=batch_size // 16, pin_memory=True)
    indices = (corpus.dev_indices if mode == 'dev' else corpus.test_indices)
    scores = torch.zeros([len(indices)]).cuda()
    index = 0
    model.eval()

    with torch.no_grad():
        for (user_ID, user_category, user_subCategory, user_title_text, user_title_mask, user_title_entity, user_content_text, user_content_mask, user_content_entity, user_history_mask, user_history_graph, user_history_category_mask, user_history_category_indices, \
             news_category, news_subCategory, news_title_text, news_title_mask, news_title_entity, news_content_text, news_content_mask, news_content_entity, user_hist_sentiment, news_sentiment, history_index, candidate_news_index) in tqdm(dataloader):
            user_ID = user_ID.cuda(non_blocking=True)
            user_category = user_category.cuda(non_blocking=True)
            user_subCategory = user_subCategory.cuda(non_blocking=True)
            user_title_text = user_title_text.cuda(non_blocking=True)
            user_title_mask = user_title_mask.cuda(non_blocking=True)
            user_title_entity = user_title_entity.cuda(non_blocking=True)
            user_content_text = user_content_text.cuda(non_blocking=True)
            user_content_mask = user_content_mask.cuda(non_blocking=True)
            user_content_entity = user_content_entity.cuda(non_blocking=True)
            user_history_mask = user_history_mask.cuda(non_blocking=True)
            user_history_graph = user_history_graph.cuda(non_blocking=True)
            user_history_category_mask = user_history_category_mask.cuda(non_blocking=True)
            user_history_category_indices = user_history_category_indices.cuda(non_blocking=True)
            news_category = news_category.cuda(non_blocking=True)
            news_subCategory = news_subCategory.cuda(non_blocking=True)
            news_title_text = news_title_text.cuda(non_blocking=True)
            news_title_mask = news_title_mask.cuda(non_blocking=True)
            news_title_entity = news_title_entity.cuda(non_blocking=True)
            news_content_text = news_content_text.cuda(non_blocking=True)
            news_content_mask = news_content_mask.cuda(non_blocking=True)
            news_content_entity = news_content_entity.cuda(non_blocking=True)
            user_hist_sentiment = user_hist_sentiment.cuda(non_blocking=True)
            news_sentiment = news_sentiment.cuda(non_blocking=True)
            # history_index = history_index.cuda(non_blocking=True)
            # candidate_news_index = candidate_news_index.cuda(non_blocking=True)

            batch_size = user_ID.size(0)
            news_category = news_category.unsqueeze(dim=1)
            news_subCategory = news_subCategory.unsqueeze(dim=1)
            news_title_text = news_title_text.unsqueeze(dim=1)
            news_title_mask = news_title_mask.unsqueeze(dim=1)
            news_content_text = news_content_text.unsqueeze(dim=1)
            news_content_mask = news_content_mask.unsqueeze(dim=1)
            candidate_news_index = candidate_news_index.unsqueeze(dim=1)

            if config.model == "TANR":
                scores[index: index + batch_size], _ = model(user_ID, user_category, user_subCategory, user_title_text,
                                                          user_title_mask, user_title_entity, user_content_text,
                                                          user_content_mask, user_content_entity, user_history_mask,
                                                          user_history_graph, user_history_category_mask,
                                                          user_history_category_indices, \
                                                          news_category, news_subCategory, news_title_text,
                                                          news_title_mask, news_title_entity, news_content_text,
                                                          news_content_mask, news_content_entity)  # [batch_size]
            elif config.model == "CNE-SUE" or config.model == "DKN" or config.model == "FIM":
                scores[index: index + batch_size] = model(user_ID, user_category, user_subCategory, user_title_text,
                                                          user_title_mask, user_title_entity, user_content_text,
                                                          user_content_mask, user_content_entity, user_history_mask,
                                                          user_history_graph, user_history_category_mask,
                                                          user_history_category_indices, \
                                                          news_category, news_subCategory, news_title_text,
                                                          news_title_mask, news_title_entity, news_content_text,
                                                          news_content_mask, news_content_entity).squeeze(1)  # [batch_size]
            elif config.model == "LKPNR":
                scores[index: index + batch_size] = model(user_ID, user_category, user_subCategory, user_title_text,
                                                          user_title_mask, user_title_entity, user_content_text,
                                                          user_content_mask, user_content_entity, user_history_mask,
                                                          user_history_graph, user_history_category_mask,
                                                          user_history_category_indices, \
                                                          news_category, news_subCategory, news_title_text,
                                                          news_title_mask, news_title_entity, news_content_text,
                                                          news_content_mask, news_content_entity, history_index,
                                                          candidate_news_index).squeeze(dim=1)  # [batch_size]
            elif config.model == "SentiDebias":
                scores[index: index + batch_size] = model(user_ID, user_category, user_subCategory, user_title_text,
                                                          user_title_mask, user_title_entity, user_content_text,
                                                          user_content_mask, user_content_entity, user_history_mask,
                                                          user_history_graph, user_history_category_mask,
                                                          user_history_category_indices, \
                                                          news_category, news_subCategory, news_title_text,
                                                          news_title_mask, news_title_entity, news_content_text,
                                                          news_content_mask, news_content_entity, user_hist_sentiment, news_sentiment)[0]

            else:
                scores[index: index + batch_size] = model(user_ID, user_category, user_subCategory, user_title_text,
                                                          user_title_mask, user_title_entity, user_content_text,
                                                          user_content_mask, user_content_entity, user_history_mask,
                                                          user_history_graph, user_history_category_mask,
                                                          user_history_category_indices, \
                                                          news_category, news_subCategory, news_title_text,
                                                          news_title_mask, news_title_entity, news_content_text,
                                                          news_content_mask, news_content_entity) # [batch_size, 5]
            index += batch_size
    scores = scores.tolist()
    sub_scores = [[] for _ in range(indices[-1] + 1)]
    for i, index in enumerate(indices):
        sub_scores[index].append([scores[i], len(sub_scores[index])])
    with open(result_file, 'w', encoding='utf-8') as result_f:
        for i, sub_score in enumerate(sub_scores):
            sub_score.sort(key=lambda x: x[0], reverse=True)
            result = [0 for _ in range(len(sub_score))]
            for j in range(len(sub_score)):
                result[sub_score[j][1]] = j + 1
            result_f.write(('' if i == 0 else '\n') + str(i + 1) + ' ' + str(result).replace(' ', ''))
    if dataset != 'submission' or mode != 'test':
        with open(config.data_path + '/' + mode + '/ref/truth-%s.txt' % config.DATASET_ROOT, 'r', encoding='utf-8') as truth_f, open(result_file, 'r', encoding='utf-8') as result_f:
            auc, mrr, ndcg5, ndcg10, mae, rmse, recall5, recall10, hit5, hit10, precision5, precision10 = scoring(truth_f, result_f)
        return auc, mrr, ndcg5, ndcg10, mae, rmse, recall5, recall10, hit5, hit10, precision5, precision10
    else:
        return None, None, None, None, None, None, None, None, None, None, None, None


def compute_scores_mmrec(config: Config, model: nn.Module, corpus, batch_size: int, mode: str, result_file: str, dataset: str):
    assert mode in ['dev', 'test'], 'mode must be chosen from \'dev\' or \'test\''
    if config.dataset_name == 'ebnerd':
        corpus = EBNeRD_Corpus(config)
        dataset = Ebnerd_DevTest_Dataset(corpus, mode)
    elif config.dataset_name == 'MIND':
        corpus = MIND_Corpus(config)
        dataset = MIND_DevTest_Dataset(corpus, mode)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                            num_workers=batch_size // 16, pin_memory=True)
    indices = (corpus.dev_indices if mode == 'dev' else corpus.test_indices)
    scores = torch.zeros([len(indices)]).cuda()
    index = 0
    model.eval()

    with torch.no_grad():
        for (user_ID, user_category, user_subCategory, user_title_text, user_title_mask, user_title_entity, user_content_text, user_content_mask, user_content_entity, user_history_mask, user_history_graph, user_history_category_mask, user_history_category_indices, \
             news_category, news_subCategory, news_title_text, news_title_mask, news_title_entity, news_content_text, news_content_mask, news_content_entity, user_hist_sentiment, news_sentiment, history_index, candidate_news_index, \
             history_image_embedding, candidate_image_embedding) in tqdm(dataloader):
            
            user_history_mask = user_history_mask.cuda(non_blocking=True)

            news_feature = {
                "input_ids": news_title_text.cuda(non_blocking=True),
                "attention_mask": news_title_mask.cuda(non_blocking=True),
                "input_imgs": candidate_image_embedding.unsqueeze(1).cuda(non_blocking=True),
                "image_loc": torch.zeros(news_title_text.shape[0], 1, 5).cuda(non_blocking=True),
            }
            history_feature = {
                "input_ids": user_title_text.cuda(non_blocking=True),
                "attention_mask": user_title_mask.cuda(non_blocking=True),
                "input_imgs": history_image_embedding.unsqueeze(2).cuda(non_blocking=True),
                "image_loc": torch.zeros(user_title_text.shape[0], user_title_text.shape[1], 1, 5).cuda(non_blocking=True),
            }
            batch_size = user_ID.size(0)
            
            score = model(news_feature, history_feature, user_history_mask, None, compute_loss=False)
            scores[index: index + batch_size] = score.squeeze(1)

            index += batch_size
    scores = scores.tolist()
    sub_scores = [[] for _ in range(indices[-1] + 1)]
    for i, index in enumerate(indices):
        sub_scores[index].append([scores[i], len(sub_scores[index])])
    with open(result_file, 'w', encoding='utf-8') as result_f:
        for i, sub_score in enumerate(sub_scores):
            sub_score.sort(key=lambda x: x[0], reverse=True)
            result = [0 for _ in range(len(sub_score))]
            for j in range(len(sub_score)):
                result[sub_score[j][1]] = j + 1
            result_f.write(('' if i == 0 else '\n') + str(i + 1) + ' ' + str(result).replace(' ', ''))
    if dataset != 'submission' or mode != 'test':
        with open(config.data_path + '/' + mode + '/ref/truth-%s.txt' % config.DATASET_ROOT, 'r', encoding='utf-8') as truth_f, open(result_file, 'r', encoding='utf-8') as result_f:
            auc, mrr, ndcg5, ndcg10, mae, rmse, recall5, recall10, hit5, hit10, precision5, precision10 = scoring(truth_f, result_f)
        return auc, mrr, ndcg5, ndcg10, mae, rmse, recall5, recall10, hit5, hit10, precision5, precision10
    else:
        return None, None, None, None, None, None, None, None, None, None, None, None


def get_run_index(result_dir: str):
    assert os.path.exists(result_dir), 'result directory does not exist'
    max_index = 0
    for result_file in os.listdir(result_dir):
        if result_file.strip()[0] == '#' and result_file.strip()[-4:] == '-dev':
            index = int(result_file.strip()[1:-4])
            max_index = max(index, max_index)
    with open(result_dir + '/#' + str(max_index + 1) + '-dev', 'w', encoding='utf-8') as result_f:
        pass
    return max_index + 1


class AvgMetric:
    def __init__(self, auc, mrr, ndcg5, ndcg10):
        self.auc = auc
        self.mrr = mrr
        self.ndcg5 = ndcg5
        self.ndcg10 = ndcg10
        self.avg = (self.auc + self.mrr + (self.ndcg5 + self.ndcg10) / 2) / 3

    def __gt__(self, value):
        return self.avg > value.avg

    def __ge__(self, value):
        return self.avg >= value.avg

    def __lt__(self, value):
        return self.avg < value.avg

    def __le__(self, value):
        return self.avg <= value.avg

    def __str__(self):
        return '%.4f\nAUC = %.4f\nMRR = %.4f\nnDCG@5 = %.4f\nnDCG@10 = %.4f' % (self.avg, self.auc, self.mrr, self.ndcg5, self.ndcg10)

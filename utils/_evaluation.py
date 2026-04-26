import os
import torch
import torch.nn as nn
from torchmetrics import MeanSquaredError, MeanAbsoluteError
from tqdm import tqdm

from dataset_corpus_preprocessing.EBNeRD_corpus_main import EBNeRD_Corpus, Ebnerd_DevTest_Dataset
from dataset_corpus_preprocessing.MIND_corpus_IPNR import MIND_DevTest_Dataset_IPNR
from dataset_corpus_preprocessing.MIND_corpus_main import MIND_Corpus, MIND_DevTest_Dataset
from dataset_corpus_preprocessing.MIND_corpus_SentiRec import MIND_Corpus_SentiRec, MIND_DevTest_Dataset_SentiRec
from torch.utils.data import DataLoader

from config import Config
import sys, os, os.path
import numpy as np
import json
from sklearn.metrics import roc_auc_score
import time

def _get_model_inputs(config, data_batch):
    data_batch = [item.cuda(non_blocking=True) if isinstance(item, torch.Tensor) else item for item in data_batch]
    
    if config.model == 'MMRec':
        (user_ID, _, _, user_title_text, user_title_mask, _, _, _, _, user_history_mask, _, _, _,
         _, _, news_title_text, news_title_mask, _, _, _, _, _, _, _, _,
         history_image_embedding, candidate_image_embedding) = data_batch

        news_feature = {
            "input_ids": news_title_text,
            "attention_mask": news_title_mask,
            "input_imgs": candidate_image_embedding.unsqueeze(1),
            "image_loc": torch.zeros(news_title_text.shape[0], 1, 5).cuda(non_blocking=True),
        }
        history_feature = {
            "input_ids": user_title_text,
            "attention_mask": user_title_mask,
            "input_imgs": history_image_embedding.unsqueeze(2),
            "image_loc": torch.zeros(user_title_text.shape[0], user_title_text.shape[1], 1, 5).cuda(non_blocking=True),
        }
        # MMRec forward signature: forward(self, news_feature, history_feature, user_history_mask, label=None, compute_loss=True)
        # compute_scores_mmrec calls: model(news_feature, history_feature, user_history_mask, None, compute_loss=False)
        return (news_feature, history_feature, user_history_mask, None, False)
    
    elif config.model == 'SentiRec':
        if config.dataset_name == 'MIND':
             # 0-9: user features, 10-17: news features, 18-19: sentiment, 20-21: indices
             # We need to insert 3 Nones for graph args at index 10
             args = list(data_batch[:10]) + [None, None, None] + list(data_batch[10:18]) + [data_batch[20], data_batch[21], data_batch[18], data_batch[19]]
             return tuple(args)
        else:
             return tuple(data_batch[:21] + [data_batch[23], data_batch[24], data_batch[21], data_batch[22]])
    
    elif config.model == 'IPNR':
        return data_batch
        
    else:
        user_ID = data_batch[0]
        news_category = data_batch[13]
        news_subCategory = data_batch[14]
        news_title_text = data_batch[15]
        news_title_mask = data_batch[16]
        news_title_entity = data_batch[17]
        news_content_text = data_batch[18]
        news_content_mask = data_batch[19]
        news_content_entity = data_batch[20]
        if config.dataset_name == 'MIND':
            candidate_news_index = data_batch[22]
        else:
            candidate_news_index = data_batch[24]

        news_category = news_category.unsqueeze(dim=1)
        news_subCategory = news_subCategory.unsqueeze(dim=1)
        news_title_text = news_title_text.unsqueeze(dim=1)
        news_title_mask = news_title_mask.unsqueeze(dim=1)
        news_title_entity = news_title_entity.unsqueeze(dim=1)
        news_content_text = news_content_text.unsqueeze(dim=1)
        news_content_mask = news_content_mask.unsqueeze(dim=1)
        news_content_entity = news_content_entity.unsqueeze(dim=1)
        candidate_news_index = candidate_news_index.unsqueeze(dim=1)

        data_batch[13] = news_category
        data_batch[14] = news_subCategory
        data_batch[15] = news_title_text
        data_batch[16] = news_title_mask
        data_batch[17] = news_title_entity
        data_batch[18] = news_content_text
        data_batch[19] = news_content_mask
        data_batch[20] = news_content_entity
        if config.dataset_name == 'MIND':
            data_batch[22] = candidate_news_index
        else:
            data_batch[24] = candidate_news_index

        if config.model == "TANR":
             return data_batch[:21]
        elif config.model == "CNE-SUE" or config.model == "DKN" or config.model == "FIM":
             return data_batch[:21]
        elif config.model == "LKPNR":
            if config.dataset_name == 'ebnerd':
                return data_batch[:21] + data_batch[23:26]
            else:
                return data_batch
        elif config.model == "SentiDebias":
            def discretize(scores):
                labels = torch.ones_like(scores, dtype=torch.long)
                labels[scores < -0.05] = 0
                labels[scores > 0.05] = 2
                return labels

            if config.dataset_name == 'MIND':
                # 0-9: user, 10-17: news, 18: sent_hist, 19: sent_cand
                return tuple(list(data_batch[:10]) + [None, None, None] + list(data_batch[10:18]) + [discretize(data_batch[18]), discretize(data_batch[19])])
            elif config.dataset_name == 'ebnerd':
                # Ebnerd: 0-9 user, 10-12 graph, 13-20 news, 21 hist_sent, 22 cand_sent
                return tuple(list(data_batch[:10]) + [None, None, None] + list(data_batch[13:21]) + [discretize(data_batch[21]), discretize(data_batch[22])])
        else:
            return data_batch[:21]

def compute_complexity(model, config, data_batch):
    try:
        from thop import profile
    except ImportError:
        print("thop is not installed. Please install it to compute complexity.")
        model.eval()
    print(f"DEBUG: model type: {type(model)}")
    print(f"DEBUG: total parameters: {sum(p.numel() for p in model.parameters())}")
    inputs = _get_model_inputs(config, data_batch)
    
    # thop requires tuple inputs
    if not isinstance(inputs, tuple) and not isinstance(inputs, list):
         inputs = (inputs,)

    # 1. Clean up attributes to allow thop to register its buffers
    for m in model.modules():
        if hasattr(m, "total_ops"): del m.total_ops
        if hasattr(m, "total_params"): del m.total_params

    def count_zero_ops(m, x, y):
        m.total_ops = torch.DoubleTensor([0])

    custom_ops = {
        nn.Dropout: count_zero_ops,
        nn.Dropout2d: count_zero_ops,
        nn.Dropout3d: count_zero_ops,
        nn.Identity: count_zero_ops,
    }

    flops, params = 0, 0
    try:
        # 2. Run profile. Use return values if successful for best accuracy.
        flops, params = profile(model, inputs=inputs, custom_ops=custom_ops, verbose=False)
    except Exception as e:
        print(f"DEBUG: thop.profile crashed: {e}")
        # import traceback
        # traceback.print_exc()
        # 3. Fallback: manual summation if profile crashes during its internal summation
        for m in model.modules():
            if len(list(m.children())) == 0: # Sum only leaf modules to avoid double counting
                if hasattr(m, "total_ops"):
                    val = m.total_ops.item() if isinstance(m.total_ops, torch.Tensor) else m.total_ops
                    flops += val
    finally:
        # 4. Ensure all hooks are removed even if profile crashes
        for m in model.modules():
            if hasattr(m, "_forward_hooks"):
                m._forward_hooks.clear()
            if hasattr(m, "_forward_pre_hooks"):
                m._forward_pre_hooks.clear()

    # Always use a reliable parameter count
    param_list = list(model.parameters())
    if not param_list:
        print("DEBUG: model.parameters() is EMPTY!")
        # Try to find parameters in submodules manually if top-level list is empty for some reason
        total_p = 0
        for m in model.modules():
            total_p += sum(p.numel() for p in m.parameters(recurse=False))
        params = total_p
    else:
        params = sum(p.numel() for p in param_list)
    
    print(f"DEBUG: Calculated params: {params}")
            
    return flops, params

def compute_inference_time(model, config, data_batch, repetitions=100):
    model.eval()
    inputs = _get_model_inputs(config, data_batch)
    
    # Check if inputs is a sequence to unpack
    is_sequence = isinstance(inputs, (tuple, list))
    
    # Warmup
    with torch.no_grad():
        if is_sequence:
            model(*inputs)
        else:
            model(inputs)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        
    start_time = time.time()
    with torch.no_grad():
        for _ in range(repetitions):
            if is_sequence:
                model(*inputs)
            else:
                model(inputs)
                
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    end_time = time.time()
    
    return (end_time - start_time) / repetitions


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
        for data_batch in tqdm(dataloader):
            data_batch = [item.cuda(non_blocking=True) if isinstance(item, torch.Tensor) else item for item in data_batch]
            (user_ID, _, _, _, _, _, _, _, _, _, _, _, _,
             news_category, news_subCategory, news_title_text, news_title_mask, _, news_content_text, news_content_mask, _, news_concept_text, news_concept_mask) = data_batch

            batch_size = user_ID.size(0)
            news_category = news_category.unsqueeze(dim=1)
            news_subCategory = news_subCategory.unsqueeze(dim=1)
            news_title_text = news_title_text.unsqueeze(dim=1)
            news_title_mask = news_title_mask.unsqueeze(dim=1)
            news_content_text = news_content_text.unsqueeze(dim=1)
            news_content_mask = news_content_mask.unsqueeze(dim=1)
            news_concept_text = news_concept_text.unsqueeze(dim=1)
            news_concept_mask = news_concept_mask.unsqueeze(dim=1)

            scores[index: index+batch_size] = model(*data_batch).squeeze(dim=1) # [batch_size]
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
        if config.model == 'SentiRec':
            corpus = MIND_Corpus_SentiRec(config)
            dataset = MIND_DevTest_Dataset_SentiRec(corpus, mode)
        elif config.model == 'SentiDebias':
            corpus = MIND_Corpus_SentiDebias(config)
            dataset = MIND_DevTest_Dataset_SentiDebias(corpus, mode)
        else:
            corpus = MIND_Corpus(config)
            dataset = MIND_DevTest_Dataset(corpus, mode)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                            num_workers=batch_size // 16, pin_memory=True)
    indices = (corpus.dev_indices if mode == 'dev' else corpus.test_indices)
    scores = torch.zeros([len(indices)]).cuda()
    index = 0
    model.eval()

    with torch.no_grad():
        for data_batch in tqdm(dataloader):
            data_batch = [item.cuda(non_blocking=True) if isinstance(item, torch.Tensor) else item for item in data_batch]
            user_ID = data_batch[0]
            batch_size = user_ID.size(0)

            if config.model == "SentiRec" and config.dataset_name == 'MIND':
                # Custom handling for SentiRec on MIND to avoid incorrect unsqueezing
                args = list(data_batch[:10]) + [None, None, None] + list(data_batch[10:18]) + [data_batch[20], data_batch[21]]
                kwargs = {
                    'history_sentiment': data_batch[18],
                    'candidate_sentiment': data_batch[19]
                }
                scores[index: index + batch_size] = model(*args, **kwargs)[0]
                index += batch_size
                continue
            
            if config.model == "SentiDebias":
                def discretize(scores):
                    labels = torch.ones_like(scores, dtype=torch.long)
                    labels[scores < -0.05] = 0
                    labels[scores > 0.05] = 2
                    return labels

                if config.dataset_name == 'MIND':
                     # 0-9: user, 10-17: news, 18: sent_hist, 19: sent_cand
                     args = list(data_batch[:10]) + [None, None, None] + list(data_batch[10:18]) + [discretize(data_batch[18]), discretize(data_batch[19])]
                     scores[index: index + batch_size] = model(*args)[1]
                elif config.dataset_name == 'ebnerd':
                     # Ebnerd: 0-9 user, 10-12 graph, 13-20 news, 21 hist_sent, 22 cand_sent
                     args = list(data_batch[:10]) + [None, None, None] + list(data_batch[13:21]) + [discretize(data_batch[21]), discretize(data_batch[22])]
                     scores[index: index + batch_size] = model(*args)[1]
                index += batch_size
                continue

            news_category = data_batch[13]
            news_subCategory = data_batch[14]
            news_title_text = data_batch[15]
            news_title_mask = data_batch[16]
            news_content_text = data_batch[18]
            news_content_mask = data_batch[19]
            if config.dataset_name == 'MIND':
                candidate_news_index = data_batch[22]
            else:
                candidate_news_index = data_batch[24]

            batch_size = user_ID.size(0)
            news_category = news_category.unsqueeze(dim=1)
            news_subCategory = news_subCategory.unsqueeze(dim=1)
            news_title_text = news_title_text.unsqueeze(dim=1)
            news_title_mask = news_title_mask.unsqueeze(dim=1)
            news_content_text = news_content_text.unsqueeze(dim=1)
            news_content_mask = news_content_mask.unsqueeze(dim=1)
            candidate_news_index = candidate_news_index.unsqueeze(dim=1)

            data_batch[13] = news_category
            data_batch[14] = news_subCategory
            data_batch[15] = news_title_text
            data_batch[16] = news_title_mask
            data_batch[18] = news_content_text
            data_batch[19] = news_content_mask
            if config.dataset_name == 'MIND':
                data_batch[22] = candidate_news_index
            else:
                data_batch[24] = candidate_news_index

            if config.model == "TANR":
                score, _ = model(*data_batch[:21])
                scores[index: index + batch_size] = score
            elif config.model == "CNE-SUE" or config.model == "DKN" or config.model == "FIM":
                scores[index: index + batch_size] = model(*data_batch[:21]).squeeze(1)
            elif config.model == "LKPNR":
                if config.dataset_name == 'ebnerd':
                    model_args = data_batch[:21] + data_batch[23:26]
                    scores[index: index + batch_size] = model(*model_args).squeeze(dim=1)
                else:
                    scores[index: index + batch_size] = model(*data_batch).squeeze(dim=1)
            elif config.model == "SentiRec":
                if config.dataset_name == 'MIND':
                    # data_batch has 24 elements. 0-9: user, 10-17: news, 18-19: sent, 20-21: idx, 22-23: img
                    args = list(data_batch[:10]) + [None, None, None] + list(data_batch[10:18]) + [data_batch[20], data_batch[21]]
                    kwargs = {
                        'history_sentiment': data_batch[18],
                        'candidate_sentiment': data_batch[19]
                    }
                else:
                    args = data_batch[:21] + [data_batch[23], data_batch[24]]
                    kwargs = {
                        'history_sentiment': data_batch[21],
                        'candidate_sentiment': data_batch[22]
                    }
                scores[index: index + batch_size] = model(*args, **kwargs)[0]

            elif config.model == "CPRS":
                scores[index: index + batch_size] = model(*data_batch)[0]
            elif config.model == "TCCM":
                out = model(*data_batch)
                if isinstance(out, tuple):
                    out = out[0]
                scores[index: index + batch_size] = out
            else:
                scores[index: index + batch_size] = model(*data_batch[:21])
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
        for data_batch in tqdm(dataloader):
            data_batch = [item.cuda(non_blocking=True) if isinstance(item, torch.Tensor) else item for item in data_batch]
            (user_ID, _, _, user_title_text, user_title_mask, _, _, _, _, user_history_mask, _, _, _,
             _, _, news_title_text, news_title_mask, _, _, _, _, _, _, _, _,
             history_image_embedding, candidate_image_embedding) = data_batch

            news_feature = {
                "input_ids": news_title_text,
                "attention_mask": news_title_mask,
                "input_imgs": candidate_image_embedding.unsqueeze(1),
                "image_loc": torch.zeros(news_title_text.shape[0], 1, 5).cuda(non_blocking=True),
            }
            history_feature = {
                "input_ids": user_title_text,
                "attention_mask": user_title_mask,
                "input_imgs": history_image_embedding.unsqueeze(2),
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

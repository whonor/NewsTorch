from datetime import datetime
from typing import List, Tuple, Dict, Any

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import multiprocessing
from sklearn.metrics import roc_auc_score, log_loss, accuracy_score
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm
from time import gmtime, strftime
import argparse
import pandas as pd
import numpy as np
import os

import config
from dataset_corpus_preprocessing.data_loader_unbert import MindDataset
from models.UNBERT import UNBERT
from config import Config
from models.modules.unbert.eval import dev, test
import wandb

class DataLoader(DataLoader):
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        shuffle: str = False,
        num_workers: int = 0
    ) -> None:
        super().__init__(
            dataset = dataset,
            batch_size = batch_size,
            shuffle = shuffle,
            num_workers = num_workers,
            collate_fn = dataset.collate
        )

def main(config: Config):
    dataset_path = config.dataset_path
    model = UNBERT(config)
    if config.restore is not None and os.path.isfile(config.restore):
        print("restore model from {}".format(config.restore))
        state_dict = torch.load(config.restore, map_location=torch.device('cpu'))
        st = {}
        for k in state_dict:
            if k.startswith('bert'):
                st['_model'+k[len('bert'):]] = state_dict[k]
            elif k.startswith('classifier'):
                st['_dense'+k[len('classifier'):]] = state_dict[k]
            else:
                st[k] = state_dict[k]
        model.load_state_dict(st)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(config.pretrain)
    if config.mode == "train":
        print('reading training cache...')
        train_set = MindDataset(
            dataset_path,
            tokenizer=tokenizer,
            mode='train',
            split=config.split,
            news_max_len=config.news_max_len,
            hist_max_len=config.hist_max_len,
            seq_max_len=config.seq_max_len
        )
        train_loader = DataLoader(
            dataset=train_set,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=8
        )
        print('reading dev cache...')
        dev_set = MindDataset(
            dataset_path,
            tokenizer=tokenizer,
            mode='dev',
            split=config.split,
            news_max_len=config.news_max_len,
            hist_max_len=config.hist_max_len,
            seq_max_len=config.seq_max_len
        )
        dev_loader = DataLoader(
            dataset=dev_set,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=8
        )

        loss_fn = nn.CrossEntropyLoss()
        m_optim = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config.lr)
        m_scheduler = get_linear_schedule_with_warmup(m_optim, 
                    num_warmup_steps=len(train_set)//config.batch_size*2,
                    num_training_steps=len(train_set)*config.epoch//config.batch_size)
        loss_fn.to(device)
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
            loss_fn = nn.DataParallel(loss_fn)
        print("start training...")

        best_auc = 0.0
        best_dev_epoch = 0
        epoch_not_increase = 0
        for epoch in range(config.epoch):
            avg_loss = 0.0
            batch_iterator = tqdm(train_loader, disable=False)
            for step, train_batch in enumerate(batch_iterator):
                batch_score = model(train_batch['input_ids'].to(device), 
                                    train_batch['input_mask'].to(device), 
                                    train_batch['segment_ids'].to(device),
                                    train_batch['news_segment_ids'].to(device),
                                    train_batch['sentence_ids'].to(device),
                                    train_batch['sentence_mask'].to(device),
                                    train_batch['sentence_segment_ids'].to(device),
                                    )
                batch_loss = loss_fn(batch_score, train_batch['label'].to(device))
                if torch.cuda.device_count() > 1:
                    batch_loss = batch_loss.mean()
                avg_loss += batch_loss.item()
                batch_loss.backward()
                m_optim.step()
                m_scheduler.step()
                m_optim.zero_grad()

            auc, mrr, ndcg5, ndcg10 = dev(model, dev_loader, device, config.output, is_epoch=True)
            wandb.log({"epoch": epoch + 1, "loss": avg_loss / len(train_loader)})
            wandb.log({"epoch": epoch + 1, "AUC": auc, "MRR": mrr, "nDCG@5": ndcg5, "nDCG@10": ndcg10})
            if auc > best_auc:
                best_auc = auc
                best_dev_epoch = epoch
                epoch_not_increase = 0

            else:
                epoch_not_increase += 1

            print("Epoch {}: \n".format(epoch+1))
            print('AUC : %.4f\nMRR : %.4f\nnDCG@5 : %.4f\nnDCG@10 : %.4f' % (auc, mrr, ndcg5, ndcg10))
            print('Best epoch :', best_dev_epoch)
            print('Best ' + config.dev_criterion + ' : ' + str('best_dev_' + config.dev_criterion))
            if epoch_not_increase == 0:
                torch.save({config.model: model.state_dict()}, config.model_dir + '/' + config.model + '-' + str(best_dev_epoch))
            if epoch_not_increase == config.early_stopping_epoch:
                break

        print("train success!")
        print('reading test cache...')
        test_set = MindDataset(
            dataset_path,
            tokenizer=tokenizer,
            mode='test',
            news_max_len=config.news_max_len,
            hist_max_len=config.hist_max_len,
            seq_max_len=config.seq_max_len
        )
        test_loader = DataLoader(
            dataset=test_set,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=8
        )

        # if torch.cuda.device_count() > 1:
        #     model = nn.DataParallel(model)
        auc, mrr, ndcg5, ndcg10 = dev(model, test_loader, device, config.output, is_epoch=True)
        print('AUC : %.4f\nMRR : %.4f\nnDCG@5 : %.4f\nnDCG@10 : %.4f' % (auc, mrr, ndcg5, ndcg10))
        print("test success!")
    elif config.mode == "dev":
        print('reading dev cache...')
        dev_set = MindDataset(
            dataset_path,
            tokenizer=tokenizer,
            mode='dev',
            split=config.split,
            news_max_len=config.news_max_len,
            hist_max_len=config.hist_max_len,
            seq_max_len=config.seq_max_len
        )
        dev_loader = DataLoader(
            dataset=dev_set,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=8
        )

        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        auc, mrr, ndcg5, ndcg10 = dev(model, dev_loader, device, config.output, is_epoch=True)
        print('AUC : %.4f\nMRR : %.4f\nnDCG@5 : %.4f\nnDCG@10 : %.4f' % (auc, mrr, ndcg5, ndcg10))
        print("dev success!")
    else:
        print('reading test cache...')
        test_set = MindDataset(
            dataset_path,
            tokenizer=tokenizer,
            mode='test',
            news_max_len=config.news_max_len,
            hist_max_len=config.hist_max_len,
            seq_max_len=config.seq_max_len
        )
        test_loader = DataLoader(
            dataset=test_set,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=8
        )

        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        auc, mrr, ndcg5, ndcg10 = dev(model, test_loader, device, config.output, is_epoch=True)
        print('AUC : %.4f\nMRR : %.4f\nnDCG@5 : %.4f\nnDCG@10 : %.4f' % (auc, mrr, ndcg5, ndcg10))
        print("test success!")

if __name__ == "__main__":
    wandb.login()
    config = Config()
    run = wandb.init(
        project="NewsRecTorch-UNBERT",  # Specify your project
        config=config.attribute_dict,
        mode='offline'
    )
    main(config)


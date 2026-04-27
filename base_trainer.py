import os
import shutil
import wandb
from config import Config
from dataset_corpus_preprocessing.MIND_corpus_main import MIND_Corpus, MIND_Train_Dataset
from dataset_corpus_preprocessing.EBNeRD_corpus_main import EBNeRD_Corpus, Ebnerd_Train_Dataset
from utils._evaluation import AvgMetric
from utils._evaluation import compute_scores
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


class Trainer:
    def __init__(self, model: nn.Module, config: Config, _corpus, wandb, run_index: int):
        self.wandb = wandb
        self.config = config
        self.model = model
        self.epoch = config.epoch
        self.batch_size = config.batch_size
        self.max_history_num = config.max_history_num
        self.negative_sample_num = config.negative_sample_num
        self.loss = self.negative_log_softmax if config.click_predictor in ['dot_product', 'mlp', 'FIM'] else self.negative_log_sigmoid
        self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=config.lr, weight_decay=config.weight_decay)
        self._dataset = config.dataset_name
        self._corpus = _corpus
        if config.dataset_name == 'MIND':
            if _corpus is None:
                _corpus = MIND_Corpus(config)
                self._corpus = _corpus
            if type(_corpus).__name__ == 'MIND_Corpus':
                self.train_dataset = MIND_Train_Dataset(_corpus)
        elif config.dataset_name == 'ebnerd':
            if _corpus is None:
                if config.model == 'CPRS':
                    from dataset_corpus_preprocessing.EBNeRD_corpus_CPRS import EBNeRD_Corpus as EBNeRD_Corpus_CPRS
                    _corpus = EBNeRD_Corpus_CPRS(config)
                elif config.model == 'TCCM':
                    from dataset_corpus_preprocessing.EBNeRD_corpus_TCCM import EBNeRD_Corpus as EBNeRD_Corpus_TCCM
                    _corpus = EBNeRD_Corpus_TCCM(config)
                else:
                    _corpus = EBNeRD_Corpus(config)
                self._corpus = _corpus
            
            if config.model == 'CPRS':
                from dataset_corpus_preprocessing.EBNeRD_corpus_CPRS import Ebnerd_Train_Dataset as Ebnerd_Train_Dataset_CPRS
                self.train_dataset = Ebnerd_Train_Dataset_CPRS(_corpus)
            elif config.model == 'TCCM':
                from dataset_corpus_preprocessing.EBNeRD_corpus_TCCM import Ebnerd_Train_Dataset as Ebnerd_Train_Dataset_TCCM
                self.train_dataset = Ebnerd_Train_Dataset_TCCM(_corpus)
            else:
                self.train_dataset = Ebnerd_Train_Dataset(_corpus)

        self.run_index = run_index
        self.model_dir = config.model_dir + '/#' + str(self.run_index)
        self.dev_res_dir = config.dev_res_dir + '/#' + str(self.run_index)
        self.best_model_dir = config.best_model_dir + '/#' + str(self.run_index)
        if not os.path.exists(self.dev_res_dir):
            os.mkdir(self.dev_res_dir)
        if self._dataset == 'large':
            self.prediction_dir = config.prediction_dir + '/#' + str(self.run_index)
            os.mkdir(self.prediction_dir)
        if not os.path.exists(self.best_model_dir):
            os.mkdir(self.best_model_dir)
        if not os.path.exists(self.model_dir):
            os.mkdir(self.model_dir)
        self.dev_criterion = config.dev_criterion
        self.early_stopping_epoch = config.early_stopping_epoch
        self.auc_results = []
        self.mrr_results = []
        self.ndcg5_results = []
        self.ndcg10_results = []
        self.best_dev_epoch = 0
        self.best_dev_auc = 0
        self.best_dev_mrr = 0
        self.best_dev_ndcg5 = 0
        self.best_dev_ndcg10 = 0
        self.best_dev_avg = AvgMetric(0, 0, 0, 0)
        self.epoch_not_increase = 0
        self.gradient_clip_norm = config.gradient_clip_norm
        self.model.cuda()
        print('Running : ' + self.config.model + '\t#' + str(self.run_index))


    def negative_log_softmax(self, logits):
        loss = (-torch.log_softmax(logits, dim=1).select(dim=1, index=0)).mean()
        return loss

    def negative_log_sigmoid(self, logits):
        positive_sigmoid = torch.clamp(torch.sigmoid(logits[:, 0]), min=1e-15, max=1)
        negative_sigmoid = torch.clamp(torch.sigmoid(-logits[:, 1:]), min=1e-15, max=1)
        loss = -(torch.log(positive_sigmoid).sum() + torch.log(negative_sigmoid).sum()) / logits.numel()
        return loss

    def train(self):
        model = self.model
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)

        for e in tqdm(range(1, self.epoch + 1)):
            self.train_dataset.negative_sampling()
            train_dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.batch_size // 16, pin_memory=True)
            model.train()
            epoch_loss = 0
            for data_batch in tqdm(train_dataloader):
                data_batch = [item.cuda(non_blocking=True) if isinstance(item, torch.Tensor) else item for item in data_batch]
                if self.config.model == "TANR":
                    logits, topic_pred_loss = self.model(*data_batch[:21])
                    loss = self.loss(logits) + self.config.topic_pred_loss_coef * topic_pred_loss
                elif self.config.model == "LKPNR":
                    if self.config.dataset_name == 'ebnerd':
                        model_args = data_batch[:21] + data_batch[23:25]
                        logits = self.model(*model_args)
                    else:
                        logits = self.model(*data_batch)
                    loss = self.loss(logits)
                elif self.config.model == "CPRS":
                    logits, sat_preds, s_i, valid_mask = self.model(*data_batch)
                    click_loss = self.loss(logits)
                    if valid_mask is not None and valid_mask.numel() > 0 and valid_mask.any():
                        sat_loss = torch.abs(s_i[valid_mask] - sat_preds[valid_mask]).mean()
                    else:
                        sat_loss = torch.tensor(0.0, device=logits.device)
                    lambda_coef = getattr(self.config, 'cprs_lambda', 0.3)
                    loss = click_loss + lambda_coef * sat_loss
                elif self.config.model == "TCCM":
                    logits = self.model(*data_batch)
                    if isinstance(logits, tuple):
                        logits = logits[0]
                    loss = self.loss(logits)
                else:
                    logits = self.model(*data_batch[:21])  # For other models like NRMS, NPA, etc.
                    if isinstance(logits, tuple):
                        logits = logits[0]
                    loss = self.loss(logits)

                epoch_loss += float(loss) * data_batch[0].size(0)
                self.optimizer.zero_grad()
                loss.backward()
                if self.gradient_clip_norm > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), self.gradient_clip_norm)
                self.optimizer.step()
            print('Epoch %d : train done' % e)
            print('loss =', epoch_loss / len(self.train_dataset))
            print('Epoch loss =', epoch_loss)
            self.wandb.log({'train epoch': e, 'loss': epoch_loss / len(self.train_dataset)})

            # validation
            auc, mrr, ndcg5, ndcg10, mae, rmse, recall5, recall10, hit5, hit10, precision5, precision10 = compute_scores(self.config, model, self._corpus, self.batch_size,
                                                     'dev', self.dev_res_dir + '/' + self.config.model + '-' + str(
                    e) + '.txt', self._dataset)

            self.auc_results.append(auc)
            self.mrr_results.append(mrr)
            self.ndcg5_results.append(ndcg5)
            self.ndcg10_results.append(ndcg10)
            print('Epoch %d : dev done\nDev criterions' % e)
            print('AUC = {:.4f}\nMRR = {:.4f}\nnDCG@5 = {:.4f}\nnDCG@10 = {:.4f}'.format(auc, mrr, ndcg5, ndcg10))
            self.wandb.log({'validation epoch': e, 'AUC': auc, 'MRR': mrr, 'nDCG@5': ndcg5, 'nDCG@10': ndcg10})
            avg = AvgMetric(auc, mrr, ndcg5, ndcg10)
            if avg >= self.best_dev_avg:
                self.best_dev_avg = avg
                self.best_dev_epoch = e
                self.epoch_not_increase = 0
            else:
                self.epoch_not_increase += 1

            print('Best epoch :', self.best_dev_epoch)
            print('Best ' + self.dev_criterion + ' : ' + str(getattr(self, 'best_dev_' + self.dev_criterion)))
            torch.cuda.empty_cache()
            if self.epoch_not_increase == 0:
                torch.save({self.config.model: model.state_dict()}, self.model_dir + '/' + self.config.model + '-' + str(self.best_dev_epoch))
            if self.epoch_not_increase == self.early_stopping_epoch:
                break
        if self.best_dev_epoch == 0:
            raise RuntimeError("Training finished, but no model was saved. "
                             "This usually means the validation scores (e.g., AUC, MRR) never improved beyond zero. "
                             "Please check your validation data and evaluation logic.")
        shutil.copy(self.model_dir + '/' + self.config.model + '-' + str(self.best_dev_epoch),
                    self.best_model_dir + '/' + self.config.model)

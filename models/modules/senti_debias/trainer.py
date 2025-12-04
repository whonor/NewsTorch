import os
import shutil

from dataset_corpus_preprocessing.EBNeRD_corpus_main import EBNeRD_Corpus, Ebnerd_Train_Dataset
from utils._evaluation import get_run_index, compute_scores_IPNR
from datetime import datetime
import wandb
from config import Config
from dataset_corpus_preprocessing.MIND_corpus_main import MIND_Corpus, MIND_Train_Dataset
from utils._evaluation import AvgMetric
from utils._evaluation import compute_scores
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


class TrainerSentiDebias:
    def __init__(self, model: nn.Module, config: Config, _corpus, wandb, run_index: int):
        self.wandb = wandb
        self.config = config
        self.model = model
        self.epoch = config.epoch
        self.batch_size = config.batch_size
        self.max_history_num = config.max_history_num
        self.negative_sample_num = config.negative_sample_num
        self.rec_loss = self.negative_log_softmax
        self.adv_loss = nn.CrossEntropyLoss()
        self.optimizer_g = optim.Adam(self.model.generator.parameters(), lr=config.lr_g, weight_decay=config.weight_decay)
        self.optimizer_d = optim.Adam(self.model.discriminator.parameters(), lr=config.lr_d, weight_decay=config.weight_decay)
        self._dataset = config.dataset_name
        self._corpus = _corpus
        if config.dataset_name == 'MIND':
            _corpus = MIND_Corpus(config)
            self.train_dataset = MIND_Train_Dataset(_corpus)
        elif config.dataset_name == 'ebnerd':
            _corpus = EBNeRD_Corpus(config)
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

    def train(self):
        model = self.model
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)

        for e in tqdm(range(1, self.epoch + 1)):
            if self.config.dataset_name == 'MIND':
                self.train_dataset.negative_sampling()
            train_dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.batch_size // 16, pin_memory=True)
            model.train()
            epoch_g_loss = 0
            epoch_d_loss = 0
            for (user_ID, user_category, user_subCategory, user_title_text, user_title_mask, user_title_entity, user_content_text, user_content_mask, user_content_entity, user_history_mask, user_history_graph, user_history_category_mask, user_history_category_indices, \
                news_category, news_subCategory, news_title_text, news_title_mask, news_title_entity, news_content_text, news_content_mask, news_content_entity, \
                user_hist_sentiment, news_sentiment, history_index, sample_index) in tqdm(train_dataloader):

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

                # Train Generator
                self.optimizer_g.zero_grad()
                combined_scores, bias_free_scores, loss_orth, hist_news_vector, cand_news_vector = model(user_ID, user_category, user_subCategory, user_title_text, user_title_mask, user_title_entity, user_content_text, user_content_mask, user_content_entity, user_history_mask, user_history_graph, user_history_category_mask, user_history_category_indices, \
                news_category, news_subCategory, news_title_text, news_title_mask, news_title_entity, news_content_text, news_content_mask, news_content_entity, \
                user_hist_sentiment, news_sentiment)
                
                pred_hist_sent, pred_cand_sent = model.module.discriminator(hist_news_vector.detach(), cand_news_vector.detach())
                
                rec_loss = self.rec_loss(combined_scores)
                
                adv_loss_g = self.adv_loss(pred_hist_sent.view(-1, self.config.num_sent_classes), user_hist_sentiment.long().view(-1)) \
                           + self.adv_loss(pred_cand_sent.view(-1, self.config.num_sent_classes), news_sentiment.long().view(-1))

                g_loss = rec_loss + self.config.beta_coefficient * loss_orth - self.config.alpha_coefficient * adv_loss_g
                
                epoch_g_loss += g_loss.mean().item()
                g_loss.mean().backward()
                if self.gradient_clip_norm > 0:
                    nn.utils.clip_grad_norm_(model.module.generator.parameters(), self.gradient_clip_norm)
                self.optimizer_g.step()

                # Train Discriminator
                self.optimizer_d.zero_grad()
                _, _, _, hist_news_vector, cand_news_vector = model(user_ID, user_category, user_subCategory, user_title_text, user_title_mask, user_title_entity, user_content_text, user_content_mask, user_content_entity, user_history_mask, user_history_graph, user_history_category_mask, user_history_category_indices, \
                news_category, news_subCategory, news_title_text, news_title_mask, news_title_entity, news_content_text, news_content_mask, news_content_entity, \
                user_hist_sentiment, news_sentiment)
                
                pred_hist_sent, pred_cand_sent = model.module.discriminator(hist_news_vector.detach(), cand_news_vector.detach())

                d_loss = self.adv_loss(pred_hist_sent.view(-1, self.config.num_sent_classes), user_hist_sentiment.view(-1).long()) + self.adv_loss(pred_cand_sent.view(-1, self.config.num_sent_classes), news_sentiment.view(-1).long())
                
                epoch_d_loss += d_loss.mean().item()
                d_loss.mean().backward()
                if self.gradient_clip_norm > 0:
                    nn.utils.clip_grad_norm_(model.module.discriminator.parameters(), self.gradient_clip_norm)
                self.optimizer_d.step()

            print('Epoch %d : train done' % e)
            print('g_loss =', epoch_g_loss / len(self.train_dataset))
            print('d_loss =', epoch_d_loss / len(self.train_dataset))
            self.wandb.log({'train epoch': e, 'g_loss': epoch_g_loss / len(self.train_dataset), 'd_loss': epoch_d_loss / len(self.train_dataset)})

            # validation
            # The validation step should use the bias_free_scores for evaluation
            # I will need to modify compute_scores to handle this.
            # For now, I will just use the combined scores.
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
        shutil.copy(self.model_dir + '/' + self.config.model + '-' + str(self.best_dev_epoch),
                    self.best_model_dir + '/' + self.config.model)
        print('Training : ' + self.config.model + ' #' + str(self.run_index) + ' completed\nDev criterions:')
        print('AUC : %.4f' % self.auc_results[self.best_dev_epoch - 1])
        print('MRR : %.4f' % self.mrr_results[self.best_dev_epoch - 1])
        print('nDCG@5 : %.4f' % self.ndcg5_results[self.best_dev_epoch - 1])
        print('nDCG@10 : %.4f' % self.ndcg10_results[self.best_dev_epoch - 1])
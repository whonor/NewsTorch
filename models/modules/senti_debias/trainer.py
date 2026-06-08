import os
import shutil

from dataset_corpus_preprocessing.EBNeRD_corpus_main import EBNeRD_Corpus, Ebnerd_Train_Dataset
from utils._evaluation import get_run_index, compute_scores_IPNR
from datetime import datetime
import wandb
from config import Config
from dataset_corpus_preprocessing.MIND_corpus_main import MIND_Corpus, MIND_Train_Dataset
from dataset_corpus_preprocessing.MIND_corpus_SentiDebias import MIND_Corpus_SentiDebias, MIND_Train_Dataset_SentiDebias
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
            if _corpus is None:
                _corpus = MIND_Corpus_SentiDebias(config)
            self.train_dataset = MIND_Train_Dataset_SentiDebias(_corpus)
        elif config.dataset_name == 'ebnerd':
            if _corpus is None:
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

    def discretize_sentiment(self, sentiment_scores):
        # Discretize VADER compound scores (-1 to 1) into 3 classes:
        # 0: Negative (<-0.05), 1: Neutral (-0.05 to 0.05), 2: Positive (>0.05)
        # Input shape: [batch_size] or [batch_size, num_items]
        labels = torch.ones_like(sentiment_scores, dtype=torch.long) # Default to Neutral (1)
        labels[sentiment_scores < -0.05] = 0
        labels[sentiment_scores > 0.05] = 2
        return labels

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
            for batch in tqdm(train_dataloader):
                user_ID = batch[0]
                user_category = batch[1]
                user_subCategory = batch[2]
                user_title_text = batch[3]
                user_title_mask = batch[4]
                user_title_entity = batch[5]
                user_content_text = batch[6]
                user_content_mask = batch[7]
                user_content_entity = batch[8]
                user_history_mask = batch[9]
                
                if self.config.dataset_name == 'ebnerd':
                    # Indices 10, 11, 12 are graph-related
                    # user_history_graph = batch[10]
                    # user_history_category_mask = batch[11]
                    # user_history_category_indices = batch[12]
                    
                    news_category = batch[13]
                    news_subCategory = batch[14]
                    news_title_text = batch[15]
                    news_title_mask = batch[16]
                    news_title_entity = batch[17]
                    news_content_text = batch[18]
                    news_content_mask = batch[19]
                    news_content_entity = batch[20]
                    
                    user_hist_sentiment = batch[21]
                    news_sentiment = batch[22]
                else:
                    # MIND (SentiDebias custom)
                    news_category = batch[10]
                    news_subCategory = batch[11]
                    news_title_text = batch[12]
                    news_title_mask = batch[13]
                    news_title_entity = batch[14]
                    news_content_text = batch[15]
                    news_content_mask = batch[16]
                    news_content_entity = batch[17]
                    
                    user_hist_sentiment = batch[18]
                    news_sentiment = batch[19]

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

                # Prepare discrete sentiment labels for discriminator loss
                user_hist_sent_labels = self.discretize_sentiment(user_hist_sentiment)
                news_sent_labels = self.discretize_sentiment(news_sentiment)

                # Train Generator
                self.optimizer_g.zero_grad()
                combined_scores, bias_free_scores, loss_orth, hist_news_vector, cand_news_vector = model(user_ID, user_category, user_subCategory, user_title_text, user_title_mask, user_title_entity, user_content_text, user_content_mask, user_content_entity, user_history_mask, None, None, None, \
                news_category, news_subCategory, news_title_text, news_title_mask, news_title_entity, news_content_text, news_content_mask, news_content_entity, \
                user_hist_sent_labels, news_sent_labels)

                pred_hist_sent, pred_cand_sent = model.discriminator(hist_news_vector, cand_news_vector)
                
                rec_loss = self.rec_loss(combined_scores)
                
                adv_loss_g = self.adv_loss(pred_hist_sent.view(-1, self.config.num_sent_classes), user_hist_sent_labels.view(-1)) \
                           + self.adv_loss(pred_cand_sent.view(-1, self.config.num_sent_classes), news_sent_labels.view(-1))

                g_loss = rec_loss + self.config.beta_coefficient * loss_orth - self.config.alpha_coefficient * adv_loss_g
                
                epoch_g_loss += g_loss.mean().item()
                g_loss.mean().backward()
                if self.gradient_clip_norm > 0:
                    nn.utils.clip_grad_norm_(model.generator.parameters(), self.gradient_clip_norm)
                self.optimizer_g.step()

                # Train Discriminator
                self.optimizer_d.zero_grad()
                _, _, _, hist_news_vector, cand_news_vector = model(user_ID, user_category, user_subCategory, user_title_text, user_title_mask, user_title_entity, user_content_text, user_content_mask, user_content_entity, user_history_mask, None, None, None, \
                news_category, news_subCategory, news_title_text, news_title_mask, news_title_entity, news_content_text, news_content_mask, news_content_entity, \
                user_hist_sent_labels, news_sent_labels)
                
                pred_hist_sent, pred_cand_sent = model.discriminator(hist_news_vector.detach(), cand_news_vector.detach())

                d_loss = self.adv_loss(pred_hist_sent.view(-1, self.config.num_sent_classes), user_hist_sent_labels.view(-1)) + self.adv_loss(pred_cand_sent.view(-1, self.config.num_sent_classes), news_sent_labels.view(-1))
                
                epoch_d_loss += d_loss.mean().item()
                d_loss.mean().backward()
                if self.gradient_clip_norm > 0:
                    nn.utils.clip_grad_norm_(model.discriminator.parameters(), self.gradient_clip_norm)
                self.optimizer_d.step()

            print('Epoch %d : train done' % e)
            print('g_loss =', epoch_g_loss / len(self.train_dataset))
            print('d_loss =', epoch_d_loss / len(self.train_dataset))
            self.wandb.log({'train epoch': e, 'g_loss': epoch_g_loss / len(self.train_dataset), 'd_loss': epoch_d_loss / len(self.train_dataset)})

            # validation
            # The validation step uses the bias_free_scores for evaluation (updated in compute_scores)
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

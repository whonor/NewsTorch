import shutil

from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from base_trainer import Trainer
from utils._evaluation import compute_scores, AvgMetric
from dataset_corpus_preprocessing.MIND_corpus_SentiRec import MIND_Train_Dataset_SentiRec


class TrainerSentiRec(Trainer):
    def __init__(self, model: nn.Module, config, corpus, wandb, run_index: int):
        super().__init__(model, config, corpus, wandb, run_index)
        if config.dataset_name == 'MIND':
            self.train_dataset = MIND_Train_Dataset_SentiRec(corpus)

    def train(self):
        model = self.model
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)

        for e in tqdm(range(1, self.epoch + 1)):
            self.train_dataset.negative_sampling()
            
            train_dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.batch_size // 16 if self.batch_size > 16 else 0, pin_memory=True)
            model.train()
            epoch_loss = 0

            for data_tuple in tqdm(train_dataloader):
                # 1. Unpack data based on dataset
                if self.config.dataset_name == 'ebnerd':
                    (user_ID, user_category, user_subCategory, user_title_text, user_title_mask, user_title_entity, user_content_text, user_content_mask, user_content_entity, user_history_mask, user_history_graph, user_history_category_mask, user_history_category_indices,
                     news_category, news_subCategory, news_title_text, news_title_mask, news_title_entity, news_content_text, news_content_mask, news_content_entity, history_sentiment, candidate_sentiment, history_index, sample_index, _, _) = data_tuple
                elif self.config.dataset_name == 'MIND':
                    (user_ID, user_category, user_subCategory, user_title_text, user_title_mask, user_title_entity, user_content_text, user_content_mask, user_content_entity, user_history_mask,
                     news_category, news_subCategory, news_title_text, news_title_mask, news_title_entity, news_content_text, news_content_mask, news_content_entity, history_sentiment, candidate_sentiment, history_index, sample_index, _, _) = data_tuple
                    
                    user_history_graph = None
                    user_history_category_mask = None
                    user_history_category_indices = None
                else:
                    raise ValueError("SentiRec model is only supported for ebnerd and MIND datasets.")

                # 2. Move tensors to CUDA
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
                if user_history_graph is not None:
                    user_history_graph = user_history_graph.cuda(non_blocking=True)
                if user_history_category_mask is not None:
                    user_history_category_mask = user_history_category_mask.cuda(non_blocking=True)
                if user_history_category_indices is not None:
                    user_history_category_indices = user_history_category_indices.cuda(non_blocking=True)
                news_category = news_category.cuda(non_blocking=True)
                news_subCategory = news_subCategory.cuda(non_blocking=True)
                news_title_text = news_title_text.cuda(non_blocking=True)
                news_title_mask = news_title_mask.cuda(non_blocking=True)
                news_title_entity = news_title_entity.cuda(non_blocking=True)
                news_content_text = news_content_text.cuda(non_blocking=True)
                news_content_mask = news_content_mask.cuda(non_blocking=True)
                news_content_entity = news_content_entity.cuda(non_blocking=True)
                history_sentiment = history_sentiment.cuda(non_blocking=True)
                candidate_sentiment = candidate_sentiment.cuda(non_blocking=True)

                # 3. Model forward pass and loss calculation for SentiRec
                click_logits, l_senti, l_div = model(
                    user_ID, user_category, user_subCategory, user_title_text, user_title_mask, user_title_entity, user_content_text, user_content_mask, user_content_entity, user_history_mask, user_history_graph, user_history_category_mask, user_history_category_indices,
                    news_category, news_subCategory, news_title_text, news_title_mask, news_title_entity, news_content_text, news_content_mask, news_content_entity,
                    history_index, sample_index, history_sentiment=history_sentiment, candidate_sentiment=candidate_sentiment
                )
                
                l_rec = self.loss(click_logits)
                
                # Average the losses from different GPUs when using DataParallel
                l_senti = l_senti.mean()
                l_div = l_div.mean()
                loss = l_rec + self.config.sent_pred_loss_coef * l_senti + self.config.sent_div_loss_coef * l_div

                # 4. Backpropagation
                epoch_loss += float(loss) * user_ID.size(0)
                self.optimizer.zero_grad()
                loss.backward()
                if self.gradient_clip_norm > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), self.gradient_clip_norm)
                self.optimizer.step()

            print('Epoch %d : train done' % e)
            print('loss =', epoch_loss / len(self.train_dataset))
            self.wandb.log({'train epoch': e, 'loss': epoch_loss / len(self.train_dataset)})

            # validation (uses the same logic as the base Trainer)
            auc, mrr, ndcg5, ndcg10, mae, rmse, recall5, recall10, hit5, hit10, precision5, precision10, tce5, tce10 = compute_scores(self.config, model, self._corpus, self.batch_size,
                                                     'dev', self.dev_res_dir + '/' + self.config.model + '-' + str(
                    e) + '.txt', self._dataset)
            
            self.auc_results.append(auc)
            self.mrr_results.append(mrr)
            self.ndcg5_results.append(ndcg5)
            self.ndcg10_results.append(ndcg10)
            print('Epoch %d : dev done\nDev criterions' % e)
            print('AUC = {:.4f}\nMRR = {:.4f}\nnDCG@5 = {:.4f}\nnDCG@10 = {:.4f}\nTCE@5 = {:.4f}\nTCE@10 = {:.4f}'.format(auc, mrr, ndcg5, ndcg10, tce5, tce10))
            self.wandb.log({'validation epoch': e, 'AUC': auc, 'MRR': mrr, 'nDCG@5': ndcg5, 'nDCG@10': ndcg10, 'TCE@5': tce5, 'TCE@10': tce10})
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
        print('Training : ' + self.config.model + ' #' + str(self.run_index) + ' completed\nDev criterions:')
        print('AUC : %.4f' % self.auc_results[self.best_dev_epoch - 1])
        print('MRR : %.4f' % self.mrr_results[self.best_dev_epoch - 1])
        print('nDCG@5 : %.4f' % self.ndcg5_results[self.best_dev_epoch - 1])
        print('nDCG@10 : %.4f' % self.ndcg10_results[self.best_dev_epoch - 1])

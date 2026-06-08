import shutil

from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from base_trainer import Trainer
from utils._evaluation import compute_scores_mmrec, AvgMetric


class TrainerMMRec(Trainer):
    def __init__(self, model: nn.Module, config, corpus, wandb, run_index: int):
        super().__init__(model, config, corpus, wandb, run_index)

    def train(self):
        model = self.model
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)

        for e in tqdm(range(1, self.epoch + 1)):
            if hasattr(self.train_dataset, 'negative_sampling'):
                self.train_dataset.negative_sampling()
            
            train_dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.batch_size // 16, pin_memory=True)
            model.train()
            epoch_loss = 0

            for data_tuple in tqdm(train_dataloader):
                data_tuple = [item.cuda(non_blocking=True) if isinstance(item, torch.Tensor) else item for item in data_tuple]
                (user_ID, user_category, user_subCategory, user_title_text, user_title_mask, _, _, _, _, user_history_mask, _, _, _,
                 news_category, news_subCategory, news_title_text, news_title_mask, _, _, _, _, _, _, _, _,
                 history_image_embedding, candidate_image_embedding) = data_tuple

                news_feature = {
                    "input_ids": news_title_text,
                    "attention_mask": news_title_mask,
                    "category": news_category,
                    "subCategory": news_subCategory,
                    "input_imgs": candidate_image_embedding.unsqueeze(2),
                    "image_loc": torch.zeros(news_title_text.shape[0], news_title_text.shape[1], 1, 5).cuda(non_blocking=True),
                }
                history_feature = {
                    "input_ids": user_title_text,
                    "attention_mask": user_title_mask,
                    "category": user_category,
                    "subCategory": user_subCategory,
                    "input_imgs": history_image_embedding.unsqueeze(2),
                    "image_loc": torch.zeros(user_title_text.shape[0], user_title_text.shape[1], 1, 5).cuda(non_blocking=True),
                }

                log_mask = user_history_mask
                targets = torch.zeros(user_ID.size(0), dtype=torch.long).cuda(non_blocking=True)

                loss, score = model(news_feature, history_feature, log_mask, targets, compute_loss=True)
                
                loss = loss.mean() # Average loss for data parallel
                
                epoch_loss += float(loss) * user_ID.size(0)
                self.optimizer.zero_grad()
                loss.backward()
                if self.gradient_clip_norm > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), self.gradient_clip_norm)
                self.optimizer.step()

            print('Epoch %d : train done' % e)
            print('loss =', epoch_loss / len(self.train_dataset))
            self.wandb.log({'train epoch': e, 'loss': epoch_loss / len(self.train_dataset)})

            # validation
            auc, mrr, ndcg5, ndcg10, mae, rmse, recall5, recall10, hit5, hit10, precision5, precision10 = compute_scores_mmrec(self.config, model, self._corpus, self.batch_size,
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
            raise RuntimeError("Training finished, but no model was saved. ")
        shutil.copy(self.model_dir + '/' + self.config.model + '-' + str(self.best_dev_epoch),
                    self.best_model_dir + '/' + self.config.model)
        print('Training : ' + self.config.model + ' #' + str(self.run_index) + ' completed\nDev criterions:')
        print('AUC : %.4f' % self.auc_results[self.best_dev_epoch - 1])
        print('MRR : %.4f' % self.mrr_results[self.best_dev_epoch - 1])
        print('nDCG@5 : %.4f' % self.ndcg5_results[self.best_dev_epoch - 1])
        print('nDCG@10 : %.4f' % self.ndcg10_results[self.best_dev_epoch - 1])

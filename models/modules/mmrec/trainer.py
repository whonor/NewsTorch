import shutil

from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from base_trainer import Trainer
from utils._evaluation import compute_scores, AvgMetric


class TrainerMMRec(Trainer):
    def __init__(self, model: nn.Module, config, corpus, wandb, run_index: int):
        super().__init__(model, config, corpus, wandb, run_index)

    def train(self):
        model = self.model
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)

        for e in tqdm(range(1, self.epoch + 1)):
            if self.config.dataset_name == 'MIND':
                 self.train_dataset.negative_sampling() # MMRec uses ebnerd which doesn't need this
            
            train_dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.batch_size // 16, pin_memory=True)
            model.train()
            epoch_loss = 0

            for data_tuple in tqdm(train_dataloader):
                # Unpack data for ebnerd dataset (assuming a multimodal version)
                # This part is an assumption and might need to be adjusted based on the actual dataset implementation.
                # Based on the MMRec model, we expect:
                # news_feature: a dict of tensors for the news encoder
                # input_ids: candidate news indices
                # log_ids: history news indices
                # log_mask: history mask
                # targets: click labels
                
                # NOTE: The following unpacking is a placeholder based on SentiRec's trainer.
                # It will need to be adapted for MMRec's data format, especially for image features.
                (user_ID, user_category, user_subCategory, user_title_text, user_title_mask, user_title_entity, user_content_text, user_content_mask, user_content_entity, user_history_mask, user_history_graph, user_history_category_mask, user_history_category_indices,
                 news_category, news_subCategory, news_title_text, news_title_mask, news_title_entity, news_content_text, news_content_mask, news_content_entity, 
                 # Assuming the dataloader for ebnerd with images will provide these:
                 news_images, news_image_locs,
                 history_index, sample_index, targets) = data_tuple

                # Create the news_feature dictionary for the model
                news_feature = {
                    "input_txt": news_title_text.cuda(non_blocking=True),
                    "input_imgs": news_images.cuda(non_blocking=True),
                    "image_loc": news_image_locs.cuda(non_blocking=True),
                    "attention_mask": news_title_mask.cuda(non_blocking=True),
                    # Other masks and token_type_ids can be added if needed
                }
                
                input_ids = sample_index.cuda(non_blocking=True)
                log_ids = history_index.cuda(non_blocking=True)
                log_mask = user_history_mask.cuda(non_blocking=True)
                targets = targets.cuda(non_blocking=True)


                loss, score = model(news_feature, input_ids, log_ids, log_mask, targets, compute_loss=True)
                
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
            raise RuntimeError("Training finished, but no model was saved. ")
        shutil.copy(self.model_dir + '/' + self.config.model + '-' + str(self.best_dev_epoch),
                    self.best_model_dir + '/' + self.config.model)
        print('Training : ' + self.config.model + ' #' + str(self.run_index) + ' completed\nDev criterions:')
        print('AUC : %.4f' % self.auc_results[self.best_dev_epoch - 1])
        print('MRR : %.4f' % self.mrr_results[self.best_dev_epoch - 1])
        print('nDCG@5 : %.4f' % self.ndcg5_results[self.best_dev_epoch - 1])
        print('nDCG@10 : %.4f' % self.ndcg10_results[self.best_dev_epoch - 1])


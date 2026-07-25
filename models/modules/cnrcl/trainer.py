import os
import shutil
import json
import wandb
from config import Config
from models.modules.cnrcl.dataset import CNRCL_Ebnerd_Train_Dataset
from models.modules.cnrcl.utils import NewsRecSupConLoss_with_neg_pos
from models.modules.cnrcl.pcgrad import PCGrad
from utils._evaluation import AvgMetric, compute_scores
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

class TrainerCNRCL:
    def __init__(self, model: nn.Module, config: Config, corpus, wandb, run_index: int):
        self.config = config
        self.model = model
        self.epoch = config.epoch
        self.batch_size = config.batch_size
        self.max_history_num = config.max_history_num
        self.negative_sample_num = config.negative_sample_num
        self.loss = self.negative_log_softmax if config.click_predictor in ['dot_product', 'mlp', 'FIM'] else self.negative_log_sigmoid
        
        # CNRCL uses PCGrad
        self.optimizer = PCGrad(optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=config.lr, weight_decay=config.weight_decay, eps=1e-4))
        
        self.augument_loss = NewsRecSupConLoss_with_neg_pos()
        
        self._dataset = config.dataset_size
        self.corpus = corpus
        # Use our custom dataset
        self.train_dataset = CNRCL_Ebnerd_Train_Dataset(corpus)
        
        self.run_index = run_index
        self.model_dir = config.model_dir + '/#' + str(self.run_index)
        self.best_model_dir = config.best_model_dir + '/#' + str(self.run_index)
        self.dev_res_dir = config.dev_res_dir + '/#' + str(self.run_index)
        self.result_dir = config.result_dir
        
        if not os.path.exists(self.model_dir):
            os.mkdir(self.model_dir)
        if not os.path.exists(self.best_model_dir):
            os.mkdir(self.best_model_dir)
        if not os.path.exists(self.dev_res_dir):
            os.mkdir(self.dev_res_dir)
            
        if self._dataset == 'large':
            self.prediction_dir = config.prediction_dir + '/#' + str(self.run_index)
            os.mkdir(self.prediction_dir)
            
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
        print('Running : ' + config.model + '\t#' + str(self.run_index))

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
        for e in tqdm(range(1, self.epoch + 1)):
            self.train_dataset.negative_sampling(epoch=e)
            train_dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
            model.train()
            epoch_loss = 0
            
            for data_batch in tqdm(train_dataloader):
                data_batch = [item.cuda(non_blocking=True) if isinstance(item, torch.Tensor) else item for item in data_batch]
                
                # CNRCL.forward returns (logits, logits_unsum) during training
                # Slice input to match forward signature (first 21 args)
                logits, logits_unsum = model(*data_batch[:21]) 
                
                loss = self.loss(logits)
                
                if model.news_encoder.auxiliary_loss is not None:
                    news_auxiliary_loss = model.news_encoder.auxiliary_loss.mean()
                    loss += news_auxiliary_loss
                if model.user_encoder.auxiliary_loss is not None:
                    user_encoder_auxiliary_loss = model.user_encoder.auxiliary_loss.mean()
                    loss += user_encoder_auxiliary_loss
                
                epoch_loss += float(loss) * data_batch[0].size(0)
                
                self.optimizer.zero_grad()
                
                # Compute contrastive loss
                # Note: negative_sample_num is from config
                sloss = self.augument_loss(logits_unsum, self.negative_sample_num)
                losses = [loss, sloss]
                
                self.optimizer.pc_backward(losses)
                self.optimizer.step()
                
                if self.gradient_clip_norm > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), self.gradient_clip_norm)
                    
            print('Epoch %d : train done' % e)
            print('loss =', epoch_loss / len(self.train_dataset))
            wandb.log({'epoch': e, 'train_loss': epoch_loss / len(self.train_dataset)}, step=e)

            # validation
            # Standard compute_scores should work now as forward returns only logits when not training
            auc, mrr, ndcg5, ndcg10, mae, rmse, recall5, recall10, hit5, hit10, precision5, precision10, categ_div5, categ_div10, categ_pers5, categ_pers10 = compute_scores(
                self.config, model, self.corpus, self.batch_size * 3 // 2, 'dev', 
                self.dev_res_dir + '/' + model.model_name + '-' + str(e) + '.txt', self._dataset
            )
            
            self.auc_results.append(auc)
            self.mrr_results.append(mrr)
            self.ndcg5_results.append(ndcg5)
            self.ndcg10_results.append(ndcg10)
            
            print('Epoch %d : dev done\nDev criterions' % e)
            print('AUC = {:.4f}\nMRR = {:.4f}\nnDCG@5 = {:.4f}\nnDCG@10 = {:.4f}\nMAE = {:.4f}\nRMSE = {:.4f}\nRecall@5 = {:.4f}\nRecall@10 = {:.4f}\nHit Rate@5 = {:.4f}\nHit Rate@10 = {:.4f}'.format(auc, mrr, ndcg5, ndcg10, mae, rmse, recall5, recall10, hit5, hit10))
            print('Categ-Div@5 = {:.4f}\nCateg-Div@10 = {:.4f}\nCateg-Pers@5 = {:.4f}\nCateg-Pers@10 = {:.4f}'.format(categ_div5, categ_div10, categ_pers5, categ_pers10))
            wandb.log({'epoch': e, 'dev_auc': auc, 'dev_mrr': mrr, 'dev_ndcg5': ndcg5, 'dev_ndcg10': ndcg10, 'dev_mae': mae, 'dev_rmse': rmse, 'dev_recall5': recall5, 'dev_recall10': recall10, 'dev_hit_rate5': hit5, 'dev_hit_rate10': hit10, 'dev_categ_div5': categ_div5, 'dev_categ_div10': categ_div10, 'dev_categ_pers5': categ_pers5, 'dev_categ_pers10': categ_pers10}, step=e)
            
            avg = AvgMetric(auc, mrr, ndcg5, ndcg10)
            if avg >= self.best_dev_avg:
                self.best_dev_avg = avg
                self.best_dev_epoch = e
                # save
                with open(self.result_dir + '/#' + str(self.run_index) + '-dev', 'w') as result_f:
                    result_f.write('#' + str(self.run_index) + '\t' + str(auc) + '\t' + str(mrr) + '\t' + str(ndcg5) + '\t' + str(ndcg10) + '\n')
                self.epoch_not_increase = 0
            else:
                self.epoch_not_increase += 1

            print('Best epoch :', self.best_dev_epoch)
            print('Best avg : ' + str(self.best_dev_avg))
            
            torch.cuda.empty_cache()
            if self.epoch_not_increase == 0:
                torch.save({model.model_name: model.state_dict()}, self.model_dir + '/' + model.model_name + '-' + str(self.best_dev_epoch))
            if self.epoch_not_increase == self.early_stopping_epoch:
                break

        shutil.copy(self.model_dir + '/' + model.model_name + '-' + str(self.best_dev_epoch), self.best_model_dir + '/' + model.model_name)
        print('Training : ' + model.model_name + ' #' + str(self.run_index) + ' completed')

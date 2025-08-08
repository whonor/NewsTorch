import os
import shutil

from dataset_corpus_preprocessing.EBNeRD_corpus_main import EBNeRD_Corpus, Ebnerd_Train_Dataset
from models.CNE_SUE import CNE_SUE
from models.DKN import DKN
from models.FIM import FIM
from models.IPNR import IPNR
from models.LSTUR import LSTUR
from models.MINS import MINS
from models.NAML import NAML
from models.NPA import NPA
from models.NRMS import NRMS
from models.TANR import TANR
from models.CenNewsRec import CenNewsRec
from models.modules.ipnr.trainer import TrainerIPNR
from utils.util import get_run_index, compute_scores_IPNR
from datetime import datetime
import wandb
from config import Config
from dataset_corpus_preprocessing.MIND_corpus_main import MIND_Corpus, MIND_Train_Dataset
from utils.util import AvgMetric
from utils.util import compute_scores
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

    def negative_log_sigmoid(self, logits):
        positive_sigmoid = torch.clamp(torch.sigmoid(logits[:, 0]), min=1e-15, max=1)
        negative_sigmoid = torch.clamp(torch.sigmoid(-logits[:, 1:]), min=1e-15, max=1)
        loss = -(torch.log(positive_sigmoid).sum() + torch.log(negative_sigmoid).sum()) / logits.numel()
        return loss

    def train(self):
        model = self.model
        if self.config.multi_gpu:
            model = nn.DataParallel(model, device_ids=self.config.device_id)
        for e in tqdm(range(1, self.epoch + 1)):
            if self.config.dataset_name == 'MIND':
                self.train_dataset.negative_sampling()
            train_dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.batch_size // 16, pin_memory=True)
            model.train()
            epoch_loss = 0
            for (user_ID, user_category, user_subCategory, user_title_text, user_title_mask, user_title_entity, user_content_text, user_content_mask, user_content_entity, user_history_mask, user_history_graph, user_history_category_mask, user_history_category_indices, \
                news_category, news_subCategory, news_title_text, news_title_mask, news_title_entity, news_content_text, news_content_mask, news_content_entity, history_index, sample_index) in train_dataloader:
                user_ID = user_ID.cuda(non_blocking=True)                                                                                                                       # [batch_size]
                user_category = user_category.cuda(non_blocking=True)                                                                                                           # [batch_size, max_history_num]
                user_subCategory = user_subCategory.cuda(non_blocking=True)                                                                                                     # [batch_size, max_history_num]
                user_title_text = user_title_text.cuda(non_blocking=True)                                                                                                       # [batch_size, max_history_num, max_title_length]
                user_title_mask = user_title_mask.cuda(non_blocking=True)                                                                                                       # [batch_size, max_history_num, max_title_length]
                user_title_entity = user_title_entity.cuda(non_blocking=True)                                                                                                   # [batch_size, max_history_num, max_title_length]
                user_content_text = user_content_text.cuda(non_blocking=True)                                                                                                   # [batch_size, max_history_num, max_content_length]
                user_content_mask = user_content_mask.cuda(non_blocking=True)                                                                                                   # [batch_size, max_history_num, max_content_length]
                user_content_entity = user_content_entity.cuda(non_blocking=True)                                                                                               # [batch_size, max_history_num, max_content_length]
                user_history_mask = user_history_mask.cuda(non_blocking=True)                                                                                                   # [batch_size, max_history_num]
                user_history_graph = user_history_graph.cuda(non_blocking=True)                                                                                                 # [batch_size, max_history_num, max_history_num]
                user_history_category_mask = user_history_category_mask.cuda(non_blocking=True)                                                                                 # [batch_size, category_num + 1]
                user_history_category_indices = user_history_category_indices.cuda(non_blocking=True)                                                                           # [batch_size, max_history_num]
                news_category = news_category.cuda(non_blocking=True)                                                                                                           # [batch_size, 1 + negative_sample_num]
                news_subCategory = news_subCategory.cuda(non_blocking=True)                                                                                                     # [batch_size, 1 + negative_sample_num]
                news_title_text = news_title_text.cuda(non_blocking=True)                                                                                                       # [batch_size, 1 + negative_sample_num, max_title_length]
                news_title_mask = news_title_mask.cuda(non_blocking=True)                                                                                                       # [batch_size, 1 + negative_sample_num, max_title_length]
                news_title_entity = news_title_entity.cuda(non_blocking=True)                                                                                                   # [batch_size, 1 + negative_sample_num, max_title_length]
                news_content_text = news_content_text.cuda(non_blocking=True)                                                                                                   # [batch_size, 1 + negative_sample_num, max_content_length]
                news_content_mask = news_content_mask.cuda(non_blocking=True)                                                                                                   # [batch_size, 1 + negative_sample_num, max_content_length]
                news_content_entity = news_content_entity.cuda(non_blocking=True)                                                                                               # [batch_size, 1 + negative_sample_num, max_content_length]

                if self.config.model == "TANR":
                    logits, topic_pred_loss = model(user_ID, user_category, user_subCategory, user_title_text, user_title_mask,
                                   user_title_entity, user_content_text, user_content_mask, user_content_entity,
                                   user_history_mask, user_history_graph, user_history_category_mask,
                                   user_history_category_indices, \
                                   news_category, news_subCategory, news_title_text, news_title_mask, news_title_entity,
                                   news_content_text, news_content_mask,
                                   news_content_entity)  # [batch_size, 1 + negative_sample_num]
                    # topic classification loss
                    loss = self.loss(logits) + self.config.topic_pred_loss_coef * topic_pred_loss
                elif config.model == "LKPNR":
                    logits = model(user_ID, user_category, user_subCategory, user_title_text, user_title_mask,
                                   user_title_entity, user_content_text, user_content_mask, user_content_entity,
                                   user_history_mask, user_history_graph, user_history_category_mask,
                                   user_history_category_indices, \
                                   news_category, news_subCategory, news_title_text, news_title_mask, news_title_entity,
                                   news_content_text, news_content_mask, news_content_entity, history_index,
                                   sample_index)
                    loss = self.loss(logits)
                else:
                    logits = model(user_ID, user_category, user_subCategory, user_title_text, user_title_mask,
                                   user_title_entity, user_content_text, user_content_mask, user_content_entity,
                                   user_history_mask, user_history_graph, user_history_category_mask,
                                   user_history_category_indices, \
                                   news_category, news_subCategory, news_title_text, news_title_mask, news_title_entity,
                                   news_content_text, news_content_mask,
                                   news_content_entity)  # [batch_size, 1 + negative_sample_num]
                    loss = self.loss(logits)

                epoch_loss += float(loss) * user_ID.size(0)
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
            auc, mrr, ndcg5, ndcg10 = compute_scores(self.config, model, self._corpus, self.batch_size,
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
        shutil.copy(self.model_dir + '/' + model.model_name + '-' + str(self.best_dev_epoch),
                    self.best_model_dir + '/' + model.model_name)
        print('Training : ' + self.config.model + ' #' + str(self.run_index) + ' completed\nDev criterions:')
        print('AUC : %.4f' % self.auc_results[self.best_dev_epoch - 1])
        print('MRR : %.4f' % self.mrr_results[self.best_dev_epoch - 1])
        print('nDCG@5 : %.4f' % self.ndcg5_results[self.best_dev_epoch - 1])
        print('nDCG@10 : %.4f' % self.ndcg10_results[self.best_dev_epoch - 1])


def negative_log_softmax(logits):
    loss = (-torch.log_softmax(logits, dim=1).select(dim=1, index=0)).mean()
    return loss

def negative_log_sigmoid(logits):
    positive_sigmoid = torch.clamp(torch.sigmoid(logits[:, 0]), min=1e-15, max=1)
    negative_sigmoid = torch.clamp(torch.sigmoid(-logits[:, 1:]), min=1e-15, max=1)
    loss = -(torch.log(positive_sigmoid).sum() + torch.log(negative_sigmoid).sum()) / logits.numel()
    return loss


def train(config: Config, corpus, wandb):
    if config.model == 'TANR':
        model = TANR(config)
    elif config.model == 'NAML':
        model = NAML(config)
    elif config.model == 'DKN':
        model = DKN(config)
    elif config.model == 'NRMS':
        model = NRMS(config)
    elif config.model == 'LSTUR':
        model = LSTUR(config)
    elif config.model == 'NPA':
        model = NPA(config)
    elif config.model == 'FIM':
        model = FIM(config)
    elif config.model == 'MINS':
        model = MINS(config)
    elif config.model == 'CENNEWSREC':
        model = CenNewsRec(config)
    elif config.model == 'IPNR':
        model = IPNR(config)
    elif config.model == 'CNE_SUE':
        model = CNE_SUE(config)
    model.initialize()
    run_index = get_run_index(config.result_dir)
    if config.dataset_name == 'MIND':
        if config.model == 'IPNR':
            trainer = TrainerIPNR(model, config, corpus, wandb, run_index)
        else:
            trainer = Trainer(model, config, corpus, wandb, run_index)
    elif config.dataset_name == 'ebnerd':
        if config.model == 'IPNR':
            trainer = TrainerIPNR(model, config, corpus, wandb, run_index)
        else:
            trainer = Trainer(model, config, corpus, wandb, run_index)


    trainer.train()
    config.run_index = run_index


def dev(config: Config, corpus):
    if config.model == 'TANR':
        model = TANR(config)
    elif config.model == 'NAML':
        model = NAML(config)
    elif config.model == 'DKN':
        model = DKN(config)
    elif config.model == 'NRMS':
        model = NRMS(config)
    elif config.model == 'LSTUR':
        model = LSTUR(config)
    elif config.model == 'NPA':
        model = NPA(config)
    elif config.model == 'FIM':
        model = FIM(config)
    elif config.model == 'MINS':
        model = MINS(config)
    elif config.model == 'CENNEWSREC':
        model = CenNewsRec(config)
    elif config.model == 'IPNR':
        model = IPNR(config)
    elif config.model == 'CNE_SUE':
        model = CNE_SUE(config)
    assert os.path.exists(config.dev_model_path), 'Dev model does not exist : ' + config.dev_model_path
    model.load_state_dict(torch.load(config.dev_model_path, map_location=torch.device('cpu'))[model.model_name])
    model.cuda()
    dev_res_dir = os.path.join(config.dev_res_dir, config.dev_model_path.replace('\\', '_').replace('/', '_'))
    if not os.path.exists(dev_res_dir):
        os.mkdir(dev_res_dir)
    if config.dataset_name == 'MIND':
        if config.model == 'IPNR':
            auc, mrr, ndcg5, ndcg10 = compute_scores_IPNR(config, model, corpus, config.batch_size, 'dev',
                                                          dev_res_dir + '/' + config.model + '.txt', config.dataset)
        else:
            auc, mrr, ndcg5, ndcg10 = compute_scores(config, model, corpus, config.batch_size,
                                                     'dev',
                                                     dev_res_dir + '/' + config.model + '.txt', config.dataset)
    elif config.dataset_name == 'ebnerd':
        auc, mrr, ndcg5, ndcg10 = compute_scores(config, model, corpus, config.batch_size,
                                                 'dev',
                                                 dev_res_dir + '/' + config.model + '.txt', config.dataset)



    print('Dev : ' + config.dev_model_path)
    print('AUC : %.4f\nMRR : %.4f\nnDCG@5 : %.4f\nnDCG@10 : %.4f' % (auc, mrr, ndcg5, ndcg10))
    return auc, mrr, ndcg5, ndcg10



def test(config: Config, corpus):
    if config.model == 'TANR':
        model = TANR(config)
    elif config.model == 'NAML':
        model = NAML(config)
    elif config.model == 'DKN':
        model = DKN(config)
    elif config.model == 'NRMS':
        model = NRMS(config)
    elif config.model == 'LSTUR':
        model = LSTUR(config)
    elif config.model == 'NPA':
        model = NPA(config)
    elif config.model == 'FIM':
        model = FIM(config)
    elif config.model == 'MINS':
        model = MINS(config)
    elif config.model == 'CENNEWSREC':
        model = CenNewsRec(config)
    elif config.model == 'IPNR':
        model = IPNR(config)
    elif config.model == 'CNE_SUE':
        model = CNE_SUE(config)

    assert os.path.exists(config.test_model_path), 'Test model does not exist : ' + config.test_model_path
    model.load_state_dict(torch.load(config.test_model_path, map_location=torch.device('cpu'))[config.model])
    model.cuda()
    test_res_dir = os.path.join(config.test_res_dir, config.test_model_path.replace('\\', '_').replace('/', '_'))
    if not os.path.exists(test_res_dir):
        os.mkdir(test_res_dir)
    print('test model path  : ' + config.test_model_path)
    print('test output file : ' + test_res_dir + '/' + config.model + '.txt')
    if config.dataset_name == 'MIND':
        if config.model == 'IPNR':
            auc, mrr, ndcg5, ndcg10 = compute_scores_IPNR(config, model, corpus, config.batch_size, 'test',
                                                          test_res_dir + '/' + config.model + '.txt', config.dataset)
        else:
            auc, mrr, ndcg5, ndcg10 = compute_scores(config, model, corpus, config.batch_size,
                                                     'test',
                                                     test_res_dir + '/' + config.model + '.txt', config.dataset)
    elif config.dataset_name == 'ebnerd':
        if config.model == 'IPNR':
            auc, mrr, ndcg5, ndcg10 = compute_scores_IPNR(config, model, corpus, config.batch_size, 'test',
                                                          test_res_dir + '/' + config.model + '.txt', config.dataset)
        else:
            auc, mrr, ndcg5, ndcg10 = compute_scores(config, model, corpus, config.batch_size,
                                                     'test',
                                                     test_res_dir + '/' + config.model + '.txt', config.dataset)

    if config.dataset != 'large':
        print('AUC : %.4f\nMRR : %.4f\nnDCG@5 : %.4f\nnDCG@10 : %.4f' % (auc, mrr, ndcg5, ndcg10))
        if config.mode == 'train':
            with open(config.result_dir + '/#' + str(config.run_index) + '-test', 'w') as result_f:
                result_f.write('#' + str(config.run_index) + '\t' + str(auc) + '\t' + str(mrr) + '\t' + str(ndcg5) + '\t' + str(ndcg10) + '\n')
        elif config.mode == 'test' and config.test_output_file != '':
            with open(config.test_output_file, 'w', encoding='utf-8') as f:
                f.write('#' + str(config.seed + 1) + '\t' + str(auc) + '\t' + str(mrr) + '\t' + str(ndcg5) + '\t' + str(ndcg10) + '\n')
    else:
        if config.mode == 'train':
            shutil.copy(test_res_dir + '/' + config.model + '.txt', 'cache/prediction/large/%s/#%d/prediction.txt' % (config.model, config.run_index))
            os.chdir('cache/prediction/large/%s/#%d' % (config.model, config.run_index))
            os.system('zip prediction.zip prediction.txt')
            os.chdir('../../../..')


if __name__ == '__main__':
    config = Config()
    wandb.login(anonymous="allow", key=config.wandb_key)  # Login to Weights & Biases
    run = wandb.init(
        project="NewsRecTorch-project",  # Specify your project
        config=config.attribute_dict,
        mode=config.wandb  # Set mode based on config
    )

    if config.dataset_name == 'MIND':
        corpus = MIND_Corpus(config)
    elif config.dataset_name == 'ebnerd':
        corpus = EBNeRD_Corpus(config)

    if config.mode == 'train':
        print("Start training at: ", datetime.now())
        train(config, corpus, wandb)
        print("Finish training at: ", datetime.now())
        config.test_model_path = config.best_model_dir + '/#' + str(config.run_index) + '/' + config.model
        print("Start testing at: ", datetime.now())
        test(config, corpus)
        print("Finish testing at: ", datetime.now())
    elif config.mode == 'dev':
        print("Start dev at: ", datetime.now())
        dev(config, corpus)
        print("Finish dev at: ", datetime.now())
    elif config.mode == 'test':
        print("Start testing at: ", datetime.now())
        test(config)
        print("Finish testing at: ", datetime.now())



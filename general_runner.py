import os
import gc
import shutil

from dataset_corpus_preprocessing.EBNeRD_corpus_main import EBNeRD_Corpus
from dataset_corpus_preprocessing.MIND_corpus_IPNR import MIND_Corpus_IPNR
from models.CNE_SUE import Model
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
from models.UNBERT import UNBERT
from models.modules.ipnr.trainer import TrainerIPNR
from util import get_run_index, compute_scores_IPNR
from datetime import datetime
import wandb
from config import Config
from dataset_corpus_preprocessing.MIND_corpus_main import MIND_Corpus
from dataset_corpus_preprocessing.MIND_dataset import MIND_Train_Dataset
from util import AvgMetric
from util import compute_scores
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from dataset_corpus_preprocessing.data_loader_unbert import MindDataset
from models.modules.unbert.eval import dev, test


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

def run_unbert(config: Config):
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
        if config.multi_gpu:
            model = nn.DataParallel(model, device_ids=config.device_id)
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


class Trainer:
    def __init__(self, model: nn.Module, config: Config, mind_corpus: MIND_Corpus, wandb, run_index: int):
        self.wandb = wandb
        self.config = config
        self.model = model
        self.epoch = config.epoch
        self.batch_size = config.batch_size
        self.max_history_num = config.max_history_num
        self.negative_sample_num = config.negative_sample_num
        self.loss = self.negative_log_softmax if config.click_predictor in ['dot_product', 'mlp', 'FIM'] else self.negative_log_sigmoid
        self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=config.lr, weight_decay=config.weight_decay)
        self._dataset = config.dataset
        self.mind_corpus = mind_corpus
        self.train_dataset = MIND_Train_Dataset(mind_corpus)
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
            self.loss = nn.DataParallel(self.loss)
        for e in tqdm(range(1, self.epoch + 1)):
            self.train_dataset.negative_sampling()
            train_dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.batch_size // 16, pin_memory=True)
            model.train()
            epoch_loss = 0
            for (user_ID, user_category, user_subCategory, user_title_text, user_title_mask, user_title_entity, user_content_text, user_content_mask, user_content_entity, user_history_mask, user_history_graph, user_history_category_mask, user_history_category_indices, \
                news_category, news_subCategory, news_title_text, news_title_mask, news_title_entity, news_content_text, news_content_mask, news_content_entity) in train_dataloader:
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
                else:
                    logits = model(user_ID, user_category, user_subCategory, user_title_text, user_title_mask,
                                   user_title_entity, user_content_text, user_content_mask, user_content_entity,
                                   user_history_mask, user_history_graph, user_history_category_mask,
                                   user_history_category_indices, \
                                   news_category, news_subCategory, news_title_text, news_title_mask, news_title_entity,
                                   news_content_text, news_content_mask,
                                   news_content_entity)  # [batch_size, 1 + negative_sample_num]
                    loss = self.loss(logits)
                if model.news_encoder.auxiliary_loss is not None:
                    news_auxiliary_loss = model.news_encoder.auxiliary_loss.mean()
                    loss += news_auxiliary_loss
                if model.user_encoder.auxiliary_loss is not None:
                    user_encoder_auxiliary_loss = model.user_encoder.auxiliary_loss.mean()
                    loss += user_encoder_auxiliary_loss
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
            auc, mrr, ndcg5, ndcg10 = compute_scores(self.config, model, self.mind_corpus, self.batch_size,
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


def train(config: Config, wandb):
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
    else:
        model = Model(config)
    model.initialize()
    run_index = get_run_index(config.result_dir)
    if config.dataset_name == 'MIND':
        if config.model == 'IPNR':
            trainer = TrainerIPNR(model, config, MIND_Corpus_IPNR, wandb, run_index)
        else:
            trainer = Trainer(model, config, MIND_Corpus, wandb, run_index)
    elif config.dataset_name == 'ebnerd':
        if config.model == 'IPNR':
            trainer = TrainerIPNR(model, config, EBNeRD_Corpus, wandb, run_index)
        else:
            trainer = Trainer(model, config, EBNeRD_Corpus, wandb, run_index)


    trainer.train()
    config.run_index = run_index


def dev(config: Config):
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
    else:
        model = Model(config)
    assert os.path.exists(config.dev_model_path), 'Dev model does not exist : ' + config.dev_model_path
    model.load_state_dict(torch.load(config.dev_model_path, map_location=torch.device('cpu'))[model.model_name])
    model.cuda()
    dev_res_dir = os.path.join(config.dev_res_dir, config.dev_model_path.replace('\\', '_').replace('/', '_'))
    if not os.path.exists(dev_res_dir):
        os.mkdir(dev_res_dir)
    if config.dataset_name == 'MIND':
        if config.model == 'IPNR':
            auc, mrr, ndcg5, ndcg10 = compute_scores_IPNR(config, model, MIND_Corpus_IPNR, config.batch_size, 'dev',
                                                          dev_res_dir + '/' + config.model + '.txt', config.dataset)
        else:
            auc, mrr, ndcg5, ndcg10 = compute_scores(config, model, MIND_Corpus, config.batch_size,
                                                     'dev',
                                                     dev_res_dir + '/' + config.model + '.txt', config.dataset)
    elif config.dataset_name == 'ebnerd':
        if config.model == 'IPNR':
            auc, mrr, ndcg5, ndcg10 = compute_scores_IPNR(config, model, EBNeRD_Corpus, config.batch_size, 'dev',
                                                          dev_res_dir + '/' + config.model + '.txt', config.dataset)
        else:
            auc, mrr, ndcg5, ndcg10 = compute_scores(config, model, EBNeRD_Corpus, config.batch_size,
                                                     'dev',
                                                     dev_res_dir + '/' + config.model + '.txt', config.dataset)



    print('Dev : ' + config.dev_model_path)
    print('AUC : %.4f\nMRR : %.4f\nnDCG@5 : %.4f\nnDCG@10 : %.4f' % (auc, mrr, ndcg5, ndcg10))
    return auc, mrr, ndcg5, ndcg10



def test(config: Config):
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
    else:
        model = Model(config)
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
            auc, mrr, ndcg5, ndcg10 = compute_scores_IPNR(config, model, MIND_Corpus, config.batch_size, 'test',
                                                          test_res_dir + '/' + config.model + '.txt', config.dataset)
        else:
            auc, mrr, ndcg5, ndcg10 = compute_scores(config, model, MIND_Corpus, config.batch_size,
                                                     'test',
                                                     test_res_dir + '/' + config.model + '.txt', config.dataset)
    elif config.dataset_name == 'ebnerd':
        if config.model == 'IPNR':
            auc, mrr, ndcg5, ndcg10 = compute_scores_IPNR(config, model, EBNeRD_Corpus, config.batch_size, 'test',
                                                          test_res_dir + '/' + config.model + '.txt', config.dataset)
        else:
            auc, mrr, ndcg5, ndcg10 = compute_scores(config, model, EBNeRD_Corpus, config.batch_size,
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

    if config.model == 'UNBERT':
        run_unbert(config)
    else:
        if config.mode == 'train':
            print("Start training at: ", datetime.now())
            train(config, wandb)
            print("Finish training at: ", datetime.now())
            config.test_model_path = config.best_model_dir + '/#' + str(config.run_index) + '/' + config.model
            print("Start testing at: ", datetime.now())
            test(config)
            print("Finish testing at: ", datetime.now())
        elif config.mode == 'dev':
            print("Start dev at: ", datetime.now())
            dev(config)
            print("Finish dev at: ", datetime.now())
        elif config.mode == 'test':
            print("Start testing at: ", datetime.now())
            test(config)
            print("Finish testing at: ", datetime.now())



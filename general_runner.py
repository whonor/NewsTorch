import csv
import os
import shutil

from dataset_corpus_preprocessing.EBNeRD_corpus_main import EBNeRD_Corpus, Ebnerd_Train_Dataset
from dataset_corpus_preprocessing.MIND_corpus_IPNR import MIND_Corpus_IPNR
from dataset_corpus_preprocessing.MIND_corpus_SentiRec import MIND_Corpus_SentiRec, MIND_Train_Dataset_SentiRec, MIND_DevTest_Dataset_SentiRec
from models.MMRec import MMRec
from models.CNE_SUE import CNE_SUE
from models.DKN import DKN
from models.FIM import FIM
from models.IPNR import IPNR
from models.LKPNR import LKPNR
from models.LSTUR import LSTUR
from models.MINS import MINS
from models.NAML import NAML
from models.NPA import NPA
from models.NRMS import NRMS
from models.SentiRec import SentiRec
from models.TANR import TANR
from models.CenNewsRec import CenNewsRec
from models.SentiDebias import SentiDebias
from models.CNRCL import CNRCL
from models.modules.cnrcl.trainer import TrainerCNRCL
from models.modules.ipnr.trainer import TrainerIPNR
from models.modules.mmrec.trainer import TrainerMMRec
from models.modules.senti_debias.trainer import TrainerSentiDebias
from models.modules.sentirec.trainer import TrainerSentiRec

from utils._evaluation import get_run_index, compute_scores_IPNR, compute_scores_mmrec, compute_complexity, compute_inference_time
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
from dataset_corpus_preprocessing.EBNeRD_corpus_main import Ebnerd_DevTest_Dataset
from dataset_corpus_preprocessing.MIND_corpus_IPNR import MIND_DevTest_Dataset_IPNR
from dataset_corpus_preprocessing.MIND_corpus_main import MIND_DevTest_Dataset


from base_trainer import Trainer


def negative_log_softmax(logits):
    loss = (-torch.log_softmax(logits, dim=1).select(dim=1, index=0)).mean()
    return loss

def negative_log_sigmoid(logits):
    positive_sigmoid = torch.clamp(torch.sigmoid(logits[:, 0]), min=1e-15, max=1)
    negative_sigmoid = torch.clamp(torch.sigmoid(-logits[:, 1:]), min=1e-15, max=1)
    loss = -(torch.log(positive_sigmoid).sum() + torch.log(negative_sigmoid).sum()) / logits.numel()
    return loss


def train(config: Config, corpus, wandb):
    model_classes = {
        'TANR': TANR,
        'NAML': NAML,
        'DKN': DKN,
        'NRMS': NRMS,
        'LSTUR': LSTUR,
        'NPA': NPA,
        'FIM': FIM,
        'MINS': MINS,
        'CENNEWSREC': CenNewsRec,
        'IPNR': IPNR,
        'CNE-SUE': CNE_SUE,
        'LKPNR': LKPNR,
        'SentiDebias': SentiDebias,
        'SentiRec': SentiRec,
        'MMRec': MMRec,
        'CNRCL': CNRCL
    }

    if config.model not in model_classes:
        raise ValueError(f"Unknown model: {config.model}")

    model = model_classes[config.model](config)
    model.initialize()

    run_index = get_run_index(config.result_dir)
    if config.dataset_name == 'MIND':
        if config.model == 'IPNR':
            trainer = TrainerIPNR(model, config, corpus, wandb, run_index)
            trainer.train()
        elif config.model == 'SentiRec':
            trainer = TrainerSentiRec(model, config, corpus, wandb, run_index)
            trainer.train()
        else:
            trainer = Trainer(model, config, corpus, wandb, run_index)
            trainer.train()
    elif config.dataset_name == 'ebnerd':
        if config.model == 'IPNR':
            trainer = TrainerIPNR(model, config, corpus, wandb, run_index)
            trainer.train()
        elif config.model == 'SentiDebias':
            trainer = TrainerSentiDebias(model, config, corpus, wandb, run_index)
            trainer.train()
        elif config.model == 'SentiRec':
            trainer = TrainerSentiRec(model, config, corpus, wandb, run_index)
            trainer.train()
        elif config.model == 'MMRec':
            trainer = TrainerMMRec(model, config, corpus, wandb, run_index)
            trainer.train()
        elif config.model == 'CNRCL':
            trainer = TrainerCNRCL(model, config, corpus, wandb, run_index)
            trainer.train()
        else:
            trainer = Trainer(model, config, corpus, wandb, run_index)
            trainer.train()
    config.run_index = run_index


def dev(config: Config, corpus):
    model_classes = {
        'TANR': TANR,
        'NAML': NAML,
        'DKN': DKN,
        'NRMS': NRMS,
        'LSTUR': LSTUR,
        'NPA': NPA,
        'FIM': FIM,
        'MINS': MINS,
        'CENNEWSREC': CenNewsRec,
        'IPNR': IPNR,
        'CNE-SUE': CNE_SUE,
        'LKPNR': LKPNR,
        'SentiDebias': SentiDebias,
        'SentiRec': SentiRec,
        'MMRec': MMRec,
        'CNRCL': CNRCL
    }

    if config.model not in model_classes:
        raise ValueError(f"Unknown model: {config.model}")

    model = model_classes[config.model](config)

    assert os.path.exists(config.dev_model_path), 'Dev model does not exist : ' + config.dev_model_path
    model.load_state_dict(torch.load(config.dev_model_path, map_location=torch.device('cpu'))[model.model_name])
    model.cuda()
    dev_res_dir = os.path.join(config.dev_res_dir, config.dev_model_path.replace('\\', '_').replace('/', '_'))
    if not os.path.exists(dev_res_dir):
        os.mkdir(dev_res_dir)
    if config.dataset_name == 'MIND':
        if config.model == 'IPNR':
            auc, mrr, ndcg5, ndcg10, mae, rmse, recall5, recall10, hit5, hit10, precision5, precision10 = compute_scores_IPNR(config, model, corpus, config.batch_size, 'dev',
                                                          dev_res_dir + '/' + config.model + '.txt', config.dataset_size)
        else:
            auc, mrr, ndcg5, ndcg10, mae, rmse, recall5, recall10, hit5, hit10, precision5, precision10 = compute_scores(config, model, corpus, config.batch_size,
                                                     'dev',
                                                     dev_res_dir + '/' + config.model + '.txt', config.dataset_size)
    elif config.dataset_name == 'ebnerd':
        auc, mrr, ndcg5, ndcg10, mae, rmse, recall5, recall10, hit5, hit10, precision5, precision10 = compute_scores(config, model, corpus, config.batch_size,
                                                 'dev',
                                                 dev_res_dir + '/' + config.model + '.txt', config.dataset_size)

    print('Dev : ' + config.dev_model_path)
    print('AUC : %.4f\nMRR : %.4f\nnDCG@5 : %.4f\nnDCG@10 : %.4f\nMAE : %.4f\nRMSE : %.4f\nrecall@5 : %.4f'
          '\nrecall@10 : %.4f\nhit@5 : %.4f\nhit@10 : %.4f\nprecision@5 : %.4f\nprecision@10 : %.4f' % (auc, mrr, ndcg5, ndcg10, mae, rmse, recall5, recall10, hit5, hit10, precision5, precision10))
    return auc, mrr, ndcg5, ndcg10, mae, rmse, recall5, recall10, hit5, hit10, precision5, precision10



def test(config: Config, corpus):
    model_classes = {
        'TANR': TANR,
        'NAML': NAML,
        'DKN': DKN,
        'NRMS': NRMS,
        'LSTUR': LSTUR,
        'NPA': NPA,
        'FIM': FIM,
        'MINS': MINS,
        'CENNEWSREC': CenNewsRec,
        'IPNR': IPNR,
        'CNE-SUE': CNE_SUE,
        'LKPNR': LKPNR,
        'SentiDebias': SentiDebias,
        'SentiRec': SentiRec,
        'MMRec': MMRec,
        'CNRCL': CNRCL
    }

    if config.model not in model_classes:
        raise ValueError(f"Unknown model: {config.model}")

    model = model_classes[config.model](config)

    assert os.path.exists(config.test_model_path), 'Test model does not exist : ' + config.test_model_path
    checkpoint = torch.load(config.test_model_path, map_location=torch.device('cpu'))
    model_state_dict = checkpoint[config.model]
    if list(model_state_dict.keys())[0].startswith('module.'):
        model_state_dict = {k[7:]: v for k, v in model_state_dict.items()}
    model.load_state_dict(model_state_dict)
    # model.load_state_dict(torch.load(config.test_model_path, map_location=torch.device('cpu'))[config.model])
    model.cuda()
    test_res_dir = os.path.join(config.test_res_dir, config.test_model_path.replace('\\', '_').replace('/', '_'))
    if not os.path.exists(test_res_dir):
        os.mkdir(test_res_dir)
    print('test model path  : ' + config.test_model_path)
    print('test output file : ' + test_res_dir + '/' + config.model + '.txt')
    if config.dataset_name == 'MIND':
        if config.model == 'IPNR':
            auc, mrr, ndcg5, ndcg10, mae, rmse, recall5, recall10, hit5, hit10, precision5, precision10 = compute_scores_IPNR(config, model, corpus, config.batch_size, 'test',
                                                          test_res_dir + '/' + config.model + '.txt', config.dataset_size)
        else:
            auc, mrr, ndcg5, ndcg10, mae, rmse, recall5, recall10, hit5, hit10, precision5, precision10 = compute_scores(config, model, corpus, config.batch_size,
                                                     'test',
                                                     test_res_dir + '/' + config.model + '.txt', config.dataset_size)
    elif config.dataset_name == 'ebnerd':
        if config.model == 'IPNR':
            auc, mrr, ndcg5, ndcg10, mae, rmse, recall5, recall10, hit5, hit10, precision5, precision10 = compute_scores_IPNR(config, model, corpus, config.batch_size, 'test',
                                                          test_res_dir + '/' + config.model + '.txt', config.dataset_size)
        elif config.model == 'MMRec':
            auc, mrr, ndcg5, ndcg10, mae, rmse, recall5, recall10, hit5, hit10, precision5, precision10 = compute_scores_mmrec(config, model, corpus, config.batch_size,
                                                                                                                               'test', test_res_dir + '/' + config.model + '.txt', config.dataset_size)
        else:
            auc, mrr, ndcg5, ndcg10, mae, rmse, recall5, recall10, hit5, hit10, precision5, precision10 = compute_scores(config, model, corpus, config.batch_size,
                                                     'test',
                                                     test_res_dir + '/' + config.model + '.txt', config.dataset_size)

    # Compute complexity and inference time
    if config.dataset_name == 'ebnerd':
        dataset = Ebnerd_DevTest_Dataset(corpus, 'test')
    elif config.dataset_name == 'MIND':
        if config.model == 'IPNR':
            dataset = MIND_DevTest_Dataset_IPNR(corpus, 'test')
        elif config.model == 'SentiRec':
            dataset = MIND_DevTest_Dataset_SentiRec(corpus, 'test')
        else:
            dataset = MIND_DevTest_Dataset(corpus, 'test')
    
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False, num_workers=0, pin_memory=True)
    try:
        data_batch = next(iter(dataloader))
        flops, params = compute_complexity(model, config, data_batch)
        inference_time = compute_inference_time(model, config, data_batch)
        print(f"FLOPs: {flops}, Params: {params}, Inference Time: {inference_time:.6f} s/batch")
    except Exception as e:
        print(f"Error computing complexity/inference time: {e}")
        flops, params, inference_time = 0, 0, 0

    if config.dataset_size != 'submission':
        print('AUC : %.4f\nMRR : %.4f\nnDCG@5 : %.4f\nnDCG@10 : %.4f\nMAE : %.4f\nRMSE : %.4f\nrecall@5 : %.4f'
          '\nrecall@10 : %.4f\nhit@5 : %.4f\nhit@10 : %.4f\nprecision@5 : %.4f\nprecision@10 : %.4f' % (auc, mrr, ndcg5, ndcg10, mae, rmse, recall5, recall10, hit5, hit10, precision5, precision10))
        if config.mode == 'train':
            metrics = {
                'AUC': auc,
                'MRR': mrr,
                'nDCG@5': ndcg5,
                'nDCG@10': ndcg10,
                'MAE': mae,
                'RMSE': rmse,
                'Recall@5': recall5,
                'Recall@10': recall10,
                'Hit@5': hit5,
                'Hit@10': hit10,
                'Precision@5': precision5,
                'Precision@10': precision10,
                'FLOPs': flops,
                'Params': params,
                'Inference Time': inference_time
            }
            csv_path = os.path.join(config.result_dir, 'metrics_results.csv')
            file_exists = os.path.isfile(csv_path)
            with open(csv_path, mode='a', newline='') as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=['Model'] + ['RunIndex'] + list(metrics.keys()))

                if not file_exists:
                    writer.writeheader()

                row = {'Model': config.model, 'RunIndex': config.run_index}
                row.update(metrics)
                writer.writerow(row)

        elif config.mode == 'test' and config.test_output_file != '':
            with open(config.test_output_file, 'w', encoding='utf-8') as f:
                f.write('#' + str(config.seed + 1) + '\t' + str(auc) + '\t' + str(mrr) + '\t' + str(ndcg5) + '\t' + str(ndcg10)
                        + '\t' + str(mae) + '\t' + str(rmse) + '\t' + str(recall5) + '\t' + str(recall10) + '\t' + str(hit5)
                        + '\t' + str(hit10) + '\t' + str(precision5) + '\t' + str(precision10) + '\n')
    else:
        shutil.copy(test_res_dir + '/' + config.model + '.txt', 'cache/prediction/large/%s/#%d/prediction.txt' % (config.model, config.run_index))
        os.chdir('cache/prediction/large/%s/#%d' % (config.model, config.run_index))
        os.system('zip prediction.zip prediction.txt')
        os.chdir('../../../..')


if __name__ == '__main__':

    global corpus

    config = Config()
    wandb.login(anonymous="allow", key=config.wandb_key)  # Login to Weights & Biases
    run = wandb.init(
        project="NewsTorch-project",  # Specify your project
        config=config.attribute_dict,
        mode=config.wandb  # Set mode based on config
    )

    if config.dataset_name == 'MIND':
        if config.model == 'IPNR':
            corpus = MIND_Corpus_IPNR(config)
        elif config.model == 'SentiRec':
            corpus = MIND_Corpus_SentiRec(config)
        else:
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
        test(config, corpus)
        print("Finish testing at: ", datetime.now())



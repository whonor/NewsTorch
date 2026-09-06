import csv
import hashlib
import json
import os
import shutil

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import nltk
nltk.download('stopwords')

from utils import _transformers_compat
from dataset_corpus_preprocessing.EBNeRD_corpus_main import EBNeRD_Corpus, Ebnerd_Train_Dataset, Ebnerd_DevTest_Dataset
from dataset_corpus_preprocessing.Adressa_corpus_main import (
    Adressa_Corpus,
    Adressa_DevTest_Dataset,
    Adressa_DevTest_Dataset_IPNR,
)
from dataset_corpus_preprocessing.EBNeRD_corpus_CPRS import EBNeRD_Corpus as EBNeRD_Corpus_CPRS
from dataset_corpus_preprocessing.EBNeRD_corpus_TCCM import EBNeRD_Corpus as EBNeRD_Corpus_TCCM
from dataset_corpus_preprocessing.EBNeRD_corpus_SEIN import EBNeRD_Corpus as EBNeRD_Corpus_SEIN
from dataset_corpus_preprocessing.MIND_corpus_IPNR import MIND_Corpus_IPNR
from dataset_corpus_preprocessing.MIND_corpus_SentiRec import MIND_Corpus_SentiRec, MIND_Train_Dataset_SentiRec, MIND_DevTest_Dataset_SentiRec
from dataset_corpus_preprocessing.Fake_MIND_corpus import Fake_MIND_Corpus, MIND_DevTest_Dataset as Fake_MIND_DevTest_Dataset
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
from models.PLM_NR import PLM_NR
from models.TCCM import TCCM
from models.SentiRec import SentiRec
from models.TANR import TANR
from models.CenNewsRec import CenNewsRec
from models.SentiDebias import SentiDebias
from models.CNRCL import CNRCL
from models.CPRS import CPRS
from models.SEIN import SEIN
from models.ONCE_DIRE_QWEN3_NAML import ONCE_DIRE_QWEN3_NAML
from models.PNR_LLM import PNR_LLM
from models.S2LENR import S2LENR
from models.Composable import Composable
from models.SOTAComposition import SOTAComposition
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


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _json_default(value):
    """Convert NumPy/PyTorch scalar metrics to JSON-native values."""
    item = getattr(value, 'item', None)
    if callable(item):
        scalar = item()
        if isinstance(scalar, (bool, int, float, str)) or scalar is None:
            return scalar
    raise TypeError('Object of type %s is not JSON serializable' % type(value).__name__)


def write_metrics_output(config, metrics):
    """Write one framework-neutral result record for comparison experiments."""
    if not config.metrics_output:
        return
    output_path = os.path.abspath(config.metrics_output)
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    payload = {
        'schema_version': 1,
        'framework': 'newstorch',
        'model': config.model,
        'seed': config.seed,
        'evaluation_split': 'test',
        'run_index': getattr(config, 'run_index', None),
        'best_epoch': getattr(config, 'best_dev_epoch', None),
        'metrics': metrics,
        'runtime_settings': {
            'dataset_root': os.path.abspath(config.DATASET_ROOT),
            'word_embedding_dim': config.word_embedding_dim,
            'max_title_length': config.max_title_length,
            'max_abstract_length': config.max_abstract_length,
            'max_history_num': config.max_history_num,
            'negative_sample_num': config.negative_sample_num,
            'optimizer': config.optimizer,
            'learning_rate': config.lr,
            'weight_decay': config.weight_decay,
            'dropout_rate': config.dropout_rate,
            'loss': 'cross_entropy' if config.click_predictor in ('dot_product', 'mlp', 'FIM') else 'binary_cross_entropy',
            'max_epochs': config.epoch,
            'early_stopping_patience': config.early_stopping_epoch,
            'early_stopping_min_delta': config.early_stopping_min_delta,
            'stopping_metric': config.dev_criterion,
        },
    }
    if config.model in {'COMPOSABLE', 'SOTA-COMPOSABLE'}:
        payload['composition'] = {
            'id': config.composition_id,
            'news_encoder': config.news_encoder_type,
            'user_encoder': config.user_encoder_type,
        }
        if config.model == 'SOTA-COMPOSABLE':
            payload['composition'].update({
                'enhancement': config.enhancement_type,
                'training': config.training_strategy,
                'special_modules': getattr(config, 'special_modules', []),
            })
    if config.protocol_manifest:
        manifest_path = os.path.abspath(config.protocol_manifest)
        if not os.path.isfile(manifest_path):
            raise FileNotFoundError('Protocol manifest does not exist: ' + manifest_path)
        with open(manifest_path, 'r', encoding='utf-8') as manifest_file:
            manifest = json.load(manifest_file)
        protocol = manifest['protocol']
        expected_training = protocol['training']
        actual_training = {
            'batch_size': config.batch_size,
            'max_epochs': config.epoch,
            'negative_sample_num': config.negative_sample_num,
            'optimizer': config.optimizer,
            'learning_rate': config.lr,
            'weight_decay': config.weight_decay,
            'dropout_rate': config.dropout_rate,
            'loss': 'cross_entropy' if config.click_predictor in ('dot_product', 'mlp', 'FIM') else 'binary_cross_entropy',
            'gradient_clip_norm': config.gradient_clip_norm,
            'early_stopping_patience': config.early_stopping_epoch,
            'early_stopping_min_delta': config.early_stopping_min_delta,
            'stopping_metric': config.dev_criterion,
            'deterministic': config.deterministic,
            'num_workers': config.num_workers,
        }
        if actual_training != expected_training:
            raise ValueError('Runtime training settings do not match the protocol')
        actual_sequence_lengths = {
            'title': config.max_title_length,
            'abstract': config.max_abstract_length,
            'history': config.max_history_num,
        }
        if actual_sequence_lengths != protocol['sequence_lengths']:
            raise ValueError('Runtime sequence lengths do not match the protocol')
        model_overrides = protocol.get('model_overrides', {}).get(config.model, {})
        for name, expected_value in model_overrides.items():
            if getattr(config, name, None) != expected_value:
                raise ValueError('Runtime model setting %s does not match the protocol' % name)
        expected_dataset_root = manifest['aligned_datasets'][str(config.seed)]['root']
        if os.path.realpath(config.DATASET_ROOT) != os.path.realpath(expected_dataset_root):
            raise ValueError('Runtime dataset root does not match the seed-specific protocol data')
        if os.path.realpath(config.word_embedding_path) != os.path.realpath(manifest['embeddings']['path']):
            raise ValueError('Runtime word embeddings do not match the protocol artifact')
        if os.path.realpath(config.vocabulary_path) != os.path.realpath(manifest['vocabulary']['path']):
            raise ValueError('Runtime vocabulary does not match the protocol artifact')
        payload['settings'] = {
            'training': expected_training,
            'sequence_lengths': protocol['sequence_lengths'],
            'model_overrides': model_overrides,
            'word_embeddings_sha256': manifest['embeddings']['logical_sha256'],
        }
        payload['protocol_manifest'] = manifest_path
        payload['protocol_manifest_sha256'] = _sha256_file(manifest_path)
    temporary_path = output_path + '.tmp'
    with open(temporary_path, 'w', encoding='utf-8') as output_file:
        json.dump(payload, output_file, indent=2, sort_keys=True, default=_json_default)
        output_file.write('\n')
    os.replace(temporary_path, output_path)


def get_model_classes():
    model_classes = {
        'TANR': TANR,
        'NAML': NAML,
        'DKN': DKN,
        'NRMS': NRMS,
        'PLM-NR': PLM_NR,
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
        'CNRCL': CNRCL,
        'CPRS': CPRS,
        'TCCM': TCCM,
        'SEIN': SEIN,
        'ONCE': ONCE_DIRE_QWEN3_NAML,
        'S2LENR': S2LENR,
        'PNR-LLM': PNR_LLM,
        'COMPOSABLE': Composable,
        'SOTA-COMPOSABLE': SOTAComposition,
    }
    return model_classes


def negative_log_softmax(logits):
    loss = (-torch.log_softmax(logits, dim=1).select(dim=1, index=0)).mean()
    return loss

def negative_log_sigmoid(logits):
    positive_sigmoid = torch.clamp(torch.sigmoid(logits[:, 0]), min=1e-15, max=1)
    negative_sigmoid = torch.clamp(torch.sigmoid(-logits[:, 1:]), min=1e-15, max=1)
    loss = -(torch.log(positive_sigmoid).sum() + torch.log(negative_sigmoid).sum()) / logits.numel()
    return loss


def train(config: Config, corpus, wandb):
    model_classes = get_model_classes()

    if config.model not in model_classes:
        raise ValueError(f"Unknown model: {config.model}")

    model = model_classes[config.model](config)
    model.initialize()
    if hasattr(model, 'set_corpus'):
        model.set_corpus(corpus)

    run_index = get_run_index(config.result_dir)
    if config.dataset_name in ['MIND', 'gossipcop']:
        if config.model == 'IPNR':
            trainer = TrainerIPNR(model, config, corpus, wandb, run_index)
            trainer.train()
        elif config.model == 'SentiRec':
            trainer = TrainerSentiRec(model, config, corpus, wandb, run_index)
            trainer.train()
        else:
            trainer = Trainer(model, config, corpus, wandb, run_index)
            trainer.train()
    elif config.dataset_name in {'ebnerd', 'Adressa'}:
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
    config.best_dev_epoch = trainer.best_dev_epoch
    return trainer.best_dev_avg


def dev(config: Config, corpus):
    model_classes = get_model_classes()

    if config.model not in model_classes:
        raise ValueError(f"Unknown model: {config.model}")

    model = model_classes[config.model](config)
    if hasattr(model, 'set_corpus'):
        model.set_corpus(corpus)

    assert os.path.exists(config.dev_model_path), 'Dev model does not exist : ' + config.dev_model_path
    model.load_state_dict(torch.load(config.dev_model_path, map_location=torch.device('cpu'))[model.model_name])
    model.cuda()
    dev_res_dir = os.path.join(config.dev_res_dir, config.dev_model_path.replace('\\', '_').replace('/', '_'))
    if not os.path.exists(dev_res_dir):
        os.mkdir(dev_res_dir)
    if config.dataset_name in ['MIND', 'gossipcop']:
        if config.model == 'IPNR':
            auc, mrr, ndcg5, ndcg10, mae, rmse, recall5, recall10, hit5, hit10, precision5, precision10 = compute_scores_IPNR(config, model, corpus, config.batch_size, 'dev',
                                                          dev_res_dir + '/' + config.model + '.txt', config.dataset_size)
        else:
            auc, mrr, ndcg5, ndcg10, mae, rmse, recall5, recall10, hit5, hit10, precision5, precision10 = compute_scores(config, model, corpus, config.batch_size,
                                                     'dev',
                                                     dev_res_dir + '/' + config.model + '.txt', config.dataset_size)
    elif config.dataset_name in {'ebnerd', 'Adressa'}:
        if config.model == 'IPNR':
            auc, mrr, ndcg5, ndcg10, mae, rmse, recall5, recall10, hit5, hit10, precision5, precision10 = compute_scores_IPNR(config, model, corpus, config.batch_size, 'dev',
                                                          dev_res_dir + '/' + config.model + '.txt', config.dataset_size)
        elif config.model == 'MMRec':
            auc, mrr, ndcg5, ndcg10, mae, rmse, recall5, recall10, hit5, hit10, precision5, precision10 = compute_scores_mmrec(config, model, corpus, config.batch_size,
                                                     'dev',
                                                     dev_res_dir + '/' + config.model + '.txt', config.dataset_size)
        else:
            auc, mrr, ndcg5, ndcg10, mae, rmse, recall5, recall10, hit5, hit10, precision5, precision10 = compute_scores(config, model, corpus, config.batch_size,
                                                     'dev',
                                                     dev_res_dir + '/' + config.model + '.txt', config.dataset_size)

    print('Dev : ' + config.dev_model_path)
    print('AUC : %.4f\nMRR : %.4f\nnDCG@5 : %.4f\nnDCG@10 : %.4f\nMAE : %.4f\nRMSE : %.4f\nRecall@5 : %.4f'
          '\nRecall@10 : %.4f\nHit Rate@5 : %.4f\nHit Rate@10 : %.4f\nPrecision@5 : %.4f\nPrecision@10 : %.4f' % (auc, mrr, ndcg5, ndcg10, mae, rmse, recall5, recall10, hit5, hit10, precision5, precision10))
    return auc, mrr, ndcg5, ndcg10, mae, rmse, recall5, recall10, hit5, hit10, precision5, precision10



def test(config: Config, corpus):
    model_classes = get_model_classes()

    if config.model not in model_classes:
        raise ValueError(f"Unknown model: {config.model}")

    model = model_classes[config.model](config)
    if hasattr(model, 'set_corpus'):
        model.set_corpus(corpus)

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
    if config.dataset_name in ['MIND', 'gossipcop']:
        if config.model == 'IPNR':
            auc, mrr, ndcg5, ndcg10, mae, rmse, recall5, recall10, hit5, hit10, precision5, precision10 = compute_scores_IPNR(config, model, corpus, config.batch_size, 'test',
                                                          test_res_dir + '/' + config.model + '.txt', config.dataset_size)
        else:
            auc, mrr, ndcg5, ndcg10, mae, rmse, recall5, recall10, hit5, hit10, precision5, precision10 = compute_scores(config, model, corpus, config.batch_size,
                                                     'test',
                                                     test_res_dir + '/' + config.model + '.txt', config.dataset_size)
    elif config.dataset_name in {'ebnerd', 'Adressa'}:
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
        if config.model == 'SEIN':
            from dataset_corpus_preprocessing.EBNeRD_corpus_SEIN import Ebnerd_DevTest_Dataset as Ebnerd_DevTest_Dataset_SEIN
            dataset = Ebnerd_DevTest_Dataset_SEIN(corpus, 'test')
        else:
            dataset = Ebnerd_DevTest_Dataset(corpus, 'test')
    elif config.dataset_name == 'Adressa':
        if config.model == 'IPNR':
            dataset = Adressa_DevTest_Dataset_IPNR(corpus, 'test')
        else:
            dataset = Adressa_DevTest_Dataset(corpus, 'test')
    elif config.dataset_name == 'MIND':
        if config.model == 'IPNR':
            dataset = MIND_DevTest_Dataset_IPNR(corpus, 'test')
        elif config.model == 'SentiRec':
            dataset = MIND_DevTest_Dataset_SentiRec(corpus, 'test')
        else:
            dataset = MIND_DevTest_Dataset(corpus, 'test')
    elif config.dataset_name == 'gossipcop':
        dataset = Fake_MIND_DevTest_Dataset(corpus, 'test')
    
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
        metrics = {
            'auc': auc,
            'mrr': mrr,
            'ndcg5': ndcg5,
            'ndcg10': ndcg10,
            'mae': mae,
            'rmse': rmse,
            'recall5': recall5,
            'recall10': recall10,
            'hit5': hit5,
            'hit10': hit10,
            'precision5': precision5,
            'precision10': precision10,
            'flops': flops,
            'params': params,
            'inference_time': inference_time,
        }
        print('AUC : %.4f\nMRR : %.4f\nnDCG@5 : %.4f\nnDCG@10 : %.4f\nMAE : %.4f\nRMSE : %.4f\nRecall@5 : %.4f'
          '\nRecall@10 : %.4f\nHit Rate@5 : %.4f\nHit Rate@10 : %.4f\nPrecision@5 : %.4f\nPrecision@10 : %.4f' % (auc, mrr, ndcg5, ndcg10, mae, rmse, recall5, recall10, hit5, hit10, precision5, precision10))
        if config.mode == 'train':
            csv_metrics = {
                'AUC': auc,
                'MRR': mrr,
                'nDCG@5': ndcg5,
                'nDCG@10': ndcg10,
                'MAE': mae,
                'RMSE': rmse,
                'Recall@5': recall5,
                'Recall@10': recall10,
                'Hit Rate@5': hit5,
                'Hit Rate@10': hit10,
                'Precision@5': precision5,
                'Precision@10': precision10,
                'FLOPs': flops,
                'Params': params,
                'Inference Time': inference_time
            }
            csv_path = os.path.join(config.result_dir, 'metrics_results.csv')
            file_exists = os.path.isfile(csv_path)
            with open(csv_path, mode='a', newline='') as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=['Model'] + ['RunIndex'] + list(csv_metrics.keys()))

                if not file_exists:
                    writer.writeheader()

                row = {'Model': config.model, 'RunIndex': config.run_index}
                row.update(csv_metrics)
                writer.writerow(row)

        elif config.mode == 'test' and config.test_output_file != '':
            with open(config.test_output_file, 'w', encoding='utf-8') as f:
                f.write('#' + str(config.seed + 1) + '\t' + str(auc) + '\t' + str(mrr) + '\t' + str(ndcg5) + '\t' + str(ndcg10)
                        + '\t' + str(mae) + '\t' + str(rmse) + '\t' + str(recall5) + '\t' + str(recall10) + '\t' + str(hit5)
                        + '\t' + str(hit10) + '\t' + str(precision5) + '\t' + str(precision10) + '\n')
    else:
        metrics = {}
        shutil.copy(test_res_dir + '/' + config.model + '.txt', 'cache/prediction/large/%s/#%d/prediction.txt' % (config.model, config.run_index))
        os.chdir('cache/prediction/large/%s/#%d' % (config.model, config.run_index))
        os.system('zip prediction.zip prediction.txt')
        os.chdir('../../../..')
    return metrics


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
    elif config.dataset_name == 'gossipcop':
        corpus = Fake_MIND_Corpus(config)

    elif config.dataset_name == 'ebnerd':
        if config.model in ['CPRS', 'DREAM']:
            corpus = EBNeRD_Corpus_CPRS(config)
        elif config.model == 'TCCM':
            corpus = EBNeRD_Corpus_TCCM(config)
        elif config.model == 'SEIN':
            corpus = EBNeRD_Corpus_SEIN(config)
        else:
            corpus = EBNeRD_Corpus(config)
    elif config.dataset_name == 'Adressa':
        corpus = Adressa_Corpus(config)

    if config.mode == 'train':
        print("Start training at: ", datetime.now())
        train(config, corpus, wandb)
        print("Finish training at: ", datetime.now())
        config.test_model_path = config.best_model_dir + '/#' + str(config.run_index) + '/' + config.model
        print("Start testing at: ", datetime.now())
        metrics = test(config, corpus)
        write_metrics_output(config, metrics)
        print("Finish testing at: ", datetime.now())
    elif config.mode == 'dev':
        print("Start dev at: ", datetime.now())
        dev(config, corpus)
        print("Finish dev at: ", datetime.now())
    elif config.mode == 'test':
        print("Start testing at: ", datetime.now())
        metrics = test(config, corpus)
        write_metrics_output(config, metrics)
        print("Finish testing at: ", datetime.now())

import os
import argparse
import time
import csv

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import pandas as pd
import torch
import random
import numpy as np
import json
from dataset_download_prepare.MIND_dataset_prepare import prepare_MIND_200k, prepare_MIND_large, prepare_MIND_small

class Config:
    '''
    """Configuration class for the news recommendation models.
    This class contains all the hyperparameters and settings for training and evaluating
    news recommendation models. It initializes the configuration based on the specified model
    and loads additional parameters from a JSON file if provided.
    Attributes:
        model (str): The name of the model to be used.
        multi_gpu (bool): Whether to use multiple GPUs for training.
        wandb (str): Whether to use Weights & Biases for experiment tracking.
        mode (str): The mode of operation, e.g., 'train', 'dev', 'test'.
        dev_model_path (str): Path to the model for development.
        test_model_path (str): Path to the model for testing.
        test_output_file (str): Output file for test results.
        device_id (int or list): The ID(s) of the GPU(s) to be used.
        seed (int): Random seed for reproducibility.
        config_file (str): Path to the configuration file.
        root (str): Root directory for the dataset.
        data_path (str): Path to the dataset.
        dataset (str): Name of the dataset to be used.
        tokenizer (str): Tokenizer to be used for text processing.
        word_threshold (int): Minimum frequency of words to be included in the vocabulary.
        max_title_length (int): Maximum length of the news title.
        max_abstract_length (int): Maximum length of the news abstract.
        negative_sample_num (int): Number of negative samples for training.
        max_history_num (int): Maximum number of historical news items to consider.
        candidate_news_num (int): Number of candidate news items for recommendation.
        epoch (int): Number of training epochs.
        batch_size (int): Batch size for training.
        lr (float): Learning rate for the optimizer.
        weight_decay (float): Weight decay for regularization.
        gradient_clip_norm (float): Gradient clipping norm.
        world_size (int): Number of processes for distributed training.
        dev_criterion (str): Criterion for development evaluation.
        early_stopping_epoch (int): Number of epochs for early stopping.
        category_embedding_dim (int): Dimension of category embeddings.
        subCategory_embedding_dim (int): Dimension of sub-category embeddings.
        entity_embedding_dim (int): Dimension of entity embeddings.
        context_embedding_dim (int): Dimension of context embeddings.
        dropout_rate (float): Dropout rate for regularization.
        no_self_connection (bool): Whether to disable self-connections in the graph.
        no_adjacent_normalization (bool): Whether to disable adjacent normalization in the graph.
        Alpha (float): Hyperparameter for the model.
        gcn_normalization_type (str): Type of normalization for GCN layers.
        click_predictor (str): Type of click predictor to be used.
    """

    '''

    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--model', type=str, default='SentiRec', help='Model name: NRMS, LSTUR, TANR, DKN, NAML, NPA, FIM, MINS, CENNEWSREC, IPNR, CNE-SUE, LKPNR, SentiDebias, SentiRec, MMRec, CNRCL, CPRS, DREAM, ONCE')
        parser.add_argument('--batch_size', type=int, default='64', help='Batch size for training')
        parser.add_argument('--seed', type=int, default=0, help='Seed')
        parser.add_argument('--epoch', type=int, default=10, help='Epoch for training')
        parser.add_argument('--mode', type=str, default='train', help='Mode')
        parser.add_argument('--DATASET_ROOT', type=str, default='ebnerd_demo', help='Default dataset name, can be ebnerd_demo, ebnerd_small, ebnerd_large, MIND-small, or MIND-large; gossipcop')
        parser.add_argument('--dataset_name', type=str, default='ebnerd', help='Name of the dataset to be used, MIND, ebnerd, gossipcop')
        parser.add_argument('--dataset_size', type=str, default='demo', help='Dataset variant, can be small, large, or demo, if submit the predictions submission')
        parser.add_argument('--images_path', type=str, default='downloaded_images_ebnerd_small',
                            help='downloaded_images for demo or small')
        parser.add_argument('--word_embedding_dim', type=int, default=300, help='ebnerd: 1024; glove: 300')
        parser.add_argument('--clickbait_score_path', type=str, default='',
                            help='Optional JSON, PKL, CSV, or Parquet file with per-news clickbait scores for TCE')

        parser.add_argument('--dev_model_path', type=str,
                            default='/home/wanro238/NewsRecTorch/cache/best_models/small/NRMS/#1/NRMS',
                            help='The path of the best model')
        parser.add_argument('--test_model_path', type=str,
                            default='/home/wanro238/NewsRecTorch/cache/best_models/small/NRMS/#1/NRMS',
                            help='The path of the best model')
        args, _ = parser.parse_known_args()
        self.model = args.model

        self.wandb = 'offline'  # Whether to use Weights & Biases for experiment tracking
        self.wandb_key = ''  # Key for Weights & Biases, if needed
        self.mode = args.mode
        self.dev_model_path = args.dev_model_path
        self.test_model_path = args.test_model_path
        self.test_output_file = ''
        self.seed = args.seed
        self.config_file = ''
        self.downloaded_images_file = args.images_path

        self.root = "."
        self.data_path = "cache/"
        self.DATASET_ROOT = args.DATASET_ROOT
        self.dataset_name = args.dataset_name
        self.dataset_size = args.dataset_size
        self.tokenizer = 'MIND'
        self.word_threshold = 3
        self.max_title_length = 32
        self.max_abstract_length = 128
        self.negative_sample_num = 4
        self.max_history_num = 50
        self.candidate_news_num = 5
        self.epoch = args.epoch

        self.batch_size = args.batch_size
        self.lr = 1e-4
        self.weight_decay = 0
        self.gradient_clip_norm = 4
        self.world_size = 1
        self.dev_criterion = 'avg'
        self.early_stopping_epoch = 5

        self.word_embedding_dim = args.word_embedding_dim
        self.clickbait_score_path = args.clickbait_score_path
        self.category_embedding_dim = 50
        self.subCategory_embedding_dim = 50
        self.entity_embedding_dim = 100
        self.context_embedding_dim = 100
        self.dropout_rate = 0.2
        self.no_self_connection = False
        self.no_adjacent_normalization = False
        self.Alpha = 0.1
        self.gcn_normalization_type = 'symmetric'
        if self.model == 'FIM':
            self.click_predictor = 'FIM'
        else:
            self.click_predictor = 'dot_product'

        # self.train_root = self.root + '/%s_%s/train' % (self.dataset_name, self.dataset)
        # self.dev_root = self.root + '/%s_%s/dev' % (self.dataset_name, self.dataset)
        # self.test_root = self.root + '/%s_%s/test' % (self.dataset_name, self.dataset)
        self.train_root = self.root + '/%s/train' % self.DATASET_ROOT
        self.dev_root = self.root + '/%s/dev' % self.DATASET_ROOT
        self.test_root = self.root + '/%s/test' % self.DATASET_ROOT

        self.seed = self.seed if self.seed >= 0 else (int)(time.time())

        import yaml

        yaml_path = f'config/{args.model.lower()}.yaml'
        if os.path.exists(yaml_path):
            with open(yaml_path, 'r') as f:
                model_params = yaml.safe_load(f)
            for k, v in model_params.items():
                setattr(self, k, v)
        else:
            print(f"Warning: Model config file {yaml_path} not found, using default parameters")

        self.image_embedding_dim = 2048


        self.attribute_dict = self.__dict__.copy()
        if self.config_file != '':
            if os.path.exists(self.config_file):
                print('Get experiment settings from the config file : ' + self.config_file)
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    configs = yaml.safe_load(f)
                    for attribute in self.attribute_dict:
                        if attribute in configs:
                            setattr(self, attribute, configs[attribute])
                            self.attribute_dict[attribute] = configs[attribute]
            else:
                raise Exception('config file does not exist : ' + self.config_file)
        assert not (
                    self.no_self_connection and not self.no_adjacent_normalization), 'Adjacent normalization of graph only can be set in case of self-connection'

        print('*' * 32 + ' Experiment setting ' + '*' * 32)
        for attribute, value in self.__dict__.items():
            print(f"{attribute} : {value}")
        print('*' * 32 + ' Experiment setting ' + '*' * 32)
        assert self.batch_size % self.world_size == 0, 'For multi-gpu training, batch size must be divisible by world size'
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '1024'
        self.preliminary_setup()
        self.set_cuda()


    def set_cuda(self):
        gpu_available = torch.cuda.is_available()
        assert gpu_available, 'GPU is not available'
        #torch.cuda.set_device(self.device_id)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.backends.cudnn.benchmark = True # For faster training
        torch.backends.cudnn.deterministic = False # For reproducibility (https://pytorch.org/docs/stable/notes/randomness.html)


    def preliminary_setup(self):
        dataset_files = [
            self.train_root + '/news.tsv', self.train_root + '/behaviors.tsv', self.train_root + '/entity_embedding.vec', self.train_root + '/context_embedding.vec',
            self.dev_root + '/news.tsv', self.dev_root + '/behaviors.tsv', self.dev_root + '/entity_embedding.vec', self.dev_root + '/context_embedding.vec',
            self.test_root + '/news.tsv', self.test_root + '/behaviors.tsv', self.test_root + '/entity_embedding.vec', self.test_root + '/context_embedding.vec'
        ]
        # if not all(list(map(os.path.exists, dataset_files))):
        #     # exec('prepare_MIND_%s()' % self.dataset)
        #     print("Please prepare the dataset first!!!")

        dataset_name = self.dataset_name
        model_name = self.model
        data_path = self.data_path
        mkdirs = lambda x: os.makedirs(x) if not os.path.exists(x) else None
        self.model_dir = data_path + 'models/' + self.DATASET_ROOT + '/' + model_name
        self.dev_res_dir = data_path + 'dev/res/' + self.DATASET_ROOT + '/' + model_name
        self.result_dir = data_path + 'results/' + self.DATASET_ROOT + '/' + model_name
        self.best_model_dir = data_path + 'best_models/' + self.DATASET_ROOT + '/' + model_name
        self.test_res_dir = data_path + 'test/res/' + self.DATASET_ROOT + '/' + model_name
        mkdirs(self.model_dir)
        mkdirs(data_path + 'dev/ref')
        mkdirs(self.dev_res_dir)
        mkdirs(data_path + 'test/ref')
        mkdirs(self.result_dir)
        mkdirs(self.best_model_dir)
        mkdirs(self.test_res_dir)
        if model_name == 'IPNR':
            mkdirs("cache/IPNR/")
        mkdirs("cache/%s/" % self.dataset_name)

        dev_truth_path = os.path.join(data_path, f'dev/ref/truth-{self.DATASET_ROOT}.txt')
        test_truth_path = os.path.join(data_path, f'test/ref/truth-{self.DATASET_ROOT}.txt')

        if dataset_name == 'MIND':
            if not os.path.exists(dev_truth_path):
                with open(os.path.join(self.dev_root, 'behaviors.tsv'), 'r', encoding='utf-8') as dev_f:
                    with open(dev_truth_path, 'w', encoding='utf-8') as truth_f:
                        for dev_ID, line in enumerate(dev_f):
                            impression_ID, user_ID, time, history, impressions = line.split('\t')
                            labels = [int(impression[-1]) for impression in impressions.strip().split(' ')]
                            truth_f.write(
                                ('' if dev_ID == 0 else '\n') + str(dev_ID + 1) + ' ' + str(labels).replace(' ', ''))
            if self.dataset_size != 'submission':
                if not os.path.exists(test_truth_path):
                    with open(os.path.join(self.test_root, 'behaviors.tsv'), 'r', encoding='utf-8') as test_f:
                        with open(test_truth_path, 'w', encoding='utf-8') as truth_f:
                            for test_ID, line in enumerate(test_f):
                                impression_ID, user_ID, time, history, impressions = line.split('\t')
                                labels = [int(impression[-1]) for impression in impressions.strip().split(' ')]
                                truth_f.write(
                                    ('' if test_ID == 0 else '\n') + str(test_ID + 1) + ' ' + str(labels).replace(' ',
                                                                                                                  ''))
            else:
                self.prediction_dir = 'prediction/large/' + model_name
                mkdirs(self.prediction_dir)

        elif dataset_name == 'ebnerd':
            def write_ebnerd_truth(behaviors_path, truth_path):
                df = pd.read_parquet(behaviors_path)
                with open(truth_path, 'w', encoding='utf-8') as truth_f:
                    for row_ID, row in df.iterrows():
                        labels = row['labels']
                        if hasattr(labels, 'tolist'):
                            labels = labels.tolist()
                        label_str = str(labels).replace(' ', '')
                        truth_f.write(('' if row_ID == 0 else '\n') + str(row_ID + 1) + ' ' + label_str)

            write_ebnerd_truth(os.path.join(self.dev_root, 'behaviors.parquet'), dev_truth_path)
            write_ebnerd_truth(os.path.join(self.test_root, 'behaviors.parquet'), test_truth_path)

        elif dataset_name == 'gossipcop':
            split_files = {
                dev_truth_path: os.path.join(self.root, self.DATASET_ROOT, 'val.csv'),
                test_truth_path: os.path.join(self.root, self.DATASET_ROOT, 'test.csv'),
            }
            for truth_path, csv_path in split_files.items():
                if not os.path.exists(truth_path):
                    with open(csv_path, newline='', encoding='utf-8') as csv_f:
                        with open(truth_path, 'w', encoding='utf-8') as truth_f:
                            for row_ID, row in enumerate(csv.DictReader(csv_f)):
                                labels = [int(label) for label in row['clicked'].split()]
                                truth_f.write(('' if row_ID == 0 else '\n') + str(row_ID + 1) + ' ' + str(labels).replace(' ', ''))

import os
import argparse
import time
import torch
import random
import numpy as np
import json
from Dataset_prepare.MIND_dataset_prepare import prepare_MIND_200k, prepare_MIND_large, prepare_MIND_small

class Config:
    '''
    """Configuration class for the news recommendation models.
    This class contains all the hyperparameters and settings for training and evaluating
    news recommendation models. It initializes the configuration based on the specified model
    and loads additional parameters from a JSON file if provided.
    Attributes:
        model (str): The name of the model to be used, can be 'LSTUR', 'NRMS', 'NPA', 'TANR', 'FIM', 'DKN', 'NAML',
        'CNE-SUE', 'MINS', 'MINER', 'UNBERT', 'CenNewsRec', 'MANNeR'.
        mode (str): The mode of operation, can be 'train', 'dev', or 'test'.
        dev_model_path (str): Path to the development model.
        test_model_path (str): Path to the test model.
        test_output_file (str): Output file for test results.
        device_id (int): ID of the GPU device to use.
        seed (int): Random seed for reproducibility.
        config_file (str): Path to a JSON configuration file for additional settings.
        root (str): Root directory of the project.
        dataset (str): Dataset type, can be 'small', '200k', or 'large'.
        tokenizer (str): Tokenizer type, can be 'MIND' or 'NLTK'.
        word_threshold (int): Minimum frequency threshold for words in the vocabulary.
        max_title_length (int): Maximum length of news titles.
        max_abstract_length (int): Maximum length of news abstracts.
        negative_sample_num (int): Number of negative samples per positive sample.
        max_history_num (int): Maximum number of history news items per user.
        epoch (int): Number of training epochs.
        batch_size (int): Batch size for training and evaluation.
        lr (float): Learning rate for the optimizer.
        weight_decay (float): Weight decay for the optimizer.
        gradient_clip_norm (float): Gradient clipping norm, non-positive value means no clipping.
        world_size (int): Number of processes in multi-GPU training.
        dev_criterion (str): Criterion for selecting the best model during development, can be 'avg', 'auc', 'mrr', etc.
        early_stopping_epoch (int): Number of epochs without improvement before stopping training early.
        word_embedding_dim (int): Dimension of word embeddings.
        entity_embedding_dim (int): Dimension of entity embeddings.
        context_embedding_dim (int): Dimension of context embeddings.
        cnn_method (str): Method for CNN, can be 'naive', 'group3', 'group4', or 'group5'.
        cnn_kernel_num (int): Number of CNN kernels.
        cnn_window_size (int): Window size for CNN kernels.
        attention_dim (int): Dimension of attention mechanism.
        head_num (int): Number of heads in multi-head attention.
        head_dim (int): Dimension of each head in multi-head attention.
        user_embedding_dim (int): Dimension of user embeddings.
        category_embedding_dim (int): Dimension of category embeddings.
        subCategory_embedding_dim (int): Dimension of sub-category embeddings.
        dropout_rate (float): Dropout rate for regularization.
        no_self_connection (bool): Whether to disable self-connection in the graph.
        no_adjacent_normalization (bool): Whether to disable normalization of the adjacency matrix.
        gcn_normalization_type (str): Type of normalization for GCN, can be 'symmetric' or 'asymmetric'.
        gcn_layer_num (int): Number of layers in GCN.
        no_gcn_residual (bool): Whether to disable residual connections in GCN.
        gcn_layer_norm (bool): Whether to apply layer normalization in GCN.
        hidden_dim (int): Hidden dimension for encoders.
        Alpha (float): Weight for reconstruction loss in DAE.
        long_term_masking_probability (float): Probability of masking long-term representation for LSTUR.
        personalized_embedding_dim (int): Dimension of personalized embeddings for NPA.
        HDC_window_size (int): Window size for HDC in FIM.
        HDC_filter_num (int): Number of filters in HDC for FIM.
        conv3D_filter_num_first (int): Number of filters in the first layer of 3D convolution for FIM.
        conv3D_kernel_size_first (int): Kernel size of the first layer of 3D convolution for FIM.
        conv3D_filter_num_second (int): Number of filters in the second layer of 3D convolution for FIM.
        conv3D_kernel_size_second (int): Kernel size of the second layer of 3D convolution for FIM.
        maxpooling3D_size (int): Size of 3D max pooling for FIM.
        maxpooling3D_stride (int): Stride of 3D max pooling for FIM.
        click_predictor (str): Type of click predictor, can be 'dot_product', 'mlp', 'sigmoid', or 'FIM'.
        train_root (str): Root directory for training data.
        dev_root (str): Root directory for development data.
        test_root (str): Root directory for test data.
        config_dir (str): Directory for saving configuration files.
        model_dir (str): Directory for saving model checkpoints.
        best_model_dir (str): Directory for saving the best model.
        dev_res_dir (str): Directory for saving development results.
        test_res_dir (str): Directory for saving test results.
        result_dir (str): Directory for saving final results.
    Methods:
        __init__(model='TANR'): Initializes the configuration with default values and loads model-specific parameters.
        set_cuda(): Sets up the CUDA environment for GPU training.
        preliminary_setup(): Prepares the dataset and checks for necessary files.
    Usage:
        config = Config(model='TANR')
    '''
    def __init__(self, model='CENNEWSREC'):
        self.model = model.upper()
        self.mode = 'train'
        self.dev_model_path = ''
        self.test_model_path = ''
        self.test_output_file = ''
        self.device_id = 0
        self.seed = 0
        self.config_file = ''

        self.root = "/home/wanro238/Pypro/NewsRecTorch"
        self.dataset = 'small'
        self.tokenizer = 'MIND'
        self.word_threshold = 3
        self.max_title_length = 32
        self.max_abstract_length = 128
        self.negative_sample_num = 4
        self.max_history_num = 50
        self.epoch = 20

        self.batch_size = 32
        self.lr = 1e-4
        self.weight_decay = 0
        self.gradient_clip_norm = 4
        self.world_size = 1
        self.dev_criterion = 'avg'
        self.early_stopping_epoch = 5
        self.category_embedding_dim = 50
        self.subCategory_embedding_dim = 50
        self.entity_embedding_dim = 100
        self.context_embedding_dim = 100
        self.dropout_rate = 0.2
        self.no_self_connection = False
        self.no_adjacent_normalization = False
        self.Alpha = 0.1
        self.gcn_normalization_type = 'symmetric'

        # self.word_embedding_dim = 300
        # self.entity_embedding_dim = 100
        # self.context_embedding_dim = 100
        # self.cnn_method = 'naive'
        # self.cnn_kernel_num = 400
        # self.cnn_window_size = 3
        # self.attention_dim = 200
        # self.head_num = 20
        # self.head_dim = 20
        # self.user_embedding_dim = 50
        # self.long_term_masking_probability = 0.1
        # self.personalized_embedding_dim = 200
        # self.HDC_window_size = 3
        # self.HDC_filter_num = 150
        # self.conv3D_filter_num_first = 32
        # self.conv3D_kernel_size_first = 3
        # self.conv3D_filter_num_second = 16
        # self.conv3D_kernel_size_second = 3
        # self.maxpooling3D_size = 3
        # self.maxpooling3D_stride = 3
        self.click_predictor = 'dot_product'

        self.train_root = self.root + '/MIND-%s/train' % self.dataset
        self.dev_root = self.root + '/MIND-%s/dev' % self.dataset
        self.test_root = self.root + '/MIND-%s/test' % self.dataset
        # if self.dataset == 'small': # suggested configuration for MIND-small
        #     self.dropout_rate = 0.25
        #     self.gcn_layer_num = 3
        # elif self.dataset == '200k': # suggested configuration for MIND-200k
        #     self.dropout_rate = 0.2
        #     self.gcn_layer_num = 4
        #     self.epoch = 8
        # else: # suggested configuration for MIND-large
        #     self.dropout_rate = 0.1
        #     self.gcn_layer_num = 4
        #     self.epoch = 6
        self.seed = self.seed if self.seed >= 0 else (int)(time.time())
        json_path = f'Config/{model.lower()}.json'
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                model_params = json.load(f)
            for k, v in model_params.items():
                setattr(self, k, v)
        else:
            print(f"Warning: Model config file {json_path} not found, using default parameters")

        self.attribute_dict = self.__dict__.copy()
        if self.config_file != '':
            if os.path.exists(self.config_file):
                print('Get experiment settings from the config file : ' + self.config_file)
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    configs = json.load(f)
                    for attribute in self.attribute_dict:
                        if attribute in configs:
                            setattr(self, attribute, configs[attribute])
                            self.attribute_dict[attribute] = configs[attribute]
            else:
                raise Exception('Config file does not exist : ' + self.config_file)
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
        torch.cuda.set_device(self.device_id)
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
        if not all(list(map(os.path.exists, dataset_files))):
            exec('prepare_MIND_%s()' % self.dataset)
            print("Please prepare the dataset first!!!")

        model_name = self.model
        mkdirs = lambda x: os.makedirs(x) if not os.path.exists(x) else None
        self.config_dir = 'logs/configs/' + self.dataset + '/' + model_name
        self.model_dir = 'logs/models/' + self.dataset + '/' + model_name
        self.best_model_dir = 'logs/best_model/' + self.dataset + '/' + model_name
        self.dev_res_dir = 'logs/dev/res/' + self.dataset + '/' + model_name
        self.test_res_dir = 'logs/test/res/' + self.dataset + '/' + model_name
        self.result_dir = 'logs/results/' + self.dataset + '/' + model_name
        self.running_files_dir = 'data'
        mkdirs(self.config_dir)
        mkdirs(self.model_dir)
        mkdirs(self.best_model_dir)
        mkdirs('logs/dev/ref')
        mkdirs(self.dev_res_dir)
        mkdirs('logs/test/ref')
        mkdirs(self.test_res_dir)
        mkdirs(self.result_dir)
        mkdirs(self.running_files_dir)
        if not os.path.exists('logs/dev/ref/truth-%s.txt' % self.dataset):
            with open(os.path.join(self.dev_root, 'behaviors.tsv'), 'r', encoding='utf-8') as dev_f:
                with open('logs/dev/ref/truth-%s.txt' % self.dataset, 'w', encoding='utf-8') as truth_f:
                    for dev_ID, line in enumerate(dev_f):
                        # impression_ID, user_ID, time, history, impressions, is_fake = line.split('\t')
                        impression_ID, user_ID, time, history, impressions = line.split('\t')
                        labels = [int(impression[-1]) for impression in impressions.strip().split(' ')]
                        truth_f.write(('' if dev_ID == 0 else '\n') + str(dev_ID + 1) + ' ' + str(labels).replace(' ', ''))
        if self.dataset != 'large':
            if not os.path.exists('logs/test/ref/truth-%s.txt' % self.dataset):
                with open(os.path.join(self.test_root, 'behaviors.tsv'), 'r', encoding='utf-8') as test_f:
                    with open('logs/test/ref/truth-%s.txt' % self.dataset, 'w', encoding='utf-8') as truth_f:
                        for test_ID, line in enumerate(test_f):
                            # impression_ID, user_ID, time, history, impressions, is_fake = line.split('\t')
                            impression_ID, user_ID, time, history, impressions = line.split('\t')
                            labels = [int(impression[-1]) for impression in impressions.strip().split(' ')]
                            truth_f.write(('' if test_ID == 0 else '\n') + str(test_ID + 1) + ' ' + str(labels).replace(' ', ''))
        else:
            self.prediction_dir = 'logs/prediction/large/' + model_name
            mkdirs(self.prediction_dir)



if __name__ == '__main__':
    config = Config()

import os
import json
import pickle
import collections
import re
from datetime import time
from random import randint

import pandas as pd
from nltk.tokenize import word_tokenize
from tqdm import tqdm

from config import Config
import torch
import numpy as np

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

pat = re.compile(r"[\w]+|[.,!?;|]")

# --- Image Encoding Functions ---

def get_image_transforms():
    """Returns a composition of image transformations for ResNet-50."""
    import torchvision.transforms as transforms
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

def get_resnet_model():
    """Initializes and returns a pre-trained ResNet-50 model."""
    import torchvision.models as models
    resnet50 = models.resnet50(pretrained=True)
    # Remove the final fully connected layer to get the feature vector
    model = torch.nn.Sequential(*(list(resnet50.children())[:-1]))
    model.eval()
    return model

def extract_image_features(image_dir, model, transforms):
    """
    Extracts feature vectors from all images in a directory.
    """
    from PIL import Image
    from torch.autograd import Variable
    print(f"🖼️  Extracting features from images in {image_dir}...")
    image_embedding_dict = {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    image_files = [f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))]
    for image_name in tqdm(image_files):
        try:
            article_id = os.path.splitext(image_name)[0]
            image_path = os.path.join(image_dir, image_name)
            
            # Open image and handle potential RGBA to RGB conversion
            img = Image.open(image_path).convert("RGB")
            
            transformed_img = transforms(img)
            batch = Variable(transformed_img.unsqueeze(0))
            batch = batch.to(device)

            with torch.no_grad():
                feature_vector = model(batch)
            
            # Flatten the feature vector and move to CPU
            feature_vector_flat = feature_vector.view(feature_vector.size(0), -1)
            image_embedding_dict[article_id] = feature_vector_flat.cpu().numpy().squeeze()
        except Exception as e:
            print(f"Could not process image {image_name}: {e}")
            
    print(f"✅ Extracted features for {len(image_embedding_dict)} images.")
    return image_embedding_dict


# Step 3: Build embedding matrix from word_dict using XLM-Roberta-large
def build_pretrain_word_embedding(word_dict, embedding_dim, output_pkl_path):
    print("🛠️ Building word embedding matrix using XLM-Roberta-large...")
    from transformers import AutoTokenizer, AutoModel
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")

    model_name = "xlm-roberta-large"
    print(f"Loading {model_name}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name).to(device)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    model.eval()
    
    model_dim = model.config.hidden_size
    if embedding_dim != model_dim:
        print(f"⚠️ Warning: Requested embedding_dim {embedding_dim} does not match model dim {model_dim}. Using {model_dim}.")
        embedding_dim = model_dim
    
    # Initialize with zeros
    word_embedding_vectors = torch.zeros([len(word_dict), embedding_dim])
    
    batch_size = 64
    words = list(word_dict.keys())
    
    # Process in batches
    for i in tqdm(range(0, len(words), batch_size), desc="Generating embeddings"):
        batch_words = words[i:i+batch_size]
        
        inputs = tokenizer(batch_words, padding=True, truncation=True, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            # Mean pooling
            attention_mask = inputs['attention_mask']
            token_embeddings = outputs.last_hidden_state
            
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            batch_embeddings = sum_embeddings / sum_mask
            
        batch_embeddings = batch_embeddings.cpu()
            
        for j, word in enumerate(batch_words):
            index = word_dict[word]
            if 0 <= index < len(word_embedding_vectors):
                word_embedding_vectors[index] = batch_embeddings[j]

    with open(output_pkl_path, 'wb') as f:
        pickle.dump(word_embedding_vectors, f)

    print(f"✅ Saved to {output_pkl_path}, shape: {word_embedding_vectors.shape}")


class EBNeRD_Corpus:
    @staticmethod
    def preprocess(config: Config):
        user_ID_file = 'cache/ebnerd/user_ID-%s.json' % config.dataset_size
        news_ID_file = 'cache/ebnerd/news_ID-%s.json' % config.dataset_size
        category_file = 'cache/ebnerd/category-%s.json' % config.dataset_size
        subCategory_file = 'cache/ebnerd/subCategory-%s.json' % config.dataset_size
        sentiment_file = 'cache/ebnerd/sentiment-%s.json' % config.dataset_size
        sentiment_label_file = 'cache/ebnerd/sentiment_label-%s.json' % config.dataset_size
        vocabulary_file = 'cache/ebnerd/vocabulary-' + str(config.word_threshold) + '-' + config.tokenizer + '-' + str(config.max_title_length) + '-' + str(config.max_abstract_length) + '-' + config.dataset_size + '.json'
        word_embedding_file = 'cache/%s/word_embedding-' % config.dataset_name + str(config.word_threshold) + '-' + str(
            config.word_embedding_dim) + '-' + config.tokenizer + '-' + str(
            config.max_title_length) + '-' + str(config.max_abstract_length) + '-' + config.dataset_size + '.pkl'
        entity_file = 'cache/ebnerd/entity-%s.json' % config.dataset_size
        entity_embedding_file = 'cache/ebnerd/entity_embedding-%s.pkl' % config.dataset_size
        context_embedding_file = 'cache/ebnerd/context_embedding-%s.pkl' % config.dataset_size
        image_embedding_file = 'cache/ebnerd/image_embedding-%s.pkl' % config.dataset_size
        user_history_graph_file = 'cache/ebnerd/user_history_graph-' + str(config.max_history_num) + ('' if config.no_self_connection else '-self') + ('' if config.no_adjacent_normalization else '-normalize-' + config.gcn_normalization_type) + '-' + config.dataset_size + '.pkl'
        
        preprocessed_data_files = [user_ID_file, news_ID_file, category_file, subCategory_file, vocabulary_file, word_embedding_file, entity_file, entity_embedding_file, context_embedding_file, user_history_graph_file, sentiment_file, sentiment_label_file]
        if config.model.lower() == 'mmrec':
            preprocessed_data_files.append(image_embedding_file)

        if not all(list(map(os.path.exists, preprocessed_data_files))):
            user_ID_dict = {'<UNK>': 0}
            news_ID_dict = {'<PAD>': 0}
            category_dict = {}
            subCategory_dict = {}
            sentiment_dict = {}
            news_sentiment_dict = {}
            word_dict = {'<PAD>': 0, '<UNK>': 1}
            word_counter = collections.Counter()
            entity_dict = {'<PAD>': 0, '<UNK>': 1}
            news_category_dict = {}

            # 1. user ID dictionary
            behaviors_df = pd.read_parquet(os.path.join(config.train_root, 'behaviors.parquet'))

            for user_id in behaviors_df['uid'].unique():
                user_id = str(user_id)
                if user_id not in user_ID_dict:
                    user_ID_dict[user_id] = len(user_ID_dict)
            with open(user_ID_file, 'w', encoding='utf-8') as user_ID_f:
                json.dump(user_ID_dict, user_ID_f)

            # 2. news ID dictionary & news category dictionary & news subCategory dictionary & sentiment dictionary
            for i, prefix in enumerate([config.train_root, config.dev_root, config.test_root]):
                news_parquet_file = os.path.join(prefix, 'news.parquet')
                df_news = pd.read_parquet(news_parquet_file)

                for _, row in df_news.iterrows():
                    news_ID = str(row['nid']).strip()
                    category = str(row['category']).strip()
                    subCategory = str(row['subcategory']).strip()
                    title = str(row['title']).lower()
                    abstract = str(row['abstract']).lower() if 'abstract' in row else ''
                    sentiment_label = str(row['sentiment_label']).strip()

                    if news_ID not in news_ID_dict:
                        news_ID_dict[news_ID] = len(news_ID_dict)
                        if category not in category_dict:
                            category_dict[category] = len(category_dict)
                        if subCategory not in subCategory_dict:
                            subCategory_dict[subCategory] = len(subCategory_dict)
                        if sentiment_label not in sentiment_dict:
                            sentiment_dict[sentiment_label] = len(sentiment_dict)

                        words = pat.findall(title) if config.tokenizer == 'MIND' else word_tokenize(title)
                        for word in words:
                            if is_number(word):
                                word_counter['<NUM>'] += 1
                            else:
                                if i == 0:  # training set
                                    word_counter[word] += 1
                                else:
                                    if word in word_counter:
                                        word_counter[word] += 1

                        if abstract:
                            words = pat.findall(abstract) if config.tokenizer == 'MIND' else word_tokenize(abstract)
                            for word in words:
                                if is_number(word):
                                    word_counter['<NUM>'] += 1
                                else:
                                    if i == 0:  # training set
                                        word_counter[word] += 1
                                    else:
                                        if word in word_counter:
                                            word_counter[word] += 1

                    news_category_dict[news_ID] = category_dict[category]
                    news_sentiment_dict[news_ID] = sentiment_dict[sentiment_label]

            # Save dictionaries
            with open(news_ID_file, 'w', encoding='utf-8') as news_ID_f:
                json.dump(news_ID_dict, news_ID_f)
            with open(category_file, 'w', encoding='utf-8') as category_f:
                json.dump(category_dict, category_f)
            with open(subCategory_file, 'w', encoding='utf-8') as subCategory_f:
                json.dump(subCategory_dict, subCategory_f)
            with open(sentiment_file, 'w', encoding='utf-8') as sentiment_f:
                json.dump(news_sentiment_dict, sentiment_f)
            with open(sentiment_label_file, 'w', encoding='utf-8') as sentiment_label_f:
                json.dump(sentiment_dict, sentiment_label_f)

            # 3. word dictionary
            word_counter_list = [[word, word_counter[word]] for word in word_counter]
            word_counter_list.sort(key=lambda x: x[1], reverse=True) # sort by word frequency
            filtered_word_counter_list = list(filter(lambda x: x[1] >= config.word_threshold, word_counter_list))
            for i, word in enumerate(filtered_word_counter_list):
                word_dict[word[0]] = i + 2
            with open(vocabulary_file, 'w', encoding='utf-8') as vocabulary_f:
                json.dump(word_dict, vocabulary_f)

            # 4. Word embedding using XLM-Roberta-large
            if not os.path.exists(word_embedding_file):
                build_pretrain_word_embedding(
                    word_dict,
                    embedding_dim=config.word_embedding_dim,
                    output_pkl_path=word_embedding_file
                )
            
            # 5. Image embeddings
            if config.model.lower() == 'mmrec':
                if not os.path.exists(image_embedding_file):
                    print("Preprocessing image embeddings...")
                    model = get_resnet_model()
                    transforms = get_image_transforms()
                    image_embedding_dict = extract_image_features(config.downloaded_images_file, model, transforms)
                    with open(image_embedding_file, 'wb') as f:
                        pickle.dump(image_embedding_dict, f)
                    print("Image embeddings preprocessed and saved.")

            # build graph
            if config.dataset_size != 'large' and config.model in ['CNE-SUE', 'CNRCL']:
                category_num = len(category_dict)
                graph_size = config.max_history_num + category_num  # |V_n| + |V_p|
                prefix_mode = ['train', 'dev', 'test']
                user_history_graph_data = {}

                for prefix_index, prefix in enumerate([config.train_root, config.dev_root, config.test_root]):
                    mode = prefix_mode[prefix_index]
                    behaviors_parquet = os.path.join(prefix, 'behaviors.parquet')

                    df_behaviors = pd.read_parquet(behaviors_parquet)
                    user_history_num = len(df_behaviors)

                    graph_path = os.path.join('cache/ebnerd', f'{config.dataset_size}-{mode}_user_history_graph.npy')
                    if os.path.exists(graph_path):
                        os.remove(graph_path)
                    user_history_graph = np.memmap(graph_path, dtype=np.float32, mode='w+', shape=(user_history_num, graph_size, graph_size))
                    user_history_category_mask = np.zeros([user_history_num, category_num + 1], dtype=np.float32)
                    user_history_category_indices = np.full([user_history_num, config.max_history_num], category_num,
                                                            dtype=np.int64)

                    for line_index, row in df_behaviors.iterrows():
                        history = str(row['history']).strip()

                        if config.no_self_connection:
                            history_graph = np.zeros([graph_size, graph_size], dtype=np.float32)
                        else:
                            history_graph = np.identity(graph_size, dtype=np.float32)

                        history_category_mask = np.zeros(category_num + 1, dtype=np.float32)
                        history_category_indices = np.full(config.max_history_num, category_num, dtype=np.int64)

                        if len(history.strip('[] ')) > 0:
                            history_news_ID = history.strip('[] ').split(',')
                            history_news_ID = [nid.strip() for nid in history_news_ID if nid.strip() != '']

                            offset = max(0, len(history_news_ID) - config.max_history_num)
                            history_news_num = min(len(history_news_ID), config.max_history_num)

                            for i in range(history_news_num):
                                news_id = history_news_ID[i + offset]
                                category_index = news_category_dict.get(news_id, category_num)

                                if category_index >= category_num:
                                    continue  # skip invalid category

                                history_category_mask[category_index] = 1.0
                                history_category_indices[i] = category_index

                                history_graph[i, config.max_history_num + category_index] = 1
                                history_graph[config.max_history_num + category_index, i] = 1

                                for j in range(i + 1, history_news_num):
                                    other_news_id = history_news_ID[j + offset]
                                    other_category_index = news_category_dict.get(other_news_id, category_num)

                                    if other_category_index >= category_num:
                                        continue  # skip invalid category

                                    if category_index == other_category_index:
                                        history_graph[i, j] = 1
                                        history_graph[j, i] = 1
                                    else:
                                        history_graph[
                                            config.max_history_num + category_index, config.max_history_num + other_category_index] = 1
                                        history_graph[
                                            config.max_history_num + other_category_index, config.max_history_num + category_index] = 1

                            if not config.no_adjacent_normalization:
                                degrees = history_graph.sum(axis=1)
                                if config.gcn_normalization_type == 'asymmetric':
                                    D_inv = np.diag(1 / np.clip(degrees, a_min=1e-12, a_max=None))
                                    history_graph = D_inv @ history_graph
                                else:
                                    D_inv_sqrt = np.diag(np.sqrt(1 / np.clip(degrees, a_min=1e-12, a_max=None)))
                                    history_graph = D_inv_sqrt @ history_graph @ D_inv_sqrt

                        user_history_graph[line_index] = history_graph
                        user_history_category_mask[line_index] = history_category_mask
                        user_history_category_indices[line_index] = history_category_indices

                    user_history_graph.flush()
                    user_history_graph_data[f'{mode}_user_history_graph_path'] = graph_path
                    user_history_graph_data[f'{mode}_user_history_category_mask'] = user_history_category_mask
                    user_history_graph_data[f'{mode}_user_history_category_indices'] = user_history_category_indices
                    del user_history_graph

                with open(user_history_graph_file, 'wb') as f:
                    pickle.dump(user_history_graph_data, f)

    def __init__(self, config: Config):
        self.config = config

        # preprocess cache
        EBNeRD_Corpus.preprocess(config)

        with open('cache/ebnerd/user_ID-%s.json' % config.dataset_size, 'r', encoding='utf-8') as user_ID_f:
            self.user_ID_dict = json.load(user_ID_f)
            config.user_num = len(self.user_ID_dict)

        with open('cache/ebnerd/news_ID-%s.json' % config.dataset_size, 'r', encoding='utf-8') as news_ID_f:
            self.news_ID_dict = json.load(news_ID_f)
            self.news_num = len(self.news_ID_dict)

        with open('cache/ebnerd/category-%s.json' % config.dataset_size, 'r', encoding='utf-8') as category_f:
            self.category_dict = json.load(category_f)
            config.category_num = len(self.category_dict)

        with open('cache/ebnerd/subCategory-%s.json' % config.dataset_size, 'r', encoding='utf-8') as subCategory_f:
            self.subCategory_dict = json.load(subCategory_f)
            config.subCategory_num = len(self.subCategory_dict)

        with open('cache/ebnerd/sentiment-%s.json' % config.dataset_size, 'r', encoding='utf-8') as sentiment_f:
            self.sentiment_dict = json.load(sentiment_f)

        with open('cache/ebnerd/sentiment_label-%s.json' % config.dataset_size, 'r', encoding='utf-8') as sentiment_label_f:
            self.sentiment_label_dict = json.load(sentiment_label_f)
            config.num_sent_classes = 3 

        with open('cache/ebnerd/vocabulary-' + str(config.word_threshold) + '-' + config.tokenizer + '-' + str(config.max_title_length) + '-' + str(config.max_abstract_length) + '-' + config.dataset_size + '.json', 'r', encoding='utf-8') as vocabulary_f:
            self.word_dict = json.load(vocabulary_f)
            config.vocabulary_size = len(self.word_dict)

        image_embedding_dict = {}
        if config.model.lower() == 'mmrec':
            with open('cache/ebnerd/image_embedding-%s.pkl' % config.dataset_size, 'rb') as f:
                image_embedding_dict = pickle.load(f)

        if config.dataset_size != 'large' and config.model in ['CNE-SUE', 'CNRCL']:
            with open('cache/ebnerd/user_history_graph-' + str(config.max_history_num) + ('' if config.no_self_connection else '-self') + ('' if config.no_adjacent_normalization else '-normalize-' + config.gcn_normalization_type) + '-' + config.dataset_size + '.pkl', 'rb') as user_history_graph_f:
                user_history_data = pickle.load(user_history_graph_f)
                graph_size = config.max_history_num + config.category_num

                train_user_history_num = len(pd.read_parquet(os.path.join(config.train_root, 'behaviors.parquet')))
                train_graph_shape = (train_user_history_num, graph_size, graph_size)
                self.train_user_history_graph = np.memmap(user_history_data['train_user_history_graph_path'], dtype=np.float32, mode='r', shape=train_graph_shape)
                self.train_user_history_category_mask = user_history_data['train_user_history_category_mask']
                self.train_user_history_category_indices = user_history_data['train_user_history_category_indices']

                dev_user_history_num = len(pd.read_parquet(os.path.join(config.dev_root, 'behaviors.parquet')))
                dev_graph_shape = (dev_user_history_num, graph_size, graph_size)
                self.dev_user_history_graph = np.memmap(user_history_data['dev_user_history_graph_path'], dtype=np.float32, mode='r', shape=dev_graph_shape)
                self.dev_user_history_category_mask = user_history_data['dev_user_history_category_mask']
                self.dev_user_history_category_indices = user_history_data['dev_user_history_category_indices']

                test_user_history_num = len(pd.read_parquet(os.path.join(config.test_root, 'behaviors.parquet')))
                test_graph_shape = (test_user_history_num, graph_size, graph_size)
                self.test_user_history_graph = np.memmap(user_history_data['test_user_history_graph_path'], dtype=np.float32, mode='r', shape=test_graph_shape)
                self.test_user_history_category_mask = user_history_data['test_user_history_category_mask']
                self.test_user_history_category_indices = user_history_data['test_user_history_category_indices']
        else:
            self.train_user_history_graph = None
            self.train_user_history_category_mask = None
            self.train_user_history_category_indices = None
            self.dev_user_history_graph = None
            self.dev_user_history_category_mask = None
            self.dev_user_history_category_indices = None
            self.test_user_history_graph = None
            self.test_user_history_category_mask = None
            self.test_user_history_category_indices = None

        # meta cache
        self.negative_sample_num = config.negative_sample_num                                          
        self.max_history_num = config.max_history_num                                                   
        self.max_title_length = config.max_title_length                                                 
        self.max_abstract_length = config.max_abstract_length                                           
        self.news_category = np.zeros([self.news_num], dtype=np.int32)                                  
        self.news_subCategory = np.zeros([self.news_num], dtype=np.int32)                               
        self.news_sentiment = np.zeros([self.news_num], dtype=np.float32)
        
        if config.model.lower() == 'mmrec':
            self.news_image_embeddings = np.zeros([self.news_num, 2048], dtype=np.float32)
        else:
            self.news_image_embeddings = None

        self.news_title_text = np.zeros([self.news_num, self.max_title_length], dtype=np.int32)         
        self.news_title_mask = np.zeros([self.news_num, self.max_title_length], dtype=np.float32)       
        self.news_title_entity = np.zeros([self.news_num, self.max_title_length], dtype=np.int32)       
        self.news_abstract_text = np.zeros([self.news_num, self.max_abstract_length], dtype=np.int32)   
        self.news_abstract_mask = np.zeros([self.news_num, self.max_abstract_length], dtype=np.float32) 
        self.news_abstract_entity = np.zeros([self.news_num, self.max_abstract_length], dtype=np.int32) 

        self.train_behaviors = []                                                                       
        self.dev_behaviors = []                                                                         
        self.dev_indices = []                                                                           
        self.test_behaviors = []                                                                        
        self.test_indices = []                                                                          
        self.title_word_num = 0
        self.abstract_word_num = 0

        news_ID_set = set(['<PAD>'])
        news_records = []
        for prefix in [config.train_root, config.dev_root, config.test_root]:
            news_parquet = os.path.join(prefix, 'news.parquet')
            df_news = pd.read_parquet(news_parquet)
            for _, row in df_news.iterrows():
                news_ID = str(row['nid']).strip()
                if news_ID not in news_ID_set:
                    news_records.append(row)
                    news_ID_set.add(news_ID)
        
        assert self.news_num == len(news_ID_set), f'news num mismatch {self.news_num} v.s. {len(news_ID_set)}'
        
        for row in news_records:
            news_ID = str(row['nid']).strip()
            category = str(row['category']).strip()
            subCategory = str(row['subcategory']).strip()
            sentiment_label = str(row['sentiment_label']).strip()
            sentiment_score = float(row['sentiment_score']) if 'sentiment_score' in row else 0.0

            if sentiment_label == 'Positive':
                final_score = sentiment_score
            elif sentiment_label == 'Negative':
                final_score = -sentiment_score
            else:
                final_score = 0.0

            title = str(row['title'])
            abstract = str(row['abstract'])
            index = self.news_ID_dict[news_ID]
            
            if config.model.lower() == 'mmrec' and news_ID in image_embedding_dict:
                self.news_image_embeddings[index] = image_embedding_dict[news_ID]
            
            self.news_category[index] = self.category_dict.get(category, 0)
            self.news_subCategory[index] = self.subCategory_dict.get(subCategory, 0)
            self.news_sentiment[index] = final_score

            words = pat.findall(title.lower()) if config.tokenizer == 'MIND' else word_tokenize(title.lower())
            offset_index = 0
            for i, word in enumerate(words):
                if i == self.max_title_length:
                    break
                if is_number(word):
                    self.news_title_text[index][i] = self.word_dict['<NUM>']
                else:
                    self.news_title_text[index][i] = self.word_dict.get(word, 1)
                self.news_title_mask[index][i] = 1
            self.title_word_num += len(words)

            words = pat.findall(abstract.lower()) if config.tokenizer == 'MIND' else word_tokenize(abstract.lower())
            for i, word in enumerate(words):
                if i == self.max_abstract_length:
                    break
                if is_number(word):
                    self.news_abstract_text[index][i] = self.word_dict['<NUM>']
                else:
                    self.news_abstract_text[index][i] = self.word_dict.get(word, 1)
                self.news_abstract_mask[index][i] = 1
            self.abstract_word_num += len(words)

        self.news_title_mask[0][0] = 1  
        self.news_abstract_mask[0][0] = 1  

        def process_behavior_df(df, mode='train'):
            for behavior_index, row in df.iterrows():
                user_ID = str(row['uid'])
                history = row['history']
                labels = row.get('labels', None)
                impressions = row['candidates']
                
                click_impressions = []
                non_click_impressions = []
                if mode == 'train' and labels is not None:
                    labels_list = [int(l) for l in labels]
                    impressions_list = [str(x).strip() for x in impressions]
                    for impression, label in zip(impressions_list, labels_list):
                        imp_id = self.news_ID_dict[impression.strip()]
                        if label == 0:
                            non_click_impressions.append(imp_id)
                        else:
                            click_impressions.append(imp_id)
                else:
                    impressions_list = [str(x).strip() for x in impressions]
                
                if isinstance(history, str):
                    history_list = list(
                        map(lambda x: self.news_ID_dict[x], history.strip('[]').split(','))) if history.strip(
                        '[]') else []
                elif isinstance(history, list) or isinstance(history, np.ndarray):
                    history_list = list(map(lambda x: self.news_ID_dict[str(x).strip()], history)) if len(history) > 0 else []
                else:
                    history_list = []

                padding_num = max(0, self.max_history_num - len(history_list))
                user_history = history_list[-self.max_history_num:] + [0] * padding_num
                user_history_mask = np.zeros(self.max_history_num, dtype=np.float32)
                user_history_mask[:min(len(history_list), self.max_history_num)] = 1.0

                if mode == 'train':
                    for click_imp in click_impressions:
                        if len(non_click_impressions) > 0:
                            self.train_behaviors.append([
                                self.user_ID_dict[user_ID],
                                user_history,
                                user_history_mask,
                                click_imp,
                                non_click_impressions,
                                behavior_index
                            ])
                elif mode == 'dev':
                    for impression in impressions_list:
                        imp_id = self.news_ID_dict[impression.strip()]
                        self.dev_indices.append(behavior_index)
                        self.dev_behaviors.append([
                            self.user_ID_dict.get(user_ID, 0),
                            user_history,
                            user_history_mask,
                            imp_id,
                            behavior_index
                        ])
                elif mode == 'test':
                    for impression in impressions_list:
                        imp_id = self.news_ID_dict.get(impression.strip(), 0)
                        self.test_indices.append(behavior_index)
                        self.test_behaviors.append([
                            self.user_ID_dict.get(user_ID, 0),
                            user_history,
                            user_history_mask,
                            imp_id,
                            behavior_index
                        ])

        train_path = os.path.join(config.train_root, 'behaviors.parquet')
        dev_path = os.path.join(config.dev_root, 'behaviors.parquet')
        test_path = os.path.join(config.test_root, 'behaviors.parquet')

        train_df = pd.read_parquet(train_path)
        dev_df = pd.read_parquet(dev_path)
        test_df = pd.read_parquet(test_path)

        process_behavior_df(train_df, 'train')
        process_behavior_df(dev_df, 'dev')
        process_behavior_df(test_df, 'test')

import time
from numpy.random import randint
import torch.utils.data as data

class Ebnerd_Train_Dataset(data.Dataset):
    def __init__(self, corpus: EBNeRD_Corpus):
        self.config = corpus.config
        self.negative_sample_num = self.config.negative_sample_num
        self.news_category = corpus.news_category
        self.news_subCategory = corpus.news_subCategory
        self.news_sentiment = corpus.news_sentiment
        
        if self.config.model.lower() == 'mmrec':
            self.news_image_embeddings = corpus.news_image_embeddings
        else:
            self.news_image_embeddings = None

        self.news_title_text =  corpus.news_title_text
        self.news_title_mask = corpus.news_title_mask
        self.news_title_entity = corpus.news_title_entity
        self.news_abstract_text =  corpus.news_abstract_text
        self.news_abstract_mask = corpus.news_abstract_mask
        self.news_abstract_entity = corpus.news_abstract_entity
        self.user_history_graph = corpus.train_user_history_graph
        self.user_history_category_mask = corpus.train_user_history_category_mask
        self.user_history_category_indices = corpus.train_user_history_category_indices
        self.train_behaviors = corpus.train_behaviors
        self.train_samples = [[0 for _ in range(1 + self.negative_sample_num)] for __ in range(len(self.train_behaviors))]
        self.num = len(self.train_behaviors)

    def negative_sampling(self, rank=None):
        print('\n%sBegin negative sampling, training sample num : %d' % ('' if rank is None else ('rank ' + str(rank) + ' : '), self.num))
        start_time = time.time()
        for i, train_behavior in enumerate(self.train_behaviors):
            self.train_samples[i][0] = train_behavior[3]
            negative_samples = train_behavior[4]
            news_num = len(negative_samples)
            if news_num <= self.negative_sample_num:
                for j in range(self.negative_sample_num):
                    self.train_samples[i][j + 1] = negative_samples[j % news_num]
            else:
                used_negative_samples = set()
                for j in range(self.negative_sample_num):
                    while True:
                        k = randint(0, news_num)
                        if k not in used_negative_samples:
                            self.train_samples[i][j + 1] = negative_samples[k]
                            used_negative_samples.add(k)
                            break
        end_time = time.time()
        print('%sEnd negative sampling, used time : %.3fs' % ('' if rank is None else ('rank ' + str(rank) + ' : '), end_time - start_time))

    def __getitem__(self, index):
        train_behavior = self.train_behaviors[index]
        history_index = train_behavior[1]
        sample_index = self.train_samples[index]
        behavior_index = train_behavior[5]

        if self.config.model in ['CNE-SUE', 'CNRCL']:
            user_history_graph = self.user_history_graph[behavior_index]
            user_history_category_mask = self.user_history_category_mask[behavior_index]
            user_history_category_indices = self.user_history_category_indices[behavior_index]
        else:
            graph_size = self.config.max_history_num + self.config.category_num
            user_history_graph = np.zeros((graph_size, graph_size), dtype=np.float32)
            user_history_category_mask = np.zeros(self.config.category_num + 1, dtype=np.float32)
            user_history_category_indices = np.zeros(self.config.max_history_num, dtype=np.int64)

        if self.config.model.lower() == 'mmrec':
            return (train_behavior[0], self.news_category[history_index], self.news_subCategory[history_index], self.news_title_text[history_index], self.news_title_mask[history_index], self.news_title_entity[history_index], self.news_abstract_text[history_index], self.news_abstract_mask[history_index], self.news_abstract_entity[history_index], train_behavior[2], user_history_graph, user_history_category_mask, user_history_category_indices,
                    self.news_category[sample_index], self.news_subCategory[sample_index], self.news_title_text[sample_index], self.news_title_mask[sample_index], self.news_title_entity[sample_index], self.news_abstract_text[sample_index], self.news_abstract_mask[sample_index], self.news_abstract_entity[sample_index], self.news_sentiment[history_index], self.news_sentiment[sample_index], history_index, sample_index,
                    self.news_image_embeddings[history_index], self.news_image_embeddings[sample_index])
        else:
            return (train_behavior[0], self.news_category[history_index], self.news_subCategory[history_index], self.news_title_text[history_index], self.news_title_mask[history_index], self.news_title_entity[history_index], self.news_abstract_text[history_index], self.news_abstract_mask[history_index], self.news_abstract_entity[history_index], train_behavior[2], user_history_graph, user_history_category_mask, user_history_category_indices,
                    self.news_category[sample_index], self.news_subCategory[sample_index], self.news_title_text[sample_index], self.news_title_mask[sample_index], self.news_title_entity[sample_index], self.news_abstract_text[sample_index], self.news_abstract_mask[sample_index], self.news_abstract_entity[sample_index], self.news_sentiment[history_index], self.news_sentiment[sample_index], history_index, sample_index,
                    np.zeros((len(history_index), 2048), dtype=np.float32), np.zeros((len(sample_index), 2048), dtype=np.float32))

    def __len__(self):
        return self.num

class Ebnerd_DevTest_Dataset(data.Dataset):
    def __init__(self, corpus: EBNeRD_Corpus, mode: str):
        self.config = corpus.config
        assert mode in ['dev', 'test'], 'mode must be chosen from \'dev\' or \'test\''
        self.news_category = corpus.news_category
        self.news_subCategory = corpus.news_subCategory
        self.news_sentiment = corpus.news_sentiment
        
        if self.config.model.lower() == 'mmrec':
            self.news_image_embeddings = corpus.news_image_embeddings
        else:
            self.news_image_embeddings = None

        self.news_title_text =  corpus.news_title_text
        self.news_title_mask = corpus.news_title_mask
        self.news_title_entity = corpus.news_title_entity
        self.news_abstract_text =  corpus.news_abstract_text
        self.news_abstract_mask = corpus.news_abstract_mask
        self.news_abstract_entity = corpus.news_abstract_entity
        self.user_history_graph = corpus.dev_user_history_graph if mode == 'dev' else corpus.test_user_history_graph
        self.user_history_category_mask = corpus.dev_user_history_category_mask if mode == 'dev' else corpus.test_user_history_category_mask
        self.user_history_category_indices = corpus.dev_user_history_category_indices if mode == 'dev' else corpus.test_user_history_category_indices
        self.behaviors = corpus.dev_behaviors if mode == 'dev' else corpus.test_behaviors
        self.num = len(self.behaviors)

    def __getitem__(self, index):
        behavior = self.behaviors[index]
        history_index = behavior[1]
        candidate_news_index = behavior[3]
        behavior_index = behavior[4]

        if self.config.model in ['CNE-SUE', 'CNRCL']:
            user_history_graph = self.user_history_graph[behavior_index]
            user_history_category_mask = self.user_history_category_mask[behavior_index]
            user_history_category_indices = self.user_history_category_indices[behavior_index]
        else:
            graph_size = self.config.max_history_num + self.config.category_num
            user_history_graph = np.zeros((graph_size, graph_size), dtype=np.float32)
            user_history_category_mask = np.zeros(self.config.category_num + 1, dtype=np.float32)
            user_history_category_indices = np.zeros(self.config.max_history_num, dtype=np.int64)

        if self.config.model.lower() == 'mmrec':
            return (behavior[0], self.news_category[history_index], self.news_subCategory[history_index], self.news_title_text[history_index], self.news_title_mask[history_index], self.news_title_entity[history_index], self.news_abstract_text[history_index], self.news_abstract_mask[history_index], self.news_abstract_entity[history_index], behavior[2], user_history_graph, user_history_category_mask, user_history_category_indices,
                    self.news_category[candidate_news_index], self.news_subCategory[candidate_news_index], self.news_title_text[candidate_news_index], self.news_title_mask[candidate_news_index], self.news_title_entity[candidate_news_index], self.news_abstract_text[candidate_news_index], self.news_abstract_mask[candidate_news_index], self.news_abstract_entity[candidate_news_index], self.news_sentiment[history_index], self.news_sentiment[candidate_news_index], history_index, candidate_news_index,
                    self.news_image_embeddings[history_index], self.news_image_embeddings[candidate_news_index])
        else:
            return (behavior[0], self.news_category[history_index], self.news_subCategory[history_index], self.news_title_text[history_index], self.news_title_mask[history_index], self.news_title_entity[history_index], self.news_abstract_text[history_index], self.news_abstract_mask[history_index], self.news_abstract_entity[history_index], behavior[2], user_history_graph, user_history_category_mask, user_history_category_indices,
                    self.news_category[candidate_news_index], self.news_subCategory[candidate_news_index], self.news_title_text[candidate_news_index], self.news_title_mask[candidate_news_index], self.news_title_entity[candidate_news_index], self.news_abstract_text[candidate_news_index], self.news_abstract_mask[candidate_news_index], self.news_abstract_entity[candidate_news_index], self.news_sentiment[history_index], self.news_sentiment[candidate_news_index], history_index, candidate_news_index,
                    np.zeros((len(history_index), 2048), dtype=np.float32), np.zeros(2048, dtype=np.float32))

    def __len__(self):
        return self.num

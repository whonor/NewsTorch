import csv
import ast
import os
import time
import torch
import numpy as np
from numpy.random import randint
from dataset_corpus_preprocessing.EBNeRD_corpus_main import Ebnerd_Train_Dataset
from sklearn.metrics.pairwise import cosine_similarity
import pickle
from tqdm import tqdm

class CNRCL_Ebnerd_Train_Dataset(Ebnerd_Train_Dataset):
    def __init__(self, corpus):
        super(CNRCL_Ebnerd_Train_Dataset, self).__init__(corpus)
        self.corpus = corpus
        self.config = corpus.config
        self.similarity_num = getattr(corpus.config, 'similarity_num', 3) # Default 3 if not set
        # Modify train_samples to hold similarity samples
        # train_samples: [positive, neg1, neg2, ..., negN, sim1, sim2, ...]
        # original train_samples size: 1 + negative_sample_num
        # new size: 1 + negative_sample_num + similarity_num
        self.train_samples = [[0 for _ in range(1 + self.negative_sample_num + self.similarity_num)] for __ in range(len(self.train_behaviors))]
        
        # We need ID_news_dict (int -> str ID)
        self.ID_news_dict = {v: k for k, v in corpus.news_ID_dict.items()}
        self.news_ID_dict = corpus.news_ID_dict
        
        # Ensure similarity file exists or generate it
        self.similarity_file = f'cache/ebnerd/similarities-{corpus.config.dataset_size}.csv'
        if not os.path.exists(self.similarity_file):
            self.generate_similarities()

        # Ensure history similarity file exists or generate it (for Curriculum Learning)
        self.similarity_results_history_file = f'cache/ebnerd/similarities_results_history-{corpus.config.dataset_size}.pkl'
        if not os.path.exists(self.similarity_results_history_file):
            self.generate_similarities_results_history_file()

    def generate_similarities_results_history_file(self):
        print("Generating history-based similarities for Curriculum Learning (CNRCL)...")
        
        # 1. Load Word Embeddings
        word_embedding_file = 'cache/%s/word_embedding-' % self.config.dataset_name + str(self.config.word_threshold) + '-' + str(
            self.config.word_embedding_dim) + '-' + self.config.tokenizer + '-' + str(
            self.config.max_title_length) + '-' + str(self.config.max_abstract_length) + '-' + self.config.dataset_size + '.pkl'
        
        if not os.path.exists(word_embedding_file):
            print("Word embedding file not found. Cannot generate history similarities.")
            return

        with open(word_embedding_file, 'rb') as f:
            word_vectors = pickle.load(f)
            if isinstance(word_vectors, torch.Tensor):
                word_vectors = word_vectors.numpy()

        # 2. Compute News Embeddings (Avg Title)
        news_num = self.corpus.news_num
        # Assuming embedding dim is same as word_vectors dim
        news_embeddings = np.zeros((news_num, word_vectors.shape[1]), dtype=np.float32)
        
        print("Computing news document embeddings...")
        title_text = self.corpus.news_title_text # numpy array [news_num, max_title_len]
        
        # Optimization: Process all at once or in loop. Loop is fine for preprocessing.
        for i in range(news_num):
            indices = title_text[i]
            valid_indices = indices[indices != 0]
            if len(valid_indices) > 0:
                vecs = word_vectors[valid_indices]
                news_embeddings[i] = np.mean(vecs, axis=0)
            # Else remains zero vector

        # 3. Calculate Similarity between User History and Non-Clicked Candidates
        similarity_results = {}
        
        print("Calculating history-candidate similarities...")
        # self.train_behaviors: [user_ID, history, history_mask, click_imp, non_click_impressions, behavior_index]
        
        from tqdm import tqdm
        for behavior_data in tqdm(self.train_behaviors, desc="Processing behaviors"):
            history_indices = behavior_data[1] # list of ints
            non_click_indices = behavior_data[4] # list of ints
            behavior_idx = behavior_data[5] # int
            
            if not non_click_indices:
                similarity_results[behavior_idx] = {}
                continue
                
            real_history = [idx for idx in history_indices if idx != 0]
            
            if not real_history:
                # No history, similarity 0
                similarity_results[behavior_idx] = {nid: 0.0 for nid in non_click_indices}
                continue
            
            # User Vector
            hist_vecs = news_embeddings[real_history]
            user_vec = np.mean(hist_vecs, axis=0)
            
            # Candidate Vectors
            cand_vecs = news_embeddings[non_click_indices]
            
            # Cosine Sim
            user_norm = np.linalg.norm(user_vec)
            cand_norm = np.linalg.norm(cand_vecs, axis=1)
            
            if user_norm == 0:
                sims = np.zeros(len(non_click_indices))
            else:
                # dot product: (num_neg, dim) . (dim,) -> (num_neg,)
                dot_prods = np.dot(cand_vecs, user_vec)
                cand_norm[cand_norm == 0] = 1e-9
                sims = dot_prods / (user_norm * cand_norm)
            
            # Store
            sim_dict = {}
            for k, nid in enumerate(non_click_indices):
                sim_dict[nid] = float(sims[k])
            
            similarity_results[behavior_idx] = sim_dict

        # 4. Save to Pickle
        print(f"Saving history similarities to {self.similarity_results_history_file}...")
        with open(self.similarity_results_history_file, 'wb') as f:
            pickle.dump(similarity_results, f)
        print("Done.")

    def generate_similarities(self):
        print("Generating similarities for CNRCL...")
        # Need word embeddings
        # corpus.config has dataset_name, etc.
        word_embedding_file = 'cache/%s/word_embedding-' % self.corpus.config.dataset_name + str(self.corpus.config.word_threshold) + '-' + str(
            self.corpus.config.word_embedding_dim) + '-' + self.corpus.config.tokenizer + '-' + str(
            self.corpus.config.max_title_length) + '-' + str(self.corpus.config.max_abstract_length) + '-' + self.corpus.config.dataset_size + '.pkl'
        
        if not os.path.exists(word_embedding_file):
            print("Word embedding file not found, cannot generate similarities properly. Using random.")
            return

        with open(word_embedding_file, 'rb') as f:
            # this is a tensor or numpy array [vocab_size, dim]
            word_vectors = pickle.load(f)
            if isinstance(word_vectors, torch.Tensor):
                word_vectors = word_vectors.numpy()

        # Compute news embeddings
        # news_title_text: [news_num, max_title_length] (indices)
        news_num = self.corpus.news_num
        news_embeddings = []
        valid_indices = []
        
        # Limit for large datasets to avoid OOM
        max_news_for_sim = 20000 
        if news_num > max_news_for_sim:
            print(f"Warning: News num {news_num} > {max_news_for_sim}. Calculating similarity on subset or skipping.")
            # We will process all but maybe in batches? For now, let's process all but be careful.
            # If strictly demo/small, it's fine.
        
        print("Computing news document embeddings...")
        for i in range(news_num):
            # word indices
            indices = self.corpus.news_title_text[i]
            # filter padding (0)
            indices = indices[indices != 0]
            if len(indices) > 0:
                # average
                vecs = word_vectors[indices]
                avg_vec = np.mean(vecs, axis=0)
                news_embeddings.append(avg_vec)
                valid_indices.append(i)
            else:
                # Empty title? random or zero
                news_embeddings.append(np.zeros(word_vectors.shape[1]))
                valid_indices.append(i) # keep index alignment

        news_embeddings = np.array(news_embeddings)
        
        # Compute similarity
        # If N is large, doing cosine_similarity(N, N) is huge.
        # We only need top K similar.
        # batch processing
        
        print("Computing cosine similarity and saving...")
        batch_size = 1000
        n_similar = 10
        
        with open(self.similarity_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['news_id', 'most_similar_news'])
            
            for start in range(0, news_num, batch_size):
                end = min(start + batch_size, news_num)
                batch_emb = news_embeddings[start:end]
                
                # sim: [batch, news_num]
                sim_matrix = cosine_similarity(batch_emb, news_embeddings)
                
                for k in range(end - start):
                    idx = start + k
                    # set self sim to -1
                    sim_matrix[k, idx] = -1
                    
                    # get top n
                    top_indices = np.argpartition(sim_matrix[k], -n_similar)[-n_similar:]
                    # sort desc
                    top_indices = top_indices[np.argsort(sim_matrix[k][top_indices])[::-1]]
                    
                    # convert to news IDs
                    current_news_id = self.ID_news_dict[idx]
                    similar_news_ids = [self.ID_news_dict[ti] for ti in top_indices]
                    
                    writer.writerow([current_news_id, str(similar_news_ids)])
                    
        print(f"Similarities saved to {self.similarity_file}")

    def negative_sampling(self, epoch=1, rank=None):
        # Load similarities (Contrastive Learning - Item-Item)
        news_similarities = {}
        if os.path.exists(self.similarity_file):
            with open(self.similarity_file, 'r') as f:
                reader = csv.reader(f, delimiter=',', quotechar='"')
                next(reader, None) # skip header
                for row in reader:
                    if len(row) >= 2:
                        key = row[0]
                        try:
                            value = ast.literal_eval(row[1])
                            news_similarities[key] = value
                        except:
                            pass
        
        # Load history similarities (Curriculum Learning - User-Item)
        similarity_results_history = {}
        if os.path.exists(self.similarity_results_history_file):
            with open(self.similarity_results_history_file, 'rb') as f:
                similarity_results_history = pickle.load(f)

        print('\n%sBegin negative sampling (CNRCL), training sample num : %d' % ('' if rank is None else ('rank ' + str(rank) + ' : '), self.num))
        start_time = time.time()
        
        miss_num = 0
        
        for i, train_behavior in enumerate(self.train_behaviors):
            # train_behavior: [user_ID, history, history_mask, click_imp, non_click_imps, behavior_index]
            self.train_samples[i][0] = train_behavior[3]
            negative_samples = train_behavior[4] # list of ints
            behavior_idx = train_behavior[5]
            news_num = len(negative_samples)

            # Curriculum Learning Logic from CNRCL
            # Use similarity_results_history to sort negative samples by similarity (ascending/hard to easy? or easy to hard?)
            # CNRCL paper/code: "least similar non-click impressions" are selected?
            # Original code:
            # similarities.sort(key=lambda x: x[1]) // Sort by similarity ASCENDING (Least similar first)
            # least_similar_non_click_impressions = [x[0] for x in similarities]
            # It selects negative samples from this sorted list.
            
            # Curriculum Schedule:
            # epoch <= 3: retain 50% (of sorted list? or just slice list?)
            # Original code:
            # if news_num > 46:
            #    index = news_num // 2
            #    if epoch <= 3: negative_samples = negative_samples[:index]
            #    ...
            
            # First, we must sort negative_samples by similarity to user history
            if behavior_idx in similarity_results_history:
                sim_map = similarity_results_history[behavior_idx]
                # Sort negative_samples by similarity value (default 0 if not found)
                # We want LEAST similar first? 
                # "Contrastive News Recommendation ... based on Curriculum Learning" usually implies starting with easy negatives 
                # (least similar, clearly distinguishable) and moving to hard negatives (most similar, confusing).
                # If we sort by similarity ASC (low to high), the first elements are "easy" negatives.
                # So taking `negative_samples[:limit]` takes the easiest negatives.
                
                # Check original code logic:
                # similarities.sort(key=lambda x: x[1]) -> Ascending. Low similarity first.
                # least_similar_non_click_impressions = ...
                # Then it slices this list.
                
                negative_samples_sorted = sorted(negative_samples, key=lambda nid: sim_map.get(nid, 0.0))
                negative_samples = negative_samples_sorted
            
            # Curriculum pruning (only if many negatives? CNRCL used > 46. 
            # EBNeRD demo has ~5-20 usually. Large might have more.)
            # If we strictly follow CNRCL code:
            # if news_num > self.negative_sample_num * 2:
            
            if news_num > self.negative_sample_num * 2:
                 index = news_num // 2
                 if epoch <= 3:
                     limit = index
                 elif 3 < epoch <= 6:
                     limit = index + news_num * 1 // 8
                 elif 6 < epoch <= 9:
                     limit = index + news_num * 2 // 8
                 elif 9 < epoch <= 12:
                     limit = index + news_num * 3 // 8
                 else:
                     limit = news_num
                 
                 # limit check
                 limit = int(limit)
                 if limit < self.negative_sample_num:
                     limit = self.negative_sample_num
                 
                 negative_samples = negative_samples[:limit]
            
            # Now sample from the (potentially pruned and sorted) list
            curr_news_num = len(negative_samples)
            
            if curr_news_num <= self.negative_sample_num:
                for j in range(self.negative_sample_num):
                    self.train_samples[i][j + 1] = negative_samples[j % curr_news_num]
            else:
                # Random sample from the allowed set
                used_negative_samples = set()
                for j in range(self.negative_sample_num):
                    while True:
                        k = randint(0, curr_news_num)
                        if k not in used_negative_samples:
                            self.train_samples[i][j + 1] = negative_samples[k]
                            used_negative_samples.add(k)
                            break
            
            # Contrastive Samples (Similar News)
            # train_samples index: 1 + negative_sample_num + n
            
            click_news_id_str = self.ID_news_dict[train_behavior[3]]
            n = 0
            if click_news_id_str in news_similarities:
                similar_news_ids = news_similarities[click_news_id_str]
                for similar_news_id in similar_news_ids:
                    if n >= self.similarity_num:
                        break
                    if click_news_id_str != similar_news_id and similar_news_id in self.news_ID_dict:
                        self.train_samples[i][1 + self.negative_sample_num + n] = self.news_ID_dict[similar_news_id]
                        n += 1
            
            # If not enough similar news found (or not in dict), fill with clicked news itself (self-similarity)
            while n < self.similarity_num:
                miss_num += 1
                self.train_samples[i][1 + self.negative_sample_num + n] = train_behavior[3]
                n += 1

        end_time = time.time()
        print('%sEnd negative sampling, used time : %.3fs. Miss num: %d' % ('' if rank is None else ('rank ' + str(rank) + ' : '), end_time - start_time, miss_num))
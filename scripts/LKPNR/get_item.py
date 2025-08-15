import os

import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
from tqdm import tqdm

from generate_merged_file import generate_merged_news_file

# Generate the merged file if it doesn't exist
if not os.path.exists("./merged_large_small_item.tsv"):
    news = generate_merged_news_file(train_news_path="../../MIND-small/download/train/news.tsv",
                                     dev_news_path="../../MIND-small/download/dev/news.tsv",
                                     output_path="./merged_large_small_item.tsv")
else:
    # If file already exists, just read it
    news = pd.read_csv(
        "./merged_large_small_item.tsv", 
        sep="\t"
    )


def construct_news_text(x):
    return "news title : " + str(x["title"]) + "\n" + "news category : " + str(x["category"]) + "\n" + "news abstract : " + str(x["abstract"])


news["news_text"] = news.apply(lambda x: construct_news_text(x), axis=1)
itemid_list = news["itemID"].tolist()
itemtext_list = news["news_text"].tolist()

# Setup device and multi-GPU
if torch.cuda.is_available():
    device = torch.device("cuda")
    n_gpu = torch.cuda.device_count()
    print(f"Using {n_gpu} GPUs")
else:
    device = torch.device("cpu")
    print("Using CPU")

# Dependence： CharGLM2-6B pip install protobuf transformers==4.30.2
tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm2-6b", trust_remote_code=True)
model = AutoModel.from_pretrained("THUDM/chatglm2-6b", trust_remote_code=True).float()

# freeze
for param in model.parameters():
    param.requires_grad = False
model.config.output_hidden_states = True

# Enable multi-GPU
if torch.cuda.is_available() and torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)
model = model.to(device)

item_text_map = {}
for idx, item_text in tqdm(enumerate(itemtext_list)):
    itemid = itemid_list[idx]
    inputs = tokenizer.encode_plus(item_text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        hidden_states = outputs.hidden_states[-1].detach()
        
    # Handle multi-GPU case
    if isinstance(hidden_states, list):
        hidden_states = hidden_states[0]  # Take the first GPU's output
        
    # Move to CPU for numpy operations
    hidden_states = hidden_states.cpu().permute(1, 0, 2)
    
    # 按照第二个维度求和
    sum_array = np.mean(hidden_states.numpy(), axis=1)
    # 将维度变为(4096)
    reshaped_array = sum_array.reshape(4096)
    item_text_map[itemid] = reshaped_array

np.save("../../KGraph_LKPNR/item_emb.npy", item_text_map)
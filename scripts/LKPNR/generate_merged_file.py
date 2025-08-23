import pandas as pd
import os

def generate_merged_news_file(train_news_path="../../MIND-small/download/train/news.tsv",
                              dev_news_path = "../../MIND-small/download/dev/news.tsv",
                              output_path = "KGraph_LKPNR/merged_large_small_item.tsv"):
    
    # Check if the files exist
    if not os.path.exists(train_news_path):
        raise FileNotFoundError(f"Train news file not found: {train_news_path}")
    
    if not os.path.exists(dev_news_path):
        raise FileNotFoundError(f"Dev news file not found: {dev_news_path}")
    
    # Read the train and dev news files
    train_news = pd.read_csv(
        train_news_path,
        sep="\t",
        header=None,
        names=["itemID", "category", "subcategory", "title", "abstract", "url", "title_entities", "abstract_entities"]
    )
    
    dev_news = pd.read_csv(
        dev_news_path,
        sep="\t",
        header=None,
        names=["itemID", "category", "subcategory", "title", "abstract", "url", "title_entities", "abstract_entities"]
    )
    
    # Merge the datasets
    merged_news = pd.concat([train_news, dev_news], ignore_index=True)
    
    # Remove duplicates based on itemID
    merged_news = merged_news.drop_duplicates(subset=["itemID"], keep="first")
    
    # Save to TSV file
    merged_news.to_csv(output_path, sep="\t", index=False)
    print(f"Merged news file saved to {output_path}")
    print(f"Total news items: {len(merged_news)}")
    return merged_news

if __name__ == "__main__":
    generate_merged_news_file(train_news_path="../../MIND-large/download/train/news.tsv",
                              dev_news_path = "../../MIND-large/download/dev/news.tsv",
                              output_path = "./merged_large_small_item.tsv")
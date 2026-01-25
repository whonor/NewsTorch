import os
try:
    import pandas as pd
    import numpy as np
except ImportError:
    pd = None
    np = None

def analyze_mind_dataset(dataset_name, root_path):
    """
    Analyzes the MIND dataset to calculate statistics.
    """
    print(f"正在分析 {dataset_name} 数据集 (路径: {root_path})...")
    
    splits = ['train', 'dev', 'test']
    
    # 使用集合去重
    news_ids = set()
    user_ids = set()
    categories = set()
    
    # 计数器
    impression_count = 0
    total_title_length = 0
    total_abstract_length = 0
    
    # 检查数据集路径是否存在
    if not os.path.exists(root_path):
        print(f"错误: 找不到路径 {root_path}")
        print("-" * 30)
        return

    for split in splits:
        split_path = os.path.join(root_path, split)
        if not os.path.exists(split_path):
            continue
            
        print(f"  正在处理 {split} 集...")
        
        # 处理 news.tsv
        # 格式: News ID, Category, SubCategory, Title, Abstract, URL, ...
        news_file = os.path.join(split_path, 'news.tsv')
        if os.path.exists(news_file):
            with open(news_file, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 4:
                        nid = parts[0]
                        # 仅处理未见过的 News ID (因为 train/dev/test 可能有重叠的新闻)
                        if nid not in news_ids:
                            news_ids.add(nid)
                            categories.add(parts[1]) # Category 在第2列
                            
                            # 计算 Title 字数 (按空格分词)
                            title = parts[3]
                            total_title_length += len(title.split())
                            
                            # 计算 Abstract 字数
                            if len(parts) >= 5:
                                abstract = parts[4]
                                total_abstract_length += len(abstract.split())
        
        # 处理 behaviors.tsv
        # 格式: Impression ID, User ID, Time, History, Impressions
        behaviors_file = os.path.join(split_path, 'behaviors.tsv')
        if os.path.exists(behaviors_file):
            with open(behaviors_file, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 3:
                        impression_count += 1
                        user_ids.add(parts[1]) # User ID 在第2列

    # 计算平均值
    num_news = len(news_ids)
    avg_title_len = total_title_length / num_news if num_news > 0 else 0
    avg_abstract_len = total_abstract_length / num_news if num_news > 0 else 0
    
    print(f"\n--- {dataset_name} 统计结果 ---")
    print(f"News (新闻总数): {num_news}")
    print(f"User (用户总数): {len(user_ids)}")
    print(f"Impression (曝光总数): {impression_count}")
    print(f"Avg Title Words (标题平均字数): {avg_title_len:.2f}")
    print(f"Avg Abstract Words (摘要平均字数): {avg_abstract_len:.2f}")
    print(f"Category Count (类别数量): {len(categories)}")
    print("-" * 30 + "\n")

def analyze_ebnerd_dataset(dataset_name, root_path):
    """
    Analyzes the EB-NeRD dataset (parquet format).
    """
    if pd is None:
        print("错误: 需要安装 pandas 和 pyarrow/fastparquet 才能分析 EB-NeRD 数据集")
        return

    print(f"正在分析 {dataset_name} 数据集 (路径: {root_path})...")
    
    # 1. Find Articles/News file
    # Possible locations based on download_ebnerd.py and standard structure
    possible_article_files = [
        os.path.join(root_path, "articles.parquet"),
        os.path.join(root_path, "download", dataset_name, "articles.parquet"),
        os.path.join(root_path, "news.parquet"), # Processed
        os.path.join(root_path, "download", dataset_name, "news.parquet")
    ]
    
    df_news = None
    for p in possible_article_files:
        if os.path.exists(p):
            print(f"  读取新闻数据: {p}")
            df_news = pd.read_parquet(p)
            break
            
    if df_news is None:
        print(f"错误: 在 {root_path} 找不到 articles.parquet 或 news.parquet")
        return

    # 2. Calculate News Stats
    num_news = len(df_news)
    
    # Title & Abstract Length
    # EB-NeRD raw: title, subtitle. Processed: title, abstract
    title_col = 'title' if 'title' in df_news.columns else None
    abs_col = 'subtitle' if 'subtitle' in df_news.columns else ('abstract' if 'abstract' in df_news.columns else None)
    
    avg_title_len = 0
    if title_col:
        avg_title_len = df_news[title_col].fillna('').astype(str).apply(lambda x: len(x.split())).mean()
        
    avg_abstract_len = 0
    if abs_col:
        avg_abstract_len = df_news[abs_col].fillna('').astype(str).apply(lambda x: len(x.split())).mean()
        
    # Categories
    # EB-NeRD raw: category_str. Processed: category (might be ID or str)
    cat_col = 'category_str' if 'category_str' in df_news.columns else ('category' if 'category' in df_news.columns else None)
    num_cats = df_news[cat_col].nunique() if cat_col else 0

    # 3. Analyze Behaviors (Train/Dev/Test)
    user_ids = set()
    impression_count = 0
    
    # Look into subdirectories
    subdirs = ['train', 'dev', 'validation', 'test']
    search_roots = [root_path, os.path.join(root_path, 'download', dataset_name)]
    
    for r in search_roots:
        if not os.path.exists(r): continue
        for subdir in subdirs:
            p = os.path.join(r, subdir)
            if os.path.exists(p):
                # Check for behaviors.parquet or behaviors_.parquet
                for fname in ['behaviors.parquet', 'behaviors_.parquet']:
                    fpath = os.path.join(p, fname)
                    if os.path.exists(fpath):
                        print(f"  读取行为数据: {fpath}")
                        try:
                            # Try 'user_id' first (standard EB-NeRD), then 'uid' (processed)
                            col_name = 'user_id'
                            try:
                                df_beh = pd.read_parquet(fpath, columns=[col_name])
                            except Exception:
                                col_name = 'uid'
                                df_beh = pd.read_parquet(fpath, columns=[col_name])

                            impression_count += len(df_beh)
                            user_ids.update(df_beh[col_name].unique())
                            break # Avoid double counting if both exist
                        except Exception as e:
                            print(f"    读取失败 {fname}: {e}")

    print(f"\n--- {dataset_name} 统计结果 ---")
    print(f"News (新闻总数): {num_news}")
    print(f"User (用户总数): {len(user_ids)}")
    print(f"Impression (曝光总数): {impression_count}")
    print(f"Avg Title Words (标题平均字数): {avg_title_len:.2f}")
    print(f"Avg Abstract Words (摘要平均字数): {avg_abstract_len:.2f}")
    print(f"Category Count (类别数量): {num_cats}")
    print("-" * 30 + "\n")

if __name__ == "__main__":
    # MIND Datasets
    if os.path.exists("MIND-small"):
        analyze_mind_dataset("MIND-small", "MIND-small")
    if os.path.exists("MIND-large"):
        analyze_mind_dataset("MIND-large", "MIND-large")
        
    # EB-NeRD Datasets
    # Folder names usually match the dataset name (e.g., ebnerd_demo, ebnerd_small)
    if os.path.exists("ebnerd_demo"):
        analyze_ebnerd_dataset("ebnerd_demo", "ebnerd_demo")
    if os.path.exists("ebnerd_small"):
        analyze_ebnerd_dataset("ebnerd_small", "ebnerd_small")
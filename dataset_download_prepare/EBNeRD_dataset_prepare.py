#!/usr/bin/env python3
import os
import json
import shutil
import random
import numpy as np
import collections

import pandas as pd

# setup random seed for reproducibility
random.seed(0)
np.random.seed(0)

root = "../"
# root
ebnerd_demo_dataset_root = root + '/ebnerd_demo'
ebnerd_small_dataset_root = root + '/ebnerd_small'
ebnerd_large_dataset_root = root + '/ebnerd_large'

def confirm_overwrite(path: str) -> bool:
    if os.path.exists(path):
        choice = input(f"already existed: {path}，recovered？(y/N): ").strip().lower()
        return choice == 'y'
    return True


def split_training_behaviors(size, train_ratio=0.9):
    if size == 'ebnerd_demo':
        behavior_file = os.path.join(ebnerd_demo_dataset_root, 'download', size, 'train', 'behaviors_.parquet')
    elif size == 'ebnerd_small':
        behavior_file = os.path.join(ebnerd_small_dataset_root, 'download', size, 'train', 'behaviors_.parquet')
    elif size == 'ebnerd_large':
        behavior_file = os.path.join(ebnerd_large_dataset_root, 'download', size, 'train', 'behaviors_.parquet')

    if not os.path.exists(behavior_file):
        raise FileNotFoundError(f"behavior file does not exist: {behavior_file}")

    #  pandas  parquet
    df = pd.read_parquet(behavior_file)

    #
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    #
    train_num = int(len(df) * train_ratio)
    train_df = df.iloc[:train_num]
    dev_df = df.iloc[train_num:]

    return train_df, dev_df


def preprocess_ebnerd_demo(size):
    global root, mode_
    train_df, dev_df = split_training_behaviors(size=size)

    if size == 'ebnerd_demo':
        root = ebnerd_demo_dataset_root
    elif size == 'ebnerd_small':
        root = ebnerd_small_dataset_root
    elif size == 'ebnerd_large':
        root = ebnerd_large_dataset_root

    # train/dev sets
    for mode, df in [('train', train_df), ('dev', dev_df)]:
        out_dir = os.path.join(root, mode)
        if os.path.exists(out_dir):
            if not confirm_overwrite(out_dir):
                print(f"Jump {mode} dataset preparation")
                continue
            shutil.rmtree(out_dir)
        os.makedirs(out_dir)

        # save behaviors.parquet
        df.to_parquet(os.path.join(out_dir, 'behaviors.parquet'), index=False)

        # copy news.parquet, history.parquet
        src_news = os.path.join(root, 'download', size, 'news.parquet')
        dst_news = os.path.join(out_dir, 'news.parquet')
        if confirm_overwrite(dst_news):
            if not os.path.exists(src_news):
                raise FileNotFoundError(f"news file does not exist: {src_news}")
            shutil.copyfile(src_news, dst_news)

        if mode == 'dev':
            mode_ = 'validation'
            src_news_ = os.path.join(root, 'download', size, mode_, 'history.parquet')
        else:
            src_news_ = os.path.join(root, 'download', size, mode, 'history.parquet')
        dst_news_ = os.path.join(out_dir, 'history.parquet')
        if confirm_overwrite(dst_news):
            if not os.path.exists(src_news):
                raise FileNotFoundError(f"news file does not exist: {src_news}")
            shutil.copyfile(src_news_, dst_news_)



    # test set
    test_dir = os.path.join(root, 'test')
    if os.path.exists(test_dir):
        if not confirm_overwrite(test_dir):
            print("Jump test dataset preparation")
            return
        shutil.rmtree(test_dir)
    os.makedirs(test_dir)

    for fname in ('behaviors_.parquet', 'news.parquet', 'history.parquet'):
        if fname == 'behaviors_.parquet':
            src = os.path.join(root, 'download', size, 'validation', fname)
            dst = os.path.join(test_dir, 'behaviors.parquet')
        elif fname == 'news.parquet':
            src = os.path.join(root, 'download', size, fname)
            dst = os.path.join(test_dir, 'news.parquet')
        else:
            src = os.path.join(root, 'download', size, 'validation', fname)
            dst = os.path.join(test_dir, 'history.parquet')

        if confirm_overwrite(dst):
            if not os.path.exists(src):
                raise FileNotFoundError(f"File does not exist: {src}")
            shutil.copyfile(src, dst)




def main():
    print("Prepare ebnerd_small...")
    preprocess_ebnerd_demo(size="ebnerd_small")

    print("Prepare ebnerd_large...")
    preprocess_ebnerd_demo(size="ebnerd_large")

    print("All datasets are finished。")


if __name__ == '__main__':
    main()

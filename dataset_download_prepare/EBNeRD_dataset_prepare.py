#!/usr/bin/env python3
import os
import shutil
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parent.parent
DATASET_ROOTS = {
    'ebnerd_demo': REPO_ROOT / 'ebnerd_demo',
    'ebnerd_small': REPO_ROOT / 'ebnerd_small',
    'ebnerd_large': REPO_ROOT / 'ebnerd_large',
}


def confirm_overwrite(path: str) -> bool:
    if os.path.exists(path):
        choice = input(f"already existed: {path}, recovered? (y/N): ").strip().lower()
        return choice == 'y'
    return True


def dataset_root(size):
    if size not in DATASET_ROOTS:
        raise ValueError(f"Unsupported EB-NeRD dataset size: {size}")
    return DATASET_ROOTS[size]


def behavior_path(root, size, split):
    return root / 'download' / size / split / 'behaviors_.parquet'


def split_validation_behaviors(size, dev_ratio=0.5):
    root = dataset_root(size)
    behavior_file = behavior_path(root, size, 'validation')
    if not os.path.exists(behavior_file):
        raise FileNotFoundError(f"behavior file does not exist: {behavior_file}")

    df = pd.read_parquet(behavior_file)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    dev_num = int(len(df) * dev_ratio)
    dev_df = df.iloc[:dev_num]
    test_df = df.iloc[dev_num:]

    return dev_df, test_df


def prepare_split(root, size, mode, source_split, behavior_df=None):
    out_dir = root / mode
    if os.path.exists(out_dir):
        if not confirm_overwrite(str(out_dir)):
            print(f"Jump {mode} dataset preparation")
            return
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)

    dst_behavior = out_dir / 'behaviors.parquet'
    if behavior_df is None:
        src_behavior = behavior_path(root, size, source_split)
        if not os.path.exists(src_behavior):
            raise FileNotFoundError(f"behavior file does not exist: {src_behavior}")
        shutil.copyfile(src_behavior, dst_behavior)
    else:
        behavior_df.to_parquet(dst_behavior, index=False)

    src_news = root / 'download' / size / 'news.parquet'
    dst_news = out_dir / 'news.parquet'
    if not os.path.exists(src_news):
        raise FileNotFoundError(f"news file does not exist: {src_news}")
    shutil.copyfile(src_news, dst_news)

    src_history = root / 'download' / size / source_split / 'history.parquet'
    dst_history = out_dir / 'history.parquet'
    if not os.path.exists(src_history):
        raise FileNotFoundError(f"history file does not exist: {src_history}")
    shutil.copyfile(src_history, dst_history)


def preprocess_ebnerd(size):
    root = dataset_root(size)
    dev_df, test_df = split_validation_behaviors(size=size)

    prepare_split(root=root, size=size, mode='train', source_split='train')
    prepare_split(root=root, size=size, mode='dev', source_split='validation', behavior_df=dev_df)
    prepare_split(root=root, size=size, mode='test', source_split='validation', behavior_df=test_df)

    train_count = len(pd.read_parquet(behavior_path(root, size, 'train')))
    print(f"{size}: train={train_count}, dev={len(dev_df)}, test={len(test_df)}")


def preprocess_ebnerd_demo(size):
    preprocess_ebnerd(size=size)


def main():
    print("Prepare ebnerd_demo...")
    preprocess_ebnerd(size="ebnerd_demo")

    print("Prepare ebnerd_small...")
    preprocess_ebnerd(size="ebnerd_small")

    # print("Prepare ebnerd_large...")
    # preprocess_ebnerd(size="ebnerd_large")

    print("All datasets are finished.")


if __name__ == '__main__':
    main()

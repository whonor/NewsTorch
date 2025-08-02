#!/usr/bin/env python3
import os
import json
import shutil
import random
import numpy as np
import collections

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


def split_training_behaviors(size='ebnerd_demo'):
    MIND_small_train_ratio = 0.9
    behavior_file = os.path.join(ebnerd_demo_dataset_root, 'download', size, 'train', 'behaviors_.tsv')
    if not os.path.exists(behavior_file):
        raise FileNotFoundError(f"behavior file no exist: {behavior_file}")

    with open(behavior_file, 'r', encoding='utf-8') as f:
        behavior_lines = [line for line in f if line.strip()]

    random.shuffle(behavior_lines)
    total = len(behavior_lines)
    train_num = int(total * MIND_small_train_ratio)
    indices = list(range(total))
    random.shuffle(indices)
    train_idx = set(indices[:train_num])

    train_behavior_lines = []
    dev_behavior_lines = []
    for i, line in enumerate(behavior_lines):
        if i in train_idx:
            train_behavior_lines.append(line)
        else:
            dev_behavior_lines.append(line)

    return train_behavior_lines, dev_behavior_lines


def preprocess_ebnerd_demo(size='ebnerd_demo'):
    train_behavior_lines, dev_behavior_lines = split_training_behaviors(size=size)

    # train/dev sets
    for mode, lines, src_news_split in [
        ('train', train_behavior_lines, 'train'),
        ('dev', dev_behavior_lines, 'train')
    ]:
        out_dir = os.path.join(ebnerd_demo_dataset_root, mode)
        if os.path.exists(out_dir):
            if not confirm_overwrite(out_dir):
                print(f"jump {mode} dataset prepare")
                continue
            shutil.rmtree(out_dir)
        os.makedirs(out_dir)

        # 写 behaviors
        with open(os.path.join(out_dir, 'behaviors.tsv'), 'w', encoding='utf-8') as f:
            f.writelines(lines)

        # 拷贝 news
        src_news = os.path.join(ebnerd_demo_dataset_root, 'download', size, 'news_.tsv')
        dst_news = os.path.join(out_dir, 'news.tsv')
        if confirm_overwrite(dst_news):
            if not os.path.exists(src_news):
                raise FileNotFoundError(f"news file no exist: {src_news}")
            shutil.copyfile(src_news, dst_news)

    # test set
    test_dir = os.path.join(ebnerd_demo_dataset_root, 'test')
    if os.path.exists(test_dir):
        if not confirm_overwrite(test_dir):
            print("jump test dataset prepare")
            return
        shutil.rmtree(test_dir)
    os.makedirs(test_dir)

    for fname in ('behaviors_.tsv', 'news_.tsv'):
        if fname == 'behaviors_.tsv':
            src = os.path.join(ebnerd_demo_dataset_root, 'download', size, 'validation', fname)
            dst = os.path.join(test_dir, 'behaviors.tsv')
            if confirm_overwrite(dst):
                if not os.path.exists(src):
                    raise FileNotFoundError(f"no exist: {src}")
                shutil.copyfile(src, dst)
        else:
            src = os.path.join(ebnerd_demo_dataset_root, 'download', size, fname)
            dst = os.path.join(test_dir, 'news.tsv')
            if confirm_overwrite(dst):
                if not os.path.exists(src):
                    raise FileNotFoundError(f"no exist: {src}")
                shutil.copyfile(src, dst)




def main():
    print("Prepare ebnerd_demo...")
    preprocess_ebnerd_demo(size="ebnerd_demo")

    print("All datasets are finished。")


if __name__ == '__main__':
    main()

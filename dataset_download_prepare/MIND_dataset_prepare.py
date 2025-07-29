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
MIND_small_dataset_root = root + '/MIND-small'
MIND_large_dataset_root = root + '/MIND-large'
MIND_200k_dataset_root = root + '/MIND-200k'

def confirm_overwrite(path: str) -> bool:
    if os.path.exists(path):
        choice = input(f"already existed: {path}，recovered？(y/N): ").strip().lower()
        return choice == 'y'
    return True


def split_training_behaviors():
    MIND_small_train_ratio = 0.9
    behavior_file = os.path.join(MIND_small_dataset_root, 'download', 'train', 'behaviors.tsv')
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


def preprocess_MIND_small():
    train_behavior_lines, dev_behavior_lines = split_training_behaviors()

    # train/dev sets
    for mode, lines, src_news_split in [
        ('train', train_behavior_lines, 'train'),
        ('dev', dev_behavior_lines, 'train')
    ]:
        out_dir = os.path.join(MIND_small_dataset_root, mode)
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
        src_news = os.path.join(MIND_small_dataset_root, 'download', src_news_split, 'news.tsv')
        dst_news = os.path.join(out_dir, 'news.tsv')
        if confirm_overwrite(dst_news):
            if not os.path.exists(src_news):
                raise FileNotFoundError(f"news file no exist: {src_news}")
            shutil.copyfile(src_news, dst_news)

    # test set
    test_dir = os.path.join(MIND_small_dataset_root, 'test')
    if os.path.exists(test_dir):
        if not confirm_overwrite(test_dir):
            print("jump test dataset prepare")
            return
        shutil.rmtree(test_dir)
    os.makedirs(test_dir)

    for fname in ('behaviors.tsv', 'news.tsv'):
        src = os.path.join(MIND_small_dataset_root, 'download', 'dev', fname)
        dst = os.path.join(test_dir, fname)
        if confirm_overwrite(dst):
            if not os.path.exists(src):
                raise FileNotFoundError(f"no exist: {src}")
            shutil.copyfile(src, dst)


def sampling_MIND_dataset(sample_num=200000):
    # 采样用户并生成行为
    user_set = set()
    for split in ('train', 'dev'):
        path = os.path.join(MIND_200k_dataset_root, 'download', split, 'behaviors.tsv')
        if not os.path.exists(path):
            raise FileNotFoundError(f"behaviors.tsv no exist: {path}")
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                _, user_ID, *_ = line.strip().split('\t')
                user_set.add(user_ID)

    if sample_num > len(user_set):
        raise ValueError(f"The number of sample {sample_num} exceed total {len(user_set)}")

    sample_users = set(random.sample(list(user_set), sample_num))
    with open(os.path.join(MIND_200k_dataset_root, 'sample_users.json'), 'w', encoding='utf-8') as f:
        json.dump(list(sample_users), f)

    # 写 behaviors
    for split, out_dir in [('train', 'train'), ('dev', 'dev'), ('dev', 'test')]:
        src = os.path.join(MIND_200k_dataset_root, 'download', split, 'behaviors.tsv')
        dst_dir = os.path.join(MIND_200k_dataset_root, out_dir)
        os.makedirs(dst_dir, exist_ok=True)
        dst_file = os.path.join(dst_dir, 'behaviors.tsv')
        if confirm_overwrite(dst_file):
            with open(src, 'r', encoding='utf-8') as f_in, open(dst_file, 'w', encoding='utf-8') as f_out:
                for i, line in enumerate(f_in):
                    _, user_ID, *_ = line.strip().split('\t')
                    if user_ID in sample_users:
                        if split == 'train' or (split == 'dev' and i % 2 == 0 and out_dir == 'dev') or (split == 'dev' and i % 2 == 1 and out_dir == 'test'):
                            f_out.write(line)

    # 写 news
    for mode in ('train', 'dev', 'test'):
        news_ids = set()
        beh_file = os.path.join(MIND_200k_dataset_root, mode, 'behaviors.tsv')
        with open(beh_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                hist, imps = parts[3], parts[4]
                news_ids.update(hist.split())
                news_ids.update([imp[:-2] for imp in imps.split()])

        src_news = os.path.join(MIND_200k_dataset_root, 'download', 'train' if mode=='train' else 'dev', 'news.tsv')
        dst_news = os.path.join(MIND_200k_dataset_root, mode, 'news.tsv')
        if confirm_overwrite(dst_news):
            with open(src_news, 'r', encoding='utf-8') as f_in, open(dst_news, 'w', encoding='utf-8') as f_out:
                for line in f_in:
                    nid = line.split('\t')[0]
                    if nid in news_ids:
                        f_out.write(line)


def generate_knowledge_entity_embedding(data_mode):
    assert data_mode in ['200k', 'small', 'large']
    # 1. copy entity embedding file
    shutil.copyfile(root + '/MIND-%s/download/train/entity_embedding.vec' % data_mode,
                    root + '/MIND-%s/train/entity_embedding.vec' % data_mode)
    shutil.copyfile(root + '/MIND-%s/download/dev/entity_embedding.vec' % data_mode,
                    root + '/MIND-%s/dev/entity_embedding.vec' % data_mode)
    if data_mode in ['200k', 'small']:
        shutil.copyfile(root + '/MIND-%s/download/dev/entity_embedding.vec' % data_mode,
                        root + '/MIND-%s/test/entity_embedding.vec' % data_mode)
    else:
        shutil.copyfile(root + '/MIND-large/download/test/entity_embedding.vec', root + '/MIND-large/test/entity_embedding.vec')


    entity_embeddings = {}
    entity_embedding_files = [root + '/MIND-%s/%s/entity_embedding.vec' % (data_mode, mode) for mode in
                              ['train', 'dev', 'test']]
    for entity_embedding_file in entity_embedding_files:
        with open(entity_embedding_file, 'r', encoding='utf-8') as f:
            for line in f:
                if len(line.strip()) > 0:
                    terms = line.strip().split('\t')
                    assert len(terms) == 101
                    entity_embeddings[terms[0]] = list(map(float, terms[1:]))
    entity_embedding_relation = collections.defaultdict(set)
    with open(root + '/MIND-%s/download/wikidata-graph/wikidata-graph/wikidata-graph.tsv' % data_mode, 'r',
              encoding='utf-8') as wikidata_graph_f:
        for line in wikidata_graph_f:
            if len(line.strip()) > 0:
                terms = line.strip().split('\t')
                entity_embedding_relation[terms[0]].add(terms[2])
                entity_embedding_relation[terms[2]].add(terms[0])
    context_embeddings = {}
    for entity in entity_embeddings:
        entity_embedding = entity_embeddings[entity]
        context_embedding = [entity_embedding[i] for i in range(100)]
        cnt = 1
        for _entity in entity_embedding_relation[entity]:
            if _entity in entity_embeddings:
                embedding = entity_embeddings[_entity]
                for i in range(100):
                    context_embedding[i] += embedding[i]
                cnt += 1
        for i in range(100):
            context_embedding[i] /= cnt
        context_embeddings[entity] = context_embedding
    for mode in ['train', 'dev', 'test']:
        with open(root + '/MIND-%s/%s/entity_embedding.vec' % (data_mode, mode), 'r',
                  encoding='utf-8') as entity_embedding_f:
            with open(root + '/MIND-%s/%s/context_embedding.vec' % (data_mode, mode), 'w',
                      encoding='utf-8') as context_embedding_f:
                for line in entity_embedding_f:
                    if len(line.strip()) > 0:
                        entity = line.split('\t')[0]
                        context_embedding_f.write(
                            entity + '\t' + '\t'.join(list(map(str, context_embeddings[entity]))) + '\n')


def prepare_MIND_small():
    preprocess_MIND_small()
    generate_knowledge_entity_embedding('small')


def prepare_MIND_large():
    generate_knowledge_entity_embedding('large')


def prepare_MIND_200k():
    sampling_MIND_dataset(sample_num=200000)
    generate_knowledge_entity_embedding('200k')

def main():
    print("Prepare MIND-small...")
    prepare_MIND_small()
    # print("准备 MIND-200k...")
    # prepare_MIND_200k()
    # print("准备 MIND-large...")
    # prepare_MIND_large()
    print("All datasets are finished。")


if __name__ == '__main__':
    main()

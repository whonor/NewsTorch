#!/usr/bin/env python3
"""Generate dataset-indexed LKPNR article vectors with a local LLM.

This generalizes the official LKPNR ``LLM/get_item.py`` pipeline: title,
category, and abstract are encoded by a frozen transformer, and the last hidden
layer is mean-pooled into one vector per article.  Dense row indices match the
NewsTorch corpus mapping so training can retrieve vectors without ID joins.

Only ``safetensors`` model weights are loaded.  This keeps preprocessing usable
with PyTorch versions for which recent Transformers releases intentionally
block the vulnerable ``torch.load`` path used by legacy ``.bin`` checkpoints.
"""

import argparse
import json
import os
import re

import numpy as np

try:
    from scripts.generate_s2lenr_qwen_news import (
        load_adressa_news,
        load_ebnerd_news,
        load_mind_news,
        patch_torch_pytree_compat,
    )
except ModuleNotFoundError:
    from generate_s2lenr_qwen_news import (
        load_adressa_news,
        load_ebnerd_news,
        load_mind_news,
        patch_torch_pytree_compat,
    )


def _dataset_name(value):
    return "Adressa" if value.lower() == "adressa" else value


def _cache_dir(dataset_name):
    return "cache" if dataset_name == "MIND" else os.path.join(
        "cache", dataset_name.lower()
    )


def _mapping_path(dataset_name, dataset_size):
    return os.path.join(
        _cache_dir(dataset_name), f"news_ID-{dataset_size}.json"
    )


def _default_output_path(dataset_name, dataset_size):
    return os.path.join(
        _cache_dir(dataset_name), f"lkpnr_item_embedding-{dataset_size}.npy"
    )


def _load_news(dataset_name, roots):
    if dataset_name == "ebnerd":
        return load_ebnerd_news(roots)
    if dataset_name == "Adressa":
        return load_adressa_news(roots)
    if dataset_name == "MIND":
        return load_mind_news(roots)
    raise ValueError("LKPNR embeddings support MIND, ebnerd, and Adressa")


def _load_or_create_mapping(dataset_name, dataset_size, news):
    path = _mapping_path(dataset_name, dataset_size)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as source:
            mapping = {str(key): int(value) for key, value in json.load(source).items()}
    else:
        if dataset_name == "Adressa":
            news_ids = sorted(news)
        else:
            news_ids = list(news)
        mapping = {"<PAD>": 0}
        for news_id in news_ids:
            if news_id not in mapping:
                mapping[news_id] = len(mapping)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as output:
            json.dump(mapping, output, ensure_ascii=False)

    missing = sorted(set(news) - set(mapping))
    if missing:
        raise ValueError(
            f"{path} is missing {len(missing)} dataset articles; rebuild the corpus cache"
        )
    return mapping, path


def _article_text(article):
    return (
        f"news title : {article.get('title', '')}\n"
        f"news category : {article.get('category', '')}\n"
        f"news abstract : {article.get('abstract', '')}"
    )


def _safe_model_name(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "model"


def _torch_dtype(torch, value):
    if value == "auto":
        return "auto"
    try:
        return getattr(torch, value)
    except AttributeError as error:
        raise ValueError(f"Unsupported torch dtype: {value}") from error


def _validate_local_checkpoint_weights(model_name_or_path, torch_version):
    """Reject a local legacy-only checkpoint before Transformers calls torch.load."""
    path = os.path.abspath(os.path.expanduser(model_name_or_path))
    if not os.path.exists(path):
        return

    if os.path.isdir(path):
        filenames = os.listdir(path)
    else:
        filenames = [os.path.basename(path)]
    has_safetensors = any(name.endswith(".safetensors") for name in filenames)
    has_pytorch_bin = any(name.endswith(".bin") for name in filenames)
    if has_pytorch_bin and not has_safetensors:
        raise RuntimeError(
            f"The local checkpoint {model_name_or_path!r} contains only PyTorch "
            f".bin weights. Transformers cannot safely load those weights with "
            f"PyTorch {torch_version}. Use a checkpoint containing .safetensors "
            "weights (for example Qwen/Qwen3-0.6B), or convert ChatGLM2 to "
            "safetensors in a separate environment with PyTorch >= 2.6. The "
            "current environment does not need to be changed."
        )


def generate(args):
    try:
        import torch

        patch_torch_pytree_compat(torch)
        from transformers import AutoModel, AutoTokenizer
    except ImportError as error:
        raise ImportError(
            "torch and transformers are required to generate LKPNR embeddings"
        ) from error

    _validate_local_checkpoint_weights(args.model_name_or_path, torch.__version__)

    dataset_name = _dataset_name(args.dataset_name)
    roots = [os.path.join(args.DATASET_ROOT, split) for split in ("train", "dev", "test")]
    news = _load_news(dataset_name, roots)
    if not news:
        raise FileNotFoundError(
            f"No {dataset_name} news records found below {args.DATASET_ROOT}"
        )
    mapping, mapping_path = _load_or_create_mapping(
        dataset_name, args.dataset_size, news
    )
    ordered = sorted(
        ((index, news_id) for news_id, index in mapping.items() if news_id in news),
        key=lambda item: item[0],
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    try:
        model = AutoModel.from_pretrained(
            args.model_name_or_path,
            trust_remote_code=args.trust_remote_code,
            local_files_only=args.local_files_only,
            torch_dtype=_torch_dtype(torch, args.torch_dtype),
            use_safetensors=True,
        )
    except (OSError, ValueError) as error:
        message = str(error).lower()
        if (
            "safetensor" in message
            or "torch.load" in message
            or "torch 2.6" in message
        ):
            raise RuntimeError(
                f"Could not load safe weights for {args.model_name_or_path!r}. "
                "This generator deliberately loads only .safetensors files so "
                "it works without upgrading the current PyTorch environment. "
                "Use Qwen/Qwen3-0.6B (or a local safetensors checkpoint). For "
                "the official ChatGLM2 encoder, first convert its .bin shards "
                "to safetensors in a separate PyTorch >= 2.6 environment."
            ) from error
        raise
    device = torch.device(args.device)
    model.to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    output_path = args.output_path or _default_output_path(
        dataset_name, args.dataset_size
    )
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    temporary_path = output_path + ".tmp.npy"
    embedding_file = None
    hidden_size = None

    for start in range(0, len(ordered), args.batch_size):
        batch = ordered[start : start + args.batch_size]
        texts = [_article_text(news[news_id]) for _, news_id in batch]
        inputs = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.max_length,
        ).to(device)
        with torch.no_grad():
            outputs = model(
                **inputs,
                output_hidden_states=True,
                return_dict=True,
            )
        hidden = outputs.hidden_states[-1]
        if hidden.size(0) != len(batch) and hidden.size(1) == len(batch):
            hidden = hidden.transpose(0, 1)
        attention_mask = inputs["attention_mask"].to(hidden.dtype).unsqueeze(-1)
        pooled = (hidden * attention_mask).sum(dim=1) / attention_mask.sum(
            dim=1
        ).clamp_min(1.0)
        pooled = pooled.float().cpu().numpy()

        if embedding_file is None:
            hidden_size = int(pooled.shape[1])
            embedding_file = np.lib.format.open_memmap(
                temporary_path,
                mode="w+",
                dtype=np.float32,
                shape=(max(mapping.values()) + 1, hidden_size),
            )
            embedding_file[:] = 0
        for row, (news_index, _) in enumerate(batch):
            embedding_file[news_index] = pooled[row]
        embedding_file.flush()
        print(f"Encoded {min(start + len(batch), len(ordered))}/{len(ordered)} articles")

    del embedding_file
    os.replace(temporary_path, output_path)
    manifest = {
        "dataset_name": dataset_name,
        "dataset_size": args.dataset_size,
        "dataset_root": os.path.abspath(args.DATASET_ROOT),
        "model_name_or_path": args.model_name_or_path,
        "hidden_size": hidden_size,
        "news_num": max(mapping.values()) + 1,
        "mapping_path": mapping_path,
        "pooling": "attention-mask mean of the last hidden layer",
    }
    with open(output_path + ".json", "w", encoding="utf-8") as output:
        json.dump(manifest, output, ensure_ascii=False, indent=2)
    print(f"Saved {output_path} with shape ({manifest['news_num']}, {hidden_size})")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate official-style LLM article embeddings for LKPNR"
    )
    parser.add_argument("--dataset_name", default="ebnerd")
    parser.add_argument("--DATASET_ROOT", default="ebnerd_small")
    parser.add_argument("--dataset_size", default="small")
    parser.add_argument("--model_name_or_path", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--output_path", default="")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument(
        "--device", default="cuda" if __import__("torch").cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--torch_dtype",
        choices=("auto", "float32", "float16", "bfloat16"),
        default="auto",
    )
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument(
        "--trust_remote_code", dest="trust_remote_code", action="store_true"
    )
    parser.add_argument(
        "--no_trust_remote_code", dest="trust_remote_code", action="store_false"
    )
    parser.set_defaults(trust_remote_code=True)
    return parser.parse_args()


if __name__ == "__main__":
    generate(parse_args())

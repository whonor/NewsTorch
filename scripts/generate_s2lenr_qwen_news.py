#!/usr/bin/env python3
import argparse
import json
import os
import re
import time
from typing import Dict, Iterable, List

import numpy as np


PROMPT_TEMPLATE = """Assuming that you are a news article generator carrying divergent thinking. Given a specific user and his few clicked news history in list format history, please first summarize keywords of interest to the user and infer his potential reading preference. Then help me generate at most {num_news} news articles step by step. The generated news articles must be reasonable, diverse, fluent and have low similarities to historical contents.

history:
{history}

Please output with the following FORMAT:
<start output>
Title : <text>; Abstract : <text>; Topic : <words>; Keywords : <words>; Evidence : <text>
<end output>
"""


def clean_text(value) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() == "nan" else text


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "default"


def as_list(value) -> List[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, np.ndarray)):
        return [clean_text(item) for item in value if clean_text(item)]
    text = clean_text(value)
    if not text:
        return []
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1]
    if "," in text:
        return [item.strip().strip("'\"") for item in text.split(",") if item.strip().strip("'\"")]
    return [item.strip() for item in text.split() if item.strip()]


def load_mind_news(roots: Iterable[str]) -> Dict[str, Dict[str, str]]:
    news = {}
    for root in roots:
        path = os.path.join(root, "news.tsv")
        if not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8") as news_f:
            for line in news_f:
                fields = line.rstrip("\n").split("\t")
                if len(fields) < 5:
                    continue
                news_id, category, subcategory, title, abstract = fields[:5]
                news.setdefault(
                    news_id,
                    {
                        "title": clean_text(title),
                        "abstract": clean_text(abstract),
                        "category": clean_text(category),
                        "subcategory": clean_text(subcategory),
                    },
                )
    return news


def load_ebnerd_news(roots: Iterable[str]) -> Dict[str, Dict[str, str]]:
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError("pandas is required to read EB-NeRD parquet files.") from exc

    news = {}
    for root in roots:
        path = os.path.join(root, "news.parquet")
        if not os.path.exists(path):
            continue
        df = pd.read_parquet(path)
        for _, row in df.iterrows():
            news_id = clean_text(row.get("nid"))
            if not news_id:
                continue
            news.setdefault(
                news_id,
                {
                    "title": clean_text(row.get("title")),
                    "abstract": clean_text(row.get("abstract")),
                    "category": clean_text(row.get("category")),
                    "subcategory": clean_text(row.get("subcategory")),
                },
            )
    return news


def collect_mind_users(train_root: str, news: Dict[str, Dict[str, str]], max_history: int):
    users = {}
    path = os.path.join(train_root, "behaviors.tsv")
    with open(path, "r", encoding="utf-8") as behavior_f:
        for line in behavior_f:
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 4:
                continue
            _, user_id, _, history = fields[:4]
            history_ids = [item for item in history.split(" ") if item]
            if not history_ids:
                continue
            if user_id not in users or len(history_ids) > len(users[user_id]):
                users[user_id] = [item for item in history_ids[-max_history:] if item in news]
    return users


def collect_ebnerd_users(train_root: str, news: Dict[str, Dict[str, str]], max_history: int):
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError("pandas is required to read EB-NeRD parquet files.") from exc

    users = {}
    path = os.path.join(train_root, "behaviors.parquet")
    df = pd.read_parquet(path)
    for _, row in df.iterrows():
        user_id = clean_text(row.get("uid"))
        history_ids = [item for item in as_list(row.get("history")) if item in news]
        if not user_id or not history_ids:
            continue
        if user_id not in users or len(history_ids) > len(users[user_id]):
            users[user_id] = history_ids[-max_history:]
    return users


def build_history_text(history_ids: List[str], news: Dict[str, Dict[str, str]], prompt_history_num: int) -> str:
    items = []
    for news_id in history_ids[-prompt_history_num:]:
        item = news.get(news_id, {})
        title = item.get("title", "")
        abstract = item.get("abstract", "")
        category = item.get("category", "")
        subcategory = item.get("subcategory", "")
        items.append(
            f"- Title: {title}; Abstract: {abstract}; Topic: {category}; Subtopic: {subcategory}"
        )
    return "\n".join(items)


def patch_torch_pytree_compat(torch_module):
    """Bridge newer transformers to torch 2.1's private pytree registration name."""
    pytree = getattr(getattr(torch_module, "utils", None), "_pytree", None)
    if pytree is None or hasattr(pytree, "register_pytree_node"):
        return
    if not hasattr(pytree, "_register_pytree_node"):
        return

    def register_pytree_node(node_type, flatten_fn, unflatten_fn, **kwargs):
        kwargs.pop("serialized_type_name", None)
        return pytree._register_pytree_node(node_type, flatten_fn, unflatten_fn, **kwargs)

    pytree.register_pytree_node = register_pytree_node


class LocalQwenGenerator:
    def __init__(self, args):
        try:
            import torch
            patch_torch_pytree_compat(torch)
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise ImportError("transformers and torch are required for local Qwen3 generation.") from exc

        self.args = args
        self.torch = torch
        tokenizer_kwargs = {
            "trust_remote_code": args.trust_remote_code,
            "local_files_only": args.local_files_only,
        }
        if args.load_in_4bit and args.load_in_8bit:
            raise ValueError("Choose at most one of --load_in_4bit and --load_in_8bit.")
        model_kwargs = {
            "torch_dtype": args.torch_dtype,
            "device_map": args.device_map,
            "trust_remote_code": args.trust_remote_code,
            "local_files_only": args.local_files_only,
        }
        if args.attn_implementation:
            model_kwargs["attn_implementation"] = args.attn_implementation
        if args.load_in_4bit:
            model_kwargs["load_in_4bit"] = True
        if args.load_in_8bit:
            model_kwargs["load_in_8bit"] = True

        self.tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, **tokenizer_kwargs)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, **model_kwargs)
        self.model.eval()

    def generate(self, prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.args.enable_thinking,
        )
        model_inputs = self.tokenizer([text], return_tensors="pt")
        model_inputs = model_inputs.to(self.model.device)
        generation_kwargs = {
            "max_new_tokens": self.args.max_new_tokens,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        if self.args.greedy:
            generation_kwargs["do_sample"] = False
        else:
            generation_kwargs.update(
                {
                    "do_sample": True,
                    "temperature": self.args.temperature,
                    "top_p": self.args.top_p,
                    "top_k": self.args.top_k,
                }
            )
        with self.torch.no_grad():
            generated_ids = self.model.generate(
                **model_inputs,
                **generation_kwargs,
            )
        output_ids = generated_ids[0][model_inputs["input_ids"].shape[-1] :]
        response = self.tokenizer.decode(output_ids, skip_special_tokens=True).strip()
        return re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()


def parse_generated_news(text: str):
    articles = []
    json_match = re.search(r"(\[[\s\S]*\]|\{[\s\S]*\})", text)
    if json_match:
        try:
            parsed = json.loads(json_match.group(1))
            if isinstance(parsed, dict):
                parsed = parsed.get("generated_news", parsed.get("news", [parsed]))
            if isinstance(parsed, list):
                for item in parsed:
                    if isinstance(item, dict):
                        articles.append(
                            {
                                "title": clean_text(item.get("title", item.get("Title"))),
                                "abstract": clean_text(item.get("abstract", item.get("Abstract"))),
                                "topic": clean_text(item.get("topic", item.get("Topic"))),
                                "keywords": clean_text(item.get("keywords", item.get("Keywords"))),
                                "evidence": clean_text(item.get("evidence", item.get("Evidence"))),
                            }
                        )
                if articles:
                    return articles
        except json.JSONDecodeError:
            pass

    pattern = re.compile(
        r"Title\s*:\s*(?P<title>.*?);\s*Abstract\s*:\s*(?P<abstract>.*?);\s*Topic\s*:\s*(?P<topic>.*?);\s*Keywords\s*:\s*(?P<keywords>.*?);\s*Evidence\s*:\s*(?P<evidence>.*?)(?=(?:\n\s*Title\s*:)|(?:\s*<end output>)|\Z)",
        flags=re.IGNORECASE | re.DOTALL,
    )
    for match in pattern.finditer(text):
        articles.append({key: clean_text(value) for key, value in match.groupdict().items()})
    if articles:
        return articles

    lines = [line.strip(" -") for line in text.splitlines() if line.strip()]
    return [{"title": line, "abstract": "", "topic": "", "keywords": "", "evidence": ""} for line in lines[:5]]


def read_done_users(path: str):
    done = set()
    if not os.path.exists(path):
        return done
    with open(path, "r", encoding="utf-8") as cache_f:
        for line in cache_f:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            user_id = clean_text(record.get("user_id_raw", record.get("user_id", record.get("uid"))))
            if user_id:
                done.add(user_id)
    return done


def default_output_path(args) -> str:
    return os.path.join("cache", "s2lenr", safe_name(args.DATASET_ROOT), safe_name(args.model_name_or_path), "generated_news.jsonl")


def parse_args():
    parser = argparse.ArgumentParser(description="Generate S2LENR synthetic news locally with Qwen3-32B.")
    parser.add_argument("--dataset_name", choices=["MIND", "ebnerd"], required=True)
    parser.add_argument("--DATASET_ROOT", required=True)
    parser.add_argument("--root", default=".")
    parser.add_argument("--output", default="")
    parser.add_argument("--model_name_or_path", default="Qwen/Qwen3-32B")
    parser.add_argument("--torch_dtype", default="auto")
    parser.add_argument("--device_map", default="auto")
    parser.add_argument("--attn_implementation", default="")
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--load_in_8bit", action="store_true")
    parser.add_argument("--num_news", type=int, default=5)
    parser.add_argument("--max_history", type=int, default=50)
    parser.add_argument("--prompt_history_num", type=int, default=10)
    parser.add_argument("--max_users", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--enable_thinking", action="store_true")
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    train_root = os.path.join(args.root, args.DATASET_ROOT, "train")
    dev_root = os.path.join(args.root, args.DATASET_ROOT, "dev")
    test_root = os.path.join(args.root, args.DATASET_ROOT, "test")
    roots = [train_root, dev_root, test_root]

    if args.dataset_name == "MIND":
        news = load_mind_news(roots)
        users = collect_mind_users(train_root, news, args.max_history)
    else:
        news = load_ebnerd_news(roots)
        users = collect_ebnerd_users(train_root, news, args.max_history)

    output = args.output or default_output_path(args)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    done_users = read_done_users(output)
    items = [(user_id, history) for user_id, history in users.items() if user_id not in done_users]
    if args.max_users > 0:
        items = items[: args.max_users]

    if args.dry_run:
        if not items:
            print("No users to generate.")
            return
        user_id, history_ids = items[0]
        prompt = PROMPT_TEMPLATE.format(
            num_news=args.num_news,
            history=build_history_text(history_ids, news, args.prompt_history_num),
        )
        print(f"user_id={user_id}")
        print(prompt)
        return

    generator = LocalQwenGenerator(args)
    with open(output, "a", encoding="utf-8") as out_f:
        for offset, (user_id, history_ids) in enumerate(items, start=1):
            prompt = PROMPT_TEMPLATE.format(
                num_news=args.num_news,
                history=build_history_text(history_ids, news, args.prompt_history_num),
            )
            raw_response = generator.generate(prompt)
            generated_news = parse_generated_news(raw_response)[: args.num_news]
            record = {
                "user_id_raw": user_id,
                "history": history_ids,
                "model": args.model_name_or_path,
                "generated_news": generated_news,
                "raw_response": raw_response,
            }
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            out_f.flush()
            print(f"[{offset}/{len(items)}] generated {len(generated_news)} articles for user {user_id}")
            if args.sleep > 0:
                time.sleep(args.sleep)


if __name__ == "__main__":
    main()

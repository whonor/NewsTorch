#!/usr/bin/env python3
import argparse
import json
import os
import re
import time
import urllib.parse
import urllib.request
from typing import Dict, Iterable, List


DIRECT_PROMPT = """Assuming you are a recommender system. Given the news article titled \"{title}\" that a user has read, provide an intriguing title that the user would be more likely to read. Preserve the article's original meaning. Return only the title."""

ENTITY_PROMPT = """Given the news article titled \"{title}\" that a user has read, what are the related entities in this title? Return only a JSON list of concise entity names."""

HIERARCHICAL_PROMPT = """Given the generated title \"{direct_title}\" and the related entities {entities}, write a concise, intriguing enriched title that highlights their connection and importance while preserving the original article's meaning. Keep it under {max_tokens} word tokens and return only the enriched title."""


def clean_text(value) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() == "nan" else text


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "default"


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
        data = pd.read_parquet(path)
        for _, row in data.iterrows():
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


def load_adressa_news(roots: Iterable[str]) -> Dict[str, Dict[str, str]]:
    news = {}
    for root in roots:
        path = os.path.join(root, "news.jsonl")
        if not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8") as news_f:
            for line in news_f:
                row = json.loads(line)
                news_id = clean_text(row.get("nid"))
                if news_id:
                    news.setdefault(news_id, {
                        "title": clean_text(row.get("title")),
                        "abstract": clean_text(row.get("abstract")),
                        "category": clean_text(row.get("category")),
                        "subcategory": clean_text(row.get("subcategory")),
                    })
    return news


def _strip_generation_markup(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", clean_text(text), flags=re.DOTALL)
    text = re.sub(r"</?(?:start|end) output>", "", text, flags=re.IGNORECASE)
    return text.strip().strip("`\"'")


def parse_generated_title(text: str) -> str:
    text = _strip_generation_markup(text)
    text = re.sub(r"^(?:enriched\s+)?title\s*:\s*", "", text, flags=re.IGNORECASE)
    return text.splitlines()[0].strip().strip("`\"'") if text else ""


def parse_entities(text: str) -> List[Dict[str, str]]:
    text = _strip_generation_markup(text)
    json_match = re.search(r"\[[\s\S]*\]", text)
    if json_match:
        try:
            values = json.loads(json_match.group(0))
        except json.JSONDecodeError:
            values = None
        if isinstance(values, list):
            entities = []
            for value in values:
                if isinstance(value, dict):
                    name = clean_text(value.get("name", value.get("label", value.get("entity"))))
                    wikidata_id = clean_text(value.get("wikidata_id", value.get("id")))
                else:
                    name = clean_text(value)
                    wikidata_id = ""
                if name:
                    entities.append({"name": name, "wikidata_id": wikidata_id})
            return entities

    parts = re.split(r"[,;\n]", text)
    entities = []
    for part in parts:
        name = re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", part).strip().strip("`\"'")
        if name:
            entities.append({"name": name, "wikidata_id": ""})
    return entities


def canonicalize_entities(entities: List[Dict[str, str]], timeout: float):
    canonical = []
    for entity in entities:
        params = urllib.parse.urlencode(
            {
                "action": "wbsearchentities",
                "search": entity["name"],
                "language": "en",
                "format": "json",
                "limit": 1,
            }
        )
        request = urllib.request.Request(
            "https://www.wikidata.org/w/api.php?" + params,
            headers={"User-Agent": "NewsTorch-PNR-LLM/1.0"},
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                result = json.load(response).get("search", [])
        except Exception as exc:
            print(f"Warning: Wikidata lookup failed for {entity['name']!r}: {exc}")
            canonical.append(entity)
            continue
        if result:
            canonical.append(
                {
                    "name": clean_text(result[0].get("label")) or entity["name"],
                    "wikidata_id": clean_text(result[0].get("id")),
                }
            )
        else:
            canonical.append(entity)
    return canonical


def patch_torch_pytree_compat(torch_module):
    pytree = getattr(getattr(torch_module, "utils", None), "_pytree", None)
    if pytree is None or hasattr(pytree, "register_pytree_node"):
        return
    if not hasattr(pytree, "_register_pytree_node"):
        return

    def register_pytree_node(node_type, flatten_fn, unflatten_fn, **kwargs):
        kwargs.pop("serialized_type_name", None)
        return pytree._register_pytree_node(node_type, flatten_fn, unflatten_fn, **kwargs)

    pytree.register_pytree_node = register_pytree_node


class QwenGenerator:
    def __init__(self, args):
        try:
            import torch

            patch_torch_pytree_compat(torch)
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise ImportError("transformers and torch are required for local generation.") from exc

        self.torch = torch
        tokenizer_args = {
            "trust_remote_code": args.trust_remote_code,
            "local_files_only": args.local_files_only,
        }
        model_args = {
            "torch_dtype": args.torch_dtype,
            "device_map": args.device_map,
            "trust_remote_code": args.trust_remote_code,
            "local_files_only": args.local_files_only,
        }
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, **tokenizer_args)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, **model_args)
        self.model.eval()
        self.max_new_tokens = args.max_new_tokens
        self.temperature = args.temperature
        self.top_p = args.top_p

    def generate(self, prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        try:
            model_text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
        except TypeError:
            model_text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        inputs = self.tokenizer([model_text], return_tensors="pt").to(self.model.device)
        with self.torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=self.temperature,
                top_p=self.top_p,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        generated = output[0][inputs["input_ids"].shape[-1] :]
        return self.tokenizer.decode(generated, skip_special_tokens=True)


def read_completed_news(path: str):
    completed = set()
    if not os.path.exists(path):
        return completed
    with open(path, "r", encoding="utf-8") as cache_f:
        for line in cache_f:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            news_id = clean_text(record.get("news_id", record.get("nid")))
            if news_id:
                completed.add(news_id)
    return completed


def default_output_path(args):
    return os.path.join(
        "cache",
        "pnr_llm",
        safe_name(args.DATASET_ROOT),
        safe_name(args.model_name_or_path),
        "enriched_news.jsonl",
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Generate offline PNR-LLM news enrichments.")
    parser.add_argument("--dataset_name", choices=["MIND", "ebnerd", "Adressa"], required=True)
    parser.add_argument("--DATASET_ROOT", required=True)
    parser.add_argument("--root", default=".")
    parser.add_argument("--output", default="")
    parser.add_argument("--model_name_or_path", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--torch_dtype", default="auto")
    parser.add_argument("--device_map", default="auto")
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--max_title_tokens", type=int, default=40)
    parser.add_argument("--max_entities", type=int, default=20)
    parser.add_argument("--max_news", type=int, default=0)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--request_timeout", type=float, default=30.0)
    parser.add_argument("--verify_wikidata", action="store_true")
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    roots = [os.path.join(args.root, args.DATASET_ROOT, split) for split in ("train", "dev", "test")]
    loaders = {"MIND": load_mind_news, "ebnerd": load_ebnerd_news, "Adressa": load_adressa_news}
    news = loaders[args.dataset_name](roots)
    output = args.output or default_output_path(args)
    output_dir = os.path.dirname(output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    completed = read_completed_news(output)
    items = [(news_id, article) for news_id, article in news.items() if news_id not in completed]
    if args.max_news > 0:
        items = items[: args.max_news]

    if not items:
        print("No news articles to enrich.")
        return

    if args.dry_run:
        news_id, article = items[0]
        direct_prompt = DIRECT_PROMPT.format(title=article["title"])
        entity_prompt = ENTITY_PROMPT.format(title=article["title"])
        hierarchical_prompt = HIERARCHICAL_PROMPT.format(
            direct_title="<direct title>",
            entities="[<entity 1>, <entity 2>]",
            max_tokens=args.max_title_tokens,
        )
        print(f"news_id={news_id}")
        print("\n[Direct Prompt]\n" + direct_prompt)
        print("\n[Entity Explorer]\n" + entity_prompt)
        print("\n[Hierarchical Prompt]\n" + hierarchical_prompt)
        return

    generator = QwenGenerator(args)
    with open(output, "a", encoding="utf-8") as output_f:
        for offset, (news_id, article) in enumerate(items, start=1):
            direct_title = parse_generated_title(
                generator.generate(DIRECT_PROMPT.format(title=article["title"]))
            )
            entities = parse_entities(
                generator.generate(ENTITY_PROMPT.format(title=article["title"]))
            )[: args.max_entities]
            if args.verify_wikidata:
                entities = canonicalize_entities(entities, args.request_timeout)
            entity_names = [entity["name"] for entity in entities]
            enriched_title = parse_generated_title(
                generator.generate(
                    HIERARCHICAL_PROMPT.format(
                        direct_title=direct_title or article["title"],
                        entities=json.dumps(entity_names, ensure_ascii=False),
                        max_tokens=args.max_title_tokens,
                    )
                )
            )
            record = {
                "news_id": news_id,
                "original_title": article["title"],
                "direct_title": direct_title,
                "enriched_title": enriched_title or direct_title or article["title"],
                "enriched_entities": entities,
                "category": article["category"],
                "subcategory": article["subcategory"],
                "model": args.model_name_or_path,
                "backend": "local-qwen",
            }
            output_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            output_f.flush()
            print(f"[{offset}/{len(items)}] enriched {news_id} with {len(entities)} entities")
            if args.sleep > 0:
                time.sleep(args.sleep)


if __name__ == "__main__":
    main()

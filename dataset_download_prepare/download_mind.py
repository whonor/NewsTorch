#!/usr/bin/env python3
import os
import sys
from pathlib import Path
import requests
from zipfile import ZipFile
from tqdm import tqdm

def download_file(url: str, dest: Path, chunk_size: int = 1024):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"already existed，skip download: {dest}")
        return
    resp = requests.get(url, stream=True)
    resp.raise_for_status()
    total = int(resp.headers.get('content-length', 0))
    with open(dest, 'wb') as f, tqdm(
        total=total, unit='iB', unit_scale=True, desc=dest.name
    ) as bar:
        for chunk in resp.iter_content(chunk_size=chunk_size):
            f.write(chunk)
            bar.update(len(chunk))
    print(f"download complete: {dest}")

def extract_zip(zip_path: Path, extract_to: Path):
    if extract_to.exists() and any(extract_to.iterdir()):
        print(f"existed and no empty，skip extracted: {extract_to}")
        return
    print(f"Extracting {zip_path} → {extract_to}")
    extract_to.mkdir(parents=True, exist_ok=True)
    with ZipFile(zip_path, 'r') as z:
        z.extractall(extract_to)
    print(f"extract complete: {extract_to}")

def process_dataset(base_dir: Path, splits: dict, include_wikidata: bool = False):
    download_dir = base_dir / 'download'
    for split, url in splits.items():
        filename = Path(url).name
        zip_path = download_dir / filename
        download_file(url, zip_path)
        extract_zip(zip_path, download_dir / split)

    if include_wikidata:
        wikidata_url = 'https://mind201910.blob.core.windows.net/knowledge-graph/wikidata-graph.zip'
        wikidata_filename = Path(wikidata_url).name
        wikidata_zip = download_dir / wikidata_filename
        # download_file(wikidata_url, wikidata_zip)
        if wikidata_zip.exists():
            extract_zip(wikidata_zip, download_dir / 'wikidata-graph')



def main():
    datasets = {
        # 'MIND-200k': {
        #     'train': 'https://recodatasets.z20.web.core.windows.net/newsrec/MINDlarge_train.zip',
        #     'dev':   'https://recodatasets.z20.web.core.windows.net/newsrec/MINDlarge_dev.zip',
        # },
        'MIND-small': {
            'train': 'https://recodatasets.z20.web.core.windows.net/newsrec/MINDsmall_train.zip',
            'dev':   'https://recodatasets.z20.web.core.windows.net/newsrec/MINDsmall_dev.zip',
        },
        # 'MIND-large': {
        #     'train': 'https://recodatasets.z20.web.core.windows.net/newsrec/MINDlarge_train.zip',
        #     'dev':   'https://recodatasets.z20.web.core.windows.net/newsrec/MINDlarge_dev.zip',
        #     'test':  'https://recodatasets.z20.web.core.windows.net/newsrec/MINDlarge_test.zip',
        # },
    }

    root = Path(__file__).resolve().parent.parent

    for name, splits in datasets.items():
        print(f"\n=== Processing dataset: {name} ===")
        ds_dir = root / name
        ds_dir.mkdir(parents=True, exist_ok=True)
        process_dataset(ds_dir, splits, include_wikidata=True)

    print("\nAll downloads and extractions completed.")

if __name__ == '__main__':
    try:
        import requests
        from tqdm import tqdm
    except ImportError:
        print("Please install dependencies：pip install requests tqdm", file=sys.stderr)
        sys.exit(1)
    main()

#!/usr/bin/env python3
"""Download and extract the official Adressa one-week light dataset."""

import argparse
import hashlib
import json
import os
import shutil
import sys
import tarfile
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_URL = "https://reclab.idi.ntnu.no/dataset/one_week.tar.gz"
LICENSE_URL = "https://creativecommons.org/licenses/by-nc-sa/4.0/"
DEFAULT_DATASET_ROOT = ROOT / "Adressa-1week"


def _sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_file(url: str, destination: Path, chunk_size: int = 1024 * 1024,
                  timeout=(30, 300), force: bool = False) -> Path:
    """Stream an HTTP file to disk and resume an existing `.part` download."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    if destination.is_file() and destination.stat().st_size > 0 and not force:
        print(f"Archive already exists, skipping download: {destination}")
        return destination
    if force:
        destination.unlink(missing_ok=True)
        partial.unlink(missing_ok=True)
    downloaded = partial.stat().st_size if partial.exists() else 0
    headers = {
        "User-Agent": "NewsTorch-Adressa-Downloader/1.0",
        "Accept-Encoding": "identity",
    }
    if downloaded:
        headers["Range"] = f"bytes={downloaded}-"

    response = requests.get(
        url,
        stream=True,
        allow_redirects=True,
        headers=headers,
        timeout=timeout,
    )
    response.raise_for_status()
    resumed = downloaded > 0 and response.status_code == 206
    if downloaded and not resumed:
        downloaded = 0
    content_length = int(response.headers.get("content-length", 0) or 0)
    total = downloaded + content_length if content_length else None
    mode = "ab" if resumed else "wb"

    with partial.open(mode) as output, tqdm(
        total=total,
        initial=downloaded,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        desc=destination.name,
    ) as progress:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if not chunk:
                continue
            output.write(chunk)
            progress.update(len(chunk))
    partial.replace(destination)
    print(f"Download complete: {destination}")
    return destination


def _validated_target(root: Path, member_name: str) -> Path:
    """Reject absolute paths and archive members escaping the extraction root."""
    root = root.resolve()
    target = (root / member_name).resolve()
    try:
        target.relative_to(root)
    except ValueError as error:
        raise ValueError(f"Unsafe archive member path: {member_name!r}") from error
    return target


def _extract_tar(archive: Path, destination: Path) -> int:
    with tarfile.open(archive, "r:*") as source:
        members = source.getmembers()
        for member in members:
            _validated_target(destination, member.name)
            if not (member.isfile() or member.isdir()):
                raise ValueError(f"Unsupported archive member type: {member.name!r}")
        source.extractall(destination, members=members)
    return len(members)


def _extract_zip(archive: Path, destination: Path) -> int:
    with zipfile.ZipFile(archive, "r") as source:
        members = source.infolist()
        for member in members:
            _validated_target(destination, member.filename)
            unix_mode = member.external_attr >> 16
            if unix_mode and (unix_mode & 0o170000) == 0o120000:
                raise ValueError(f"Archive links are not allowed: {member.filename!r}")
        source.extractall(destination)
    return len(members)


def extract_archive(archive: Path, destination: Path, force: bool = False) -> Path:
    """Extract a tar or ZIP archive once and record the source archive identity."""
    archive = Path(archive)
    destination = Path(destination)
    marker = destination / ".newstorch-extracted.json"
    archive_identity = {
        "archive": archive.name,
        "size": archive.stat().st_size,
        "mtime_ns": archive.stat().st_mtime_ns,
    }
    if marker.exists() and not force:
        try:
            previous = json.loads(marker.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            previous = {}
        if all(previous.get(key) == value for key, value in archive_identity.items()):
            print(f"Archive already extracted, skipping: {destination}")
            return destination
    if force and destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=True)

    print(f"Extracting {archive} -> {destination}")
    if tarfile.is_tarfile(archive):
        member_count = _extract_tar(archive, destination)
        archive_type = "tar"
    elif zipfile.is_zipfile(archive):
        member_count = _extract_zip(archive, destination)
        archive_type = "zip"
    else:
        raise ValueError(f"Unsupported or corrupt archive: {archive}")
    marker.write_text(
        json.dumps(
            {**archive_identity, "type": archive_type, "members": member_count},
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Extraction complete: {destination} ({member_count} archive members)")
    return destination


def download_adressa_1week(dataset_root=DEFAULT_DATASET_ROOT, url=DEFAULT_URL,
                           expected_sha256="", force_download=False,
                           force_extract=False):
    dataset_root = Path(dataset_root).expanduser().resolve()
    archive_name = Path(url.split("?", 1)[0]).name or "one_week.tar.gz"
    archive = download_file(
        url,
        dataset_root / "download" / archive_name,
        force=force_download,
    )
    if expected_sha256:
        actual = _sha256(archive)
        if actual.lower() != expected_sha256.lower():
            raise ValueError(
                f"SHA-256 mismatch for {archive}: expected {expected_sha256}, got {actual}"
            )
        print(f"SHA-256 verified: {actual}")
    raw_dir = extract_archive(archive, dataset_root / "raw", force=force_extract)
    return archive, raw_dir


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--url", default=DEFAULT_URL, help="Archive URL; override for a mirror")
    parser.add_argument("--sha256", default="", help="Optional expected archive SHA-256")
    parser.add_argument("--accept-license", action="store_true", help="Confirm CC BY-NC-SA 4.0 non-commercial dataset terms")
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--force-extract", action="store_true")
    parser.add_argument("--prepare", action="store_true", help="Also create NewsTorch train/dev/test files")
    parser.add_argument("--negative-num", type=int, default=20)
    parser.add_argument("--max-history-num", type=int, default=50)
    parser.add_argument("--min-history", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.accept_license:
        raise SystemExit(
            "Adressa is licensed CC BY-NC-SA 4.0 for non-commercial use. "
            f"Review {LICENSE_URL} and rerun with --accept-license."
        )
    dataset_root = args.dataset_root.expanduser().resolve()
    archive_name = Path(args.url.split("?", 1)[0]).name or "one_week.tar.gz"
    if args.dry_run:
        print(f"URL:       {args.url}")
        print(f"Archive:   {dataset_root / 'download' / archive_name}")
        print(f"Extract to:{dataset_root / 'raw'}")
        print(f"Prepare:   {args.prepare}")
        return

    archive, raw_dir = download_adressa_1week(
        dataset_root=dataset_root,
        url=args.url,
        expected_sha256=args.sha256,
        force_download=args.force_download,
        force_extract=args.force_extract,
    )
    print(f"Adressa archive ready: {archive}")
    print(f"Adressa raw files ready: {raw_dir}")
    if args.prepare:
        from dataset_download_prepare.Adressa_dataset_prepare import prepare_adressa_1week

        manifest = prepare_adressa_1week(
            dataset_root=str(dataset_root),
            raw_dir=str(raw_dir),
            negative_num=args.negative_num,
            max_history_num=args.max_history_num,
            min_history=args.min_history,
            seed=args.seed,
            force=args.force_extract,
        )
        print(f"NewsTorch Adressa split ready: {manifest}")


if __name__ == "__main__":
    main()

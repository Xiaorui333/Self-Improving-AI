#!/usr/bin/env python3
"""Download benchmark datasets from the official AgentFlow repository.

Each benchmark's data.json is fetched from
  https://raw.githubusercontent.com/lupantech/AgentFlow/main/test/{name}/data/data.json
and stored locally under  benchmarks/data/{name}/data.json.
"""

from __future__ import annotations

import json
import os
import urllib.request

REPO_RAW = "https://raw.githubusercontent.com/lupantech/AgentFlow/main/test"

BENCHMARKS = [
    "bamboogle",
    "2wiki",
    "hotpotqa",
    "musique",
    "gaia",
    "aime24",
    "amc23",
    "gameof24",
    "gpqa",
    "medqa",
]

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def download_benchmark(name: str) -> str:
    """Download a single benchmark and return the local path."""
    dest_dir = os.path.join(DATA_DIR, name)
    os.makedirs(dest_dir, exist_ok=True)
    dest_path = os.path.join(dest_dir, "data.json")

    if os.path.exists(dest_path):
        print(f"  {name}: already exists, skipping")
        return dest_path

    url = f"{REPO_RAW}/{name}/data/data.json"
    print(f"  {name}: downloading from {url} ...")
    try:
        urllib.request.urlretrieve(url, dest_path)
        with open(dest_path) as f:
            data = json.load(f)
        print(f"  {name}: {len(data)} samples")
    except Exception as exc:
        print(f"  {name}: FAILED -- {exc}")
        if os.path.exists(dest_path):
            os.remove(dest_path)
        return ""

    return dest_path


def main() -> None:
    print("Downloading AgentFlow benchmark datasets ...\n")
    for name in BENCHMARKS:
        download_benchmark(name)
    print("\nDone.")


if __name__ == "__main__":
    main()

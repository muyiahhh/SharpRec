#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Convert Amazon Reviews'23 review_*.jsonl(.gz) -> CSV
Keep only: user_id, parent_asin, timestamp, rating
"""

import json
import gzip
import csv
import argparse
import os
import pandas as pd
from tqdm import tqdm


def open_maybe_gzip(path, mode="rt", encoding="utf-8"):
    """
    Open .jsonl or .jsonl.gz transparently.
    """
    if path.endswith(".gz"):
        # use text mode for gzip
        return gzip.open(path, mode=mode, encoding=encoding)
    return open(path, mode=mode, encoding=encoding)

def run(original_jsonl, new_csv):
    headers = ["user_id", "parent_asin", "timestamp", "rating"]
    open_func = open_maybe_gzip

    total, kept, skipped = 0, 0, 0

    with open_func(original_jsonl, "rt") as fin, open(new_csv, "w", newline='', encoding="utf-8") as fout:
        writer = csv.writer(fout)
        writer.writerow(headers)
        for line in tqdm(fin, desc="Streaming JSONL → CSV"):
            total += 1
            try:
                record = json.loads(line.strip())
                if all(k in record for k in headers):
                    writer.writerow([
                        record["user_id"],
                        record["parent_asin"],
                        int(record["timestamp"]),
                        float(record["rating"])
                    ])
                    kept += 1
            except Exception:
                skipped += 1
                continue
    print(f"[Done] Input: {original_jsonl}")
    print(f"  total lines        : {total:,}")
    print(f"  written rows       : {kept:,}")
    print(f"  skipped (missing)  : {skipped:,}")  # missing
    print(f"[Schema] Columns: {headers}")
    print(f"[Done] Wrote {kept} rows → {new_csv}")



if __name__ == "__main__":
    
    # sports
    original_jsonl = "Sports_and_Outdoors.jsonl"
    new_csv = "Sports.csv"

    # Toys_and_Games
    original_jsonl = "Toys_and_Games.jsonl"
    new_csv = "Toys_and_Games.csv"


    os.makedirs(os.path.dirname(new_csv), exist_ok=True)

    run(original_jsonl, new_csv)

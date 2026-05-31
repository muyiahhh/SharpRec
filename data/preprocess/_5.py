import os
import random
from typing import List, Optional, Set
from collections import Counter
from datetime import datetime
from tqdm import tqdm

ONE_MONTH_MS = 30 * 24 * 60 * 60 * 1000

def parse_full_line(line: str):
    line = line.strip()
    if not line: return None, [], [], []
    parts = line.split()
    if len(parts) < 2: return None, [], [], []
    user_id = parts[0]
    item_tokens = parts[1:]
    items, ts_list, domains = [], [], []
    for tok in item_tokens:
        try:
            item_id, ts_str, domain = tok.split("|")
            ts = int(ts_str)
        except ValueError: continue
        items.append(item_id)
        ts_list.append(ts)
        domains.append(domain)
    if len(ts_list) > 1:
        zipped = sorted(zip(ts_list, items, domains), key=lambda x: x[0])
        ts_list, items, domains = zip(*zipped)
        ts_list, items, domains = list(ts_list), list(items), list(domains)
    return user_id, items, ts_list, domains

def ts_to_datetime_str(ts_ms: int) -> str:
    dt = datetime.fromtimestamp(ts_ms / 1000)
    return dt.strftime("%Y-%m-%d_%H:%M:%S")

def sample_length_linear_increasing(min_len: int, max_len: int, bias_alpha: float = 1.0) -> int:
    if max_len <= min_len: return min_len
    lengths = list(range(min_len, max_len + 1))
    weights = [1.0 + bias_alpha * i for i in range(len(lengths))]
    return random.choices(lengths, weights=weights, k=1)[0]

def generate_sparse_subsequences_for_user(items, ts_list, domains, window_ms, min_len, len_high, min_stride, max_stride, required_domains, min_count_per_domain, length_bias_alpha):
    n = len(items)
    if n < min_len: return []
    subseqs, last_end_idx, i = [], -1, 0
    while i <= n - min_len:
        start_ts = ts_list[i]
        window_end_ts = start_ts + window_ms
        j_time = i
        while j_time + 1 < n and ts_list[j_time + 1] <= window_end_ts: j_time += 1
        candidate_len = j_time - i + 1
        if candidate_len < min_len:
            i += 1
            continue
        target_len = candidate_len if candidate_len <= len_high else sample_length_linear_increasing(min_len, len_high, length_bias_alpha)
        end_idx = i + target_len - 1
        if end_idx >= n: break
        if end_idx <= last_end_idx:
            i += random.randint(min_stride, max_stride)
            continue
        sub_items, sub_ts, sub_domains = items[i:end_idx+1], ts_list[i:end_idx+1], domains[i:end_idx+1]
        if required_domains:
            cnt = Counter(sub_domains)
            if any(cnt.get(d, 0) < min_count_per_domain for d in required_domains):
                i += random.randint(min_stride, max_stride)
                continue
        subseqs.append((sub_items, sub_ts, sub_domains))
        last_end_idx = end_idx
        i += random.randint(min_stride, max_stride)
    return subseqs

def select_drop_users_with_one_sample(input_path, window_ms, min_len, len_high, min_stride, max_stride, required_domains, min_count_per_domain, length_bias_alpha, drop_ratio, seed):
    rng = random.Random(seed)
    user2cnt = Counter()
    with open(input_path, "r", encoding="utf-8") as fin:
        for line in fin:
            uid, items, ts, doms = parse_full_line(line)
            if uid: user2cnt[uid] = len(generate_sparse_subsequences_for_user(items, ts, doms, window_ms, min_len, len_high, min_stride, max_stride, required_domains, min_count_per_domain, length_bias_alpha))
    one_sample_users = [u for u, c in user2cnt.items() if c == 1]
    drop_n = int(len(one_sample_users) * drop_ratio)
    return set(rng.sample(one_sample_users, drop_n)) if drop_n > 0 else set()

def process_long_sequence_file(input_path, output_path, window_ms=ONE_MONTH_MS, min_len=6, len_high=20, min_stride=1, max_stride=3, required_domains=None, min_count_per_domain=3, length_bias_alpha=1.0, seed=42, drop_one_sample_users=False, drop_ratio=0.0):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    random.seed(seed)
    drop_users = select_drop_users_with_one_sample(input_path, window_ms, min_len, len_high, min_stride, max_stride, required_domains, min_count_per_domain, length_bias_alpha, drop_ratio, seed) if drop_one_sample_users and drop_ratio > 0 else set()
    with open(input_path, "r", encoding="utf-8") as fin, open(output_path, "w", encoding="utf-8") as fout:
        for line in tqdm(fin):
            uid, items, ts_list, domains = parse_full_line(line)
            if not uid or uid in drop_users: continue
            subseqs = generate_sparse_subsequences_for_user(items, ts_list, domains, window_ms, min_len, len_high, min_stride, max_stride, required_domains, min_count_per_domain, length_bias_alpha)
            for s_i, s_t, s_d in subseqs:
                tokens = [f"{it}|{t}|{ts_to_datetime_str(t)}|{d}" for it, t, d in zip(s_i, s_t, s_d)]
                fout.write(f"{uid} {len(s_i)} " + " ".join(tokens) + "\n")

if __name__ == "__main__":
    SOURCE, TARGET = "sport", "toy"
    PAIR_DIR = f"data/{SOURCE}2{TARGET}"
    overlap_in, source_in, target_in = f"{PAIR_DIR}/{SOURCE}2{TARGET}_overlap.txt", f"{PAIR_DIR}/{SOURCE}_partial.txt", f"{PAIR_DIR}/{TARGET}_partial.txt"
    overlap_out, source_out, target_out = f"{PAIR_DIR}/preprocess/{SOURCE}2{TARGET}_overlap_1month.txt", f"{PAIR_DIR}/preprocess/{SOURCE}_partial_1month.txt", f"{PAIR_DIR}/preprocess/{TARGET}_partial_1month.txt"

    process_long_sequence_file(overlap_in, overlap_out, window_ms=ONE_MONTH_MS*3, min_len=6, len_high=20, min_stride=1, max_stride=3, required_domains=[SOURCE, TARGET], min_count_per_domain=2, length_bias_alpha=3, seed=42)
    process_long_sequence_file(source_in, source_out, window_ms=ONE_MONTH_MS*8, min_len=6, len_high=20, min_stride=1, max_stride=3, seed=43)
    process_long_sequence_file(target_in, target_out, window_ms=ONE_MONTH_MS, min_len=6, len_high=20, min_stride=10, max_stride=20, seed=44, drop_one_sample_users=True, drop_ratio=0.3)

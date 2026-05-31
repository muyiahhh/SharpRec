import os
import shutil
from collections import defaultdict, Counter
from typing import List, Optional, Set, Dict, Tuple
from tqdm import tqdm

def parse_subseq_line(line: str):
    line = line.strip()
    if not line:
        return None, 0, [], [], []
    parts = line.split()
    if len(parts) < 3:
        return None, 0, [], [], []
    user_id = parts[0]
    try:
        seq_len = int(parts[1])
        token_start = 2
    except ValueError:
        seq_len = len(parts) - 1
        token_start = 1
    tokens = parts[token_start:]
    item_ids = []
    domains = []
    for tok in tokens:
        pieces = tok.split("|")
        if len(pieces) < 2:
            continue
        item_id = pieces[0]
        domain = pieces[-1]
        item_ids.append(item_id)
        domains.append(domain)
    return user_id, seq_len, tokens, item_ids, domains

def compute_degrees_from_subseq_files(file_paths: List[str]) -> Tuple[Dict[str, Set[str]], Dict[str, Set[str]]]:
    user2items: Dict[str, Set[str]] = defaultdict(set)
    item2users: Dict[str, Set[str]] = defaultdict(set)
    for path in file_paths:
        if not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8") as f:
            for line in tqdm(f, desc=f"Parsing {os.path.basename(path)}"):
                user_id, _, _, item_ids, _ = parse_subseq_line(line)
                if user_id is None or not item_ids:
                    continue
                uniq_items = set(item_ids)
                for iid in uniq_items:
                    user2items[user_id].add(iid)
                    item2users[iid].add(user_id)
    return user2items, item2users

def filter_subseq_file_once(
    input_path: str,
    output_path: str,
    kept_users: Set[str],
    kept_items: Set[str],
    min_len: int,
    max_len: int,
    required_domains: Optional[List[str]] = None,
    min_count_per_domain: int = 3,
) -> Tuple[int, int]:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    num_samples = 0
    num_edges = 0
    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:
        for line in tqdm(fin, desc=f"Filtering {os.path.basename(input_path)}"):
            if not line.strip():
                continue
            user_id, _, tokens, item_ids, domains = parse_subseq_line(line)
            if user_id is None or not tokens:
                continue
            if user_id not in kept_users:
                continue
            new_tokens = []
            new_domains = []
            for tok in tokens:
                pieces = tok.split("|")
                if len(pieces) < 2:
                    continue
                iid = pieces[0]
                if iid not in kept_items:
                    continue
                new_tokens.append(tok)
                new_domains.append(pieces[-1])
            new_len = len(new_tokens)
            if new_len < min_len or new_len > max_len:
                continue
            if required_domains is not None:
                cnt = Counter(new_domains)
                if any(cnt.get(d, 0) < min_count_per_domain for d in required_domains):
                    continue
            line_out = f"{user_id} {new_len} " + " ".join(new_tokens) + "\n"
            fout.write(line_out)
            num_samples += 1
            num_edges += new_len
    return num_samples, num_edges

def iterative_kcore_cleaning(
    overlap_in: str,
    source_in: str,
    target_in: str,
    overlap_out: str,
    source_out: str,
    target_out: str,
    source_domain: str,
    target_domain: str,
    min_items_per_user: int = 10,
    min_users_per_item: int = 10,
    min_len: int = 6,
    max_len: int = 20,
    max_iters: int = 10,
):
    work_overlap, work_source, work_target = overlap_out + ".work", source_out + ".work", target_out + ".work"
    shutil.copyfile(overlap_in, work_overlap)
    shutil.copyfile(source_in,  work_source)
    shutil.copyfile(target_in,  work_target)
    current_files = {"overlap": work_overlap, "source": work_source, "target": work_target}
    prev_edges = None
    for it in range(max_iters):
        user2items, item2users = compute_degrees_from_subseq_files([current_files["overlap"], current_files["source"], current_files["target"]])
        kept_users = {u for u, items in user2items.items() if len(items) >= min_items_per_user}
        kept_items = {i for i, users in item2users.items() if len(users) >= min_users_per_item}
        if not kept_users or not kept_items:
            break
        total_edges = 0
        total_samples = 0
        tmp_ov, tmp_src, tmp_tgt = [current_files[k] + f".iter{it+1}" for k in ["overlap", "source", "target"]]
        s_ov, e_ov = filter_subseq_file_once(current_files["overlap"], tmp_ov, kept_users, kept_items, min_len, max_len, [source_domain, target_domain], 2)
        total_samples += s_ov; total_edges += e_ov
        s_src, e_src = filter_subseq_file_once(current_files["source"], tmp_src, kept_users, kept_items, min_len, max_len, None)
        total_samples += s_src; total_edges += e_src
        s_tgt, e_tgt = filter_subseq_file_once(current_files["target"], tmp_tgt, kept_users, kept_items, min_len, max_len, None)
        total_samples += s_tgt; total_edges += e_tgt
        if total_edges == 0 or total_samples == 0:
            for role in ["overlap", "source", "target"]: open(current_files[role], "w").close()
            break
        shutil.move(tmp_ov, current_files["overlap"]); shutil.move(tmp_src, current_files["source"]); shutil.move(tmp_tgt, current_files["target"])
        if prev_edges is not None and total_edges == prev_edges:
            break
        prev_edges = total_edges
    shutil.copyfile(current_files["overlap"], overlap_out); shutil.copyfile(current_files["source"], source_out); shutil.copyfile(current_files["target"], target_out)

if __name__ == "__main__":
    SOURCE, TARGET = "sport", "toy"
    PRE = f"data/{SOURCE}2{TARGET}/preprocess"
    iterative_kcore_cleaning(f"{PRE}/{SOURCE}2{TARGET}_overlap_1month.txt", f"{PRE}/{SOURCE}_partial_1month.txt", f"{PRE}/{TARGET}_partial_1month.txt",
                             f"{PRE}/{SOURCE}2{TARGET}_overlap_1month_kcore.txt", f"{PRE}/{SOURCE}_partial_1month_kcore.txt", f"{PRE}/{TARGET}_partial_1month_kcore.txt",
                             SOURCE, TARGET, 5, 5, 6, 20, 200)

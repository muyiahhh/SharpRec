import json
import math
import argparse
import os

def calculate_all_metrics(rank_lists, truth_lists, cutoffs=[1, 3, 5, 10]):
    """
    一次性计算多个k值的 NDCG, HR, MRR
    """
    # 初始化结果字典
    metrics = {k: {'NDCG': 0.0, 'HR': 0.0, 'MRR': 0.0} for k in cutoffs}
    n = len(truth_lists)
    
    if n == 0:
        return {}

    for r_items, t_item in zip(rank_lists, truth_lists):
        # 查找真实标签在预测列表中的位置
        try:
            # 这里的 r_items 和 t_item 都已经是 strip() 过的，可以直接比较
            idx = r_items.index(t_item)
            rank = idx + 1
        except ValueError:
            rank = float('inf') # 未找到

        # 如果找到了，计算各项指标贡献值
        if rank != float('inf'):
            ndcg_val = 1.0 / math.log2(rank + 1)
            rr_val = 1.0 / rank
            
            for k in cutoffs:
                if rank <= k:
                    metrics[k]['HR'] += 1
                    metrics[k]['NDCG'] += ndcg_val
                    metrics[k]['MRR'] += rr_val
    
    # 计算平均值
    final_results = {}
    for k in cutoffs:
        final_results[f'NDCG@{k}'] = metrics[k]['NDCG'] / n
        final_results[f'HR@{k}'] = metrics[k]['HR'] / n
        final_results[f'MRR@{k}'] = metrics[k]['MRR'] / n
        
    return final_results

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--truth_file', type=str, required=True, help="Path to the ground truth JSONL file")
    parser.add_argument('--res_path', type=str, required=True, help="Directory containing result files")
    args = parser.parse_args()
    
    # 1. Load truth data
    truth = []
    print(f"Loading truth file from: {args.truth_file}")
    with open(args.truth_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    truth.append(json.loads(line))
                except json.JSONDecodeError:
                    print(f"Warning: Failed to decode truth line: {line.strip()[:50]}...")
    
    # 提取 Ground Truth 字符串，并进行 strip() 清洗
    # 修复点 1：确保 Truth 去除首尾空格
    t_list = []
    for i in truth:
        try:
            content = i['messages'][2]['content']
            # 取第一部分并清洗
            t_list.append(content.split('||')[0].strip())
        except (KeyError, IndexError, AttributeError):
            # 容错处理
            t_list.append(str(i).split('||')[0].strip())

    metrics_storage = {}
    
    # 2. Process each result file
    if not os.path.exists(args.res_path):
        print(f"Result path {args.res_path} does not exist.")
        exit(1)

    print(f"Evaluating results in: {args.res_path}")
    
    for file in os.listdir(args.res_path):
        file_path = os.path.join(args.res_path, file)
        if not os.path.isfile(file_path): 
            continue
            
        res_raw = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                try:
                    parsed = json.loads(line)
                    res_raw.append(parsed if isinstance(parsed, str) else str(parsed))
                except json.JSONDecodeError:
                    res_raw.append(line)
        
        # 修复点 2：在分割 Prediction 列表时，对每个元素都进行 strip()
        # [x.strip() for x in i.split('||')] 确保列表里每个元素都是干净的
        r_list = [[x.strip() for x in i.split('||')] for i in res_raw]
        
        # 对齐数据长度
        eval_len = min(len(r_list), len(t_list))
        if eval_len == 0:
            print(f"Skipping {file}: No valid data overlap (r={len(r_list)}, t={len(t_list)}).")
            continue
            
        print(f"Processing file: {file} | Samples: {eval_len}")

        # 计算所有指标
        results = calculate_all_metrics(r_list[:eval_len], t_list[:eval_len], cutoffs=[1, 3, 5, 10])
        metrics_storage[file] = results

    # 3. Find best model and Print
    if not metrics_storage:
        print("No metrics calculated.")
        exit()

    # 沿用之前的选择标准：NDCG@1 + NDCG@3 最大
    best_file = max(metrics_storage, key=lambda x: metrics_storage[x]['NDCG@1'] + metrics_storage[x]['NDCG@3'])
    
    print("\n" + "="*50)
    print(f"Best Model File: {best_file}")
    best_model_name = best_file.replace(".jsonl", "").replace(".json", "")
    print(f"Model Name: {best_model_name}")
    print("-" * 50)
    
    best_metrics = metrics_storage[best_file]
    
    headers = ["K", "NDCG@K", "HR@K", "MRR@K"]
    print(f"{headers[0]:<5} {headers[1]:<12} {headers[2]:<12} {headers[3]:<12}")
    print("-" * 45)
    
    for k in [1, 3, 5, 10]:
        ndcg = best_metrics[f'NDCG@{k}']
        hr = best_metrics[f'HR@{k}']
        mrr = best_metrics[f'MRR@{k}']
        print(f"{k:<5} {ndcg:<12.4f} {hr:<12.4f} {mrr:<12.4f}")
    
    print("="*50)

    print(f"metrics : NDCG@3: {best_metrics[f'NDCG@3']:<12.4f} HR@3:{best_metrics[f'HR@3']:<12.4f} MRR@3:{best_metrics[f'MRR@3']:<12.4f} NDCG@5:{best_metrics[f'NDCG@5']:<12.4f} HR@5:{best_metrics[f'HR@5']:<12.4f} MRR@5{best_metrics[f'MRR@5']:<12.4f}")
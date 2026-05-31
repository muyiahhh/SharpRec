import json
import pandas as pd
from tqdm import tqdm

def run(processed_df, original_review, new_review, new_item, new_csv, min_items_per_user=5, min_users_per_item=5):
    df = processed_df.copy()

    while True:
        before = len(df)
        user_item_counts = df.groupby('user_id')['parent_asin'].nunique()
        df = df[df['user_id'].isin(user_item_counts[user_item_counts >= min_items_per_user].index)]
        item_user_counts = df.groupby('parent_asin')['user_id'].nunique()
        df = df[df['parent_asin'].isin(item_user_counts[item_user_counts >= min_users_per_item].index)]
        if len(df) == before or len(df) == 0:
            break

    if df.empty:
        df.to_csv(new_csv, index=False)
        open(new_review, 'w').close()
        open(new_item, 'w').close()
        return df

    df.to_csv(new_csv, index=False)
    kept_parent_asins = set(df['parent_asin'].unique())
    updated_items = []

    with open(new_item, 'r', encoding='utf-8') as f:
        for line in f:
            item_obj = json.loads(line.strip())
            if item_obj.get("parent_asin") in kept_parent_asins:
                updated_items.append(item_obj)

    with open(new_item, 'w', encoding='utf-8') as f:
        for obj in updated_items:
            f.write(json.dumps(obj, ensure_ascii=False) + '\n')

    kept_user_ids = set(df['user_id'].unique())
    seen_combos = set()
    with open(original_review, 'r', encoding='utf-8') as fin, open(new_review, 'w', encoding='utf-8') as fout:
        for line in tqdm(fin):
            data = json.loads(line.strip())
            uid, asin, ts = data.get('user_id'), data.get('parent_asin'), data.get('timestamp')
            if uid in kept_user_ids and asin in kept_parent_asins:
                combo = (uid, asin, ts)
                if combo not in seen_combos:
                    seen_combos.add(combo)
                    fout.write(json.dumps(data, ensure_ascii=False) + '\n')
    return df

if __name__ == "__main__":
    original_review = "data/originalData/Sports_and_Outdoors.jsonl"
    new_review = "data/sport/userReview.jsonl"
    new_item = "data/sport/itemDescription.jsonl"
    new_csv = "data/sport/data.csv"

    df_processed = pd.read_csv(new_csv)
    run(df_processed, original_review, new_review, new_item, new_csv)

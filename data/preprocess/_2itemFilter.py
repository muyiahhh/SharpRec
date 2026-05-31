import pandas as pd
import json
from tqdm import tqdm

def run(df_processed, original_mata, new_item):
    kept_parent_asins = set(df_processed['parent_asin'].unique())
    none_description_asins = set()
    processed_meta_data = []

    with open(original_mata, 'r') as file:
        for line in tqdm(file, desc="Processing item metadata"):
            data = json.loads(line.strip())
            if data.get('parent_asin') not in kept_parent_asins:
                continue

            parent_asin = data['parent_asin']
            title = data.get('title')
            if title is None or title == "":
                none_description_asins.add(parent_asin)
                continue

            description = data.get('description')
            if description is None or len(description) < 1:
                none_description_asins.add(parent_asin)
                continue

            item_obj = {
                "parent_asin": parent_asin,
                "title": title,
                "description": description,
                "categories": data.get('categories', []),
                "features": data.get('features', [])
            }
            processed_meta_data.append(item_obj)

    with open(new_item, 'w') as file:
        for item in processed_meta_data:
            file.write(json.dumps(item) + '\n')

    df_csv = df_processed[~df_processed['parent_asin'].isin(none_description_asins)]
    return df_csv

if __name__ == "__main__":
    original_mata = "data/originalData/meta_Sports_and_Outdoors.jsonl"
    new_item = "data/sport/itemDescription.jsonl"
    new_csv = "data/sport/data.csv"
    df_processed = pd.read_csv(new_csv)

    df_csv = run(df_processed, original_mata, new_item)
    df_csv.to_csv(new_csv, index=False)

import _1dataFliter, _2itemFilter, _3userFliter
import os

name1 = 'Sports_and_Outdoors'
name2 = 'sport'

original_csv = "data/originalData/" + name1 + ".csv"
original_mata = "data/originalData/meta_" + name1 + ".jsonl"
original_review = "data/originalData/" + name1 + ".jsonl"

new_csv = "data/" + name2 + "/data.csv"
new_item = "data/" + name2 + "/itemDescription.jsonl"
new_review = "data/" + name2 + "/userReview.jsonl"

os.makedirs(os.path.dirname(new_csv), exist_ok=True)
os.makedirs(os.path.dirname(new_item), exist_ok=True)
os.makedirs(os.path.dirname(new_review), exist_ok=True)

df = _1dataFliter.run(original_csv)
df = _2itemFilter.run(df, original_mata, new_item)
df = _3userFliter.run(df, original_review, new_review, new_item, new_csv)

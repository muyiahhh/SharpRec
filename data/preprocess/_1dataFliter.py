import pandas as pd

def run(original_csv):
    df = pd.read_csv(original_csv)
    user_counts = df['user_id'].value_counts()
    df = df[df['user_id'].isin(user_counts[user_counts >= 5].index)]
    return df

if __name__ == "__main__":
    original_csv = "data/originalData/Sports_and_Outdoors.csv"
    new_csv = "data/sport/data.csv"
    df = run(original_csv)
    df.to_csv(new_csv, index=False)

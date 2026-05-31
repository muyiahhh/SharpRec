import os
import pandas as pd
from tqdm import tqdm

def build_cross_domain_sequences_fast(domainA_name, domainB_name, A_data_path, B_data_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    use_cols = ["user_id", "parent_asin", "timestamp"]
    df_A = pd.read_csv(A_data_path, usecols=use_cols)
    df_B = pd.read_csv(B_data_path, usecols=use_cols)

    for df in (df_A, df_B):
        if not pd.api.types.is_integer_dtype(df["timestamp"]):
            df["timestamp"] = df["timestamp"].astype("int64")

    df_A["domain"] = domainA_name
    df_B["domain"] = domainB_name

    users_A = set(df_A["user_id"].unique())
    users_B = set(df_B["user_id"].unique())
    overlap_users = users_A & users_B
    A_only_users = users_A - users_B
    B_only_users = users_B - users_A

    df_A["user_id"] = df_A["user_id"].astype("category")
    df_B["user_id"] = df_B["user_id"].astype("category")

    if overlap_users:
        df_A_overlap = df_A[df_A["user_id"].isin(overlap_users)]
        df_B_overlap = df_B[df_B["user_id"].isin(overlap_users)]
        df_overlap = pd.concat([df_A_overlap, df_B_overlap], ignore_index=True)
        df_overlap = df_overlap.sort_values(["user_id", "timestamp"])
        overlap_path = os.path.join(output_dir, f"{domainA_name}2{domainB_name}_overlap.txt")
        with open(overlap_path, "w", encoding="utf-8") as f_out:
            for user_id, group in tqdm(df_overlap.groupby("user_id", sort=False)):
                asins = group["parent_asin"].tolist()
                ts_list = group["timestamp"].tolist()
                domains = group["domain"].tolist()
                tokens = [f"{asin}|{int(ts)}|{dom}" for asin, ts, dom in zip(asins, ts_list, domains)]
                if tokens:
                    f_out.write(f"{user_id} " + " ".join(tokens) + "\n")

    def save_partial_fast(users, df_domain, name_prefix):
        if not users: return
        df_part = df_domain[df_domain["user_id"].isin(users)].copy()
        df_part = df_part.sort_values(["user_id", "timestamp"])
        path = os.path.join(output_dir, f"{name_prefix}_partial.txt")
        with open(path, "w", encoding="utf-8") as f_out:
            for user_id, group in tqdm(df_part.groupby("user_id", sort=False)):
                asins = group["parent_asin"].tolist()
                ts_list = group["timestamp"].tolist()
                domains = group["domain"].tolist()
                tokens = [f"{asin}|{int(ts)}|{dom}" for asin, ts, dom in zip(asins, ts_list, domains)]
                if tokens:
                    f_out.write(f"{user_id} " + " ".join(tokens) + "\n")

    save_partial_fast(A_only_users, df_A, domainA_name)
    save_partial_fast(B_only_users, df_B, domainB_name)

if __name__ == "__main__":
    domainA_name = "sport"
    domainB_name = "toy"
    base_dir = "data"
    A_data_path = f"{base_dir}/{domainA_name}/data.csv"
    B_data_path = f"{base_dir}/{domainB_name}/data.csv"
    output_dir = f"{base_dir}/{domainA_name}2{domainB_name}"

    build_cross_domain_sequences_fast(
        domainA_name=domainA_name,
        domainB_name=domainB_name,
        A_data_path=A_data_path,
        B_data_path=B_data_path,
        output_dir=output_dir,
    )

import fire
import json
import pandas as pd
import numpy as np
from tqdm import tqdm

def process_data(data_path):
    data = pd.read_csv(data_path)
    product_id = data["Product_Code"].unique()
    product_dist = {id: 0 for id in product_id}
    for i, row in tqdm(data.iterrows()):
        quantity_raw = row["Order_Demand"]
        if quantity_raw.startswith("("):
            quantity = float(quantity_raw[1:-1])
        else:
            quantity = float(quantity_raw)
        if not np.isnan(quantity):
            product_dist[row["Product_Code"]] += quantity
        else:
            print(f"NaN value found in row {i}")
    dist_vals = sorted(product_dist.values(), reverse=True)
    dist_vals = np.array(dist_vals)
    dist_vals /= np.sum(dist_vals)
    with open("package_dist/real_data_dist.json", "w") as f:
        json.dump(list(dist_vals), f)

if __name__ == "__main__":
    fire.Fire(process_data)
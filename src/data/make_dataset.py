import os
import pathlib
import sys
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

def load_data(data_path):
    return pd.read_csv(data_path)

def split_data(df, test_split, seed):
    return train_test_split(df, test_size=test_split, random_state=seed)

def save_data(train, test, output_path):
    pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)
    train.to_csv(os.path.join(output_path, "train.csv"), index=False)
    test.to_csv(os.path.join(output_path, "test.csv"), index=False)

def main():
    curr_dir = pathlib.Path(__file__).resolve().parent  # points to src/data
    home_dir = curr_dir.parent.parent                  # points to insurance/ root

    params_file = os.path.join(home_dir, "params.yaml")
    with open(params_file, "r") as f:
        params = yaml.safe_load(f)["make_dataset"]

    input_file = sys.argv[1] if len(sys.argv) > 1 else os.path.join(home_dir, "data", "raw", "insurance_fast.csv")
    output_path = os.path.join(home_dir, "data", "processed")

    data = load_data(input_file)
    train_data, test_data = split_data(data, params["test_split"], params["seed"])
    save_data(train_data, test_data, output_path)
    print(f"Data split complete. Processed files written to {output_path}")
if __name__ == "__main__":
    main()
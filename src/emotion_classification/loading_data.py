import pandas as pd
import os

from .config import data_path

def main():

    os.makedirs(data_path, exist_ok=True)

    # Loading and saving data from Hugging face datasets
    splits = {'train': 'split/train-00000-of-00001.parquet',
              'validation': 'split/validation-00000-of-00001.parquet',
              'test': 'split/test-00000-of-00001.parquet'}

    train_df = pd.read_parquet("hf://datasets/dair-ai/emotion/" + splits["train"])
    train_df.to_csv(os.path.join(data_path,"train_df.csv"), index=False)

    val_df = pd.read_parquet("hf://datasets/dair-ai/emotion/" + splits["validation"])
    val_df.to_csv(os.path.join(data_path,"val_df.csv"), index=False)

    test_df = pd.read_parquet("hf://datasets/dair-ai/emotion/" + splits["test"])
    test_df.to_csv(os.path.join(data_path,"test_df.csv"), index=False)


if __name__ == "__main__":
    main()
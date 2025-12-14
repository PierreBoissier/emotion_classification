import pandas as pd

# Loading data from Hugging face datasets
splits = {'train': 'split/train-00000-of-00001.parquet',
          'validation': 'split/validation-00000-of-00001.parquet',
          'test': 'split/test-00000-of-00001.parquet'}

train_df = pd.read_parquet("hf://datasets/dair-ai/emotion/" + splits["train"])
val_df = pd.read_parquet("hf://datasets/dair-ai/emotion/" + splits["validation"])
test_df = pd.read_parquet("hf://datasets/dair-ai/emotion/" + splits["test"])
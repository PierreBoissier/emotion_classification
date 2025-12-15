import pandas as pd

def main():

    # Loading and saving data from Hugging face datasets
    splits = {'train': 'split/train-00000-of-00001.parquet',
              'validation': 'split/validation-00000-of-00001.parquet',
              'test': 'split/test-00000-of-00001.parquet'}

    train_df = pd.read_parquet("hf://datasets/dair-ai/emotion/" + splits["train"])
    train_df.to_csv("/Users/pierre/emotion_classification/data/train_df.csv", index=False)

    val_df = pd.read_parquet("hf://datasets/dair-ai/emotion/" + splits["validation"])
    val_df.to_csv("/Users/pierre/emotion_classification/data/val_df.csv", index=False)

    test_df = pd.read_parquet("hf://datasets/dair-ai/emotion/" + splits["test"])
    test_df.to_csv("/Users/pierre/emotion_classification/data/test_df.csv", index=False)


if __name__ == "__main__":
    main()
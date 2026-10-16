import pandas as pd
import os
from torch.utils.data import DataLoader
from .dataset_builder import DatasetBuilder
from .config import data_path, model_config


def load_data(df_name='train_df.csv', data_dir=data_path):
    return pd.read_csv(os.path.join(data_dir, df_name))

def create_dataloaders(batch_size, return_dataset=False, evaluation=False,
                       data_dir=data_path, model_name_or_path=None):
    model_name_or_path = model_name_or_path or model_config['model_name_or_path']
    train_dataset, val_dataset, test_dataset = None, None, None
    train_dataloader, val_dataloader, test_dataloader = None, None, None

    if evaluation:
        test_df = load_data('test_df.csv', data_dir)
        test_dataset = DatasetBuilder(test_df, model_name_or_path)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    else:
        train_df = load_data('train_df.csv', data_dir)
        val_df = load_data('val_df.csv', data_dir)
        train_dataset = DatasetBuilder(train_df, model_name_or_path)
        val_dataset = DatasetBuilder(val_df, model_name_or_path)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    if return_dataset :
        return train_dataset, val_dataset, test_dataset
    else:
        return train_dataloader, val_dataloader, test_dataloader

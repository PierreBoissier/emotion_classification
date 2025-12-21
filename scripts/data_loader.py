import pandas as pd
import os
from torch.utils.data import DataLoader
from scripts.dataset_builder import DatasetBuilder
from config import data_path, model_config

def load_data():
    train_df = pd.read_csv(os.path.join(data_path, 'train_df.csv'))
    val_df = pd.read_csv(os.path.join(data_path, 'val_df.csv'))
    test_df = pd.read_csv(os.path.join(data_path, 'test_df.csv'))
    return train_df, val_df, test_df

def create_dataloaders(train_df, val_df, test_df, batch_size, dataset=False):
    train_dataset = DatasetBuilder(train_df, model_config['model_name_or_path'])
    val_dataset = DatasetBuilder(val_df, model_config['model_name_or_path'])
    test_dataset = DatasetBuilder(test_df, model_config['model_name_or_path'])

    if dataset :
        return train_dataset, val_dataset, test_dataset

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_dataloader, val_dataloader, test_dataloader

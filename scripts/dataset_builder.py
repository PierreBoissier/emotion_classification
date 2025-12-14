import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

# Need to be adapted to current data

class DatasetBuilder(Dataset):
    def __init__(self, df, model_name_or_path):
        super().__init__()
        self.all_labels = self.get_all_labels(df)
        self.label2id = {l:i for i,l in enumerate(self.all_labels)}
        df['label_int'] = df['label'].map(self.label2id)
        self.texts = df['sentence'].values
        self.labels = df['label_int'].values
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.max_length = 256

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])
        encoding = self.tokenizer(text,padding='max_length',truncation=True,max_length=self.max_length,return_tensors='pt')
        return {'input_ids':encoding['input_ids'].squeeze(0),
                'attention_mask':encoding['attention_mask'].squeeze(0),
                'labels':torch.tensor(label, dtype=torch.long)}

    @staticmethod
    def get_all_labels(df):
        set_of_labels = set()
        for _, row in df.iterrows():
            if not pd.isna(row['label']):
                set_of_labels.add(row['label'])
        final_labels = sorted(list(set_of_labels))
        return final_labels
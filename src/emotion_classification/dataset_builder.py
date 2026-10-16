import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

class DatasetBuilder(Dataset):
    def __init__(self, df, model_name_or_path):
        super().__init__()
        self.texts = df['text'].values
        self.labels = df['label'].values
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.max_length = 256     # To be adapted to model

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])
        encoding = self.tokenizer(text,padding='max_length',truncation=True,max_length=self.max_length,return_tensors='pt')
        return {'input_ids':encoding['input_ids'].squeeze(0),
                'attention_mask':encoding['attention_mask'].squeeze(0),
                'labels':torch.tensor(label, dtype=torch.long)}
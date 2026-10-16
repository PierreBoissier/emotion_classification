from torch.utils.data import Dataset

class DatasetBuilder(Dataset):
    """Tokenised texts and Labels of one Split. Texts are not padded: batches are padded by the collator."""
    def __init__(self, df, tokenizer, max_length=256):
        super().__init__()
        self.texts = df['text'].values
        self.labels = df['label'].values
        self.tokenizer = tokenizer
        self.max_length = max_length     # To be adapted to model

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])
        encoding = self.tokenizer(text, truncation=True, max_length=self.max_length)
        return {'input_ids': encoding['input_ids'],
                'attention_mask': encoding['attention_mask'],
                'labels': label}

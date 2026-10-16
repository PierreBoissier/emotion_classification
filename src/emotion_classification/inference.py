import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .config import max_length
from .device import select_device

class BertClassifierInference:
    def __init__(self, model_path):
        self.model_path = model_path
        self.device = select_device()
        self.model, self.tokenizer = self._load_model_and_tokenizer()

    def _load_model_and_tokenizer(self):
        model = AutoModelForSequenceClassification.from_pretrained(self.model_path).to(self.device)
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        return model, tokenizer

    def _logits(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length).to(self.device)
        with torch.no_grad():
            return self.model(**inputs).logits

    def predict(self, text):
        preds = torch.argmax(self._logits(text), dim=1)
        return preds.item()

    def predict_proba(self, text):
        probs = torch.nn.functional.softmax(self._logits(text), dim=-1)
        return probs.cpu().numpy()

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import platform

class BertClassifierInference:
    def __init__(self, model_path):
        self.model_path = model_path
        self.device = self._set_device()
        self.model, self.tokenizer = self._load_model_and_tokenizer()

    def _set_device(self):
        cuda_available = torch.cuda.is_available()
        mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available() and \
                        platform.processor() in {"arm", "arm64"}
        device = torch.device("cuda" if cuda_available else "mps" if mps_available else "cpu")
        print(f"Device: {device}")
        return device

    def _load_model_and_tokenizer(self):
        model = AutoModelForSequenceClassification.from_pretrained(self.model_path).to(self.device)
        tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        return model, tokenizer

    def predict(self, text):
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        logits = outputs.logits
        preds = torch.argmax(logits, dim=1)
        return preds.item()

    def predict_proba(self, text):
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        return probs.cpu().detach().numpy()


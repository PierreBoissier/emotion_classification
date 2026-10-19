---
language: en
license: mit
library_name: transformers
pipeline_tag: text-classification
base_model: sentence-transformers/paraphrase-albert-small-v2
datasets:
  - dair-ai/emotion
metrics:
  - accuracy
  - f1
tags:
  - emotion-classification
  - albert
---

# paraphrase-albert-small-v2 fine-tuned for emotion classification

A small ALBERT-based Classifier that assigns one of six Emotions to a short English text: sadness, joy, love, anger, fear or surprise.
It is [`sentence-transformers/paraphrase-albert-small-v2`](https://huggingface.co/sentence-transformers/paraphrase-albert-small-v2) fine-tuned on [`dair-ai/emotion`](https://huggingface.co/datasets/dair-ai/emotion).

Code, training notebook and API: https://github.com/PierreBoissier/emotion_classification

## Usage

```python
from transformers import pipeline

classifier = pipeline("text-classification", model="{{REPO_ID}}", top_k=None)
classifier("i am so happy to see you again today")
```

Texts longer than 256 tokens are truncated.

## Results

Evaluated on the 2,000-text test Split of `dair-ai/emotion`:

| Metric | Value |
|---|---|
| Accuracy | 0.921 |
| Macro F1 | 0.875 |

| Emotion | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| sadness | 0.960 | 0.952 | 0.956 | 581 |
| joy | 0.937 | 0.948 | 0.943 | 695 |
| love | 0.813 | 0.792 | 0.803 | 159 |
| anger | 0.911 | 0.931 | 0.921 | 275 |
| fear | 0.869 | 0.920 | 0.894 | 224 |
| surprise | 0.875 | 0.636 | 0.737 | 66 |

![Confusion matrix on the test Split](confusion_matrix.png)

The weakest Emotion is surprise (recall 0.636), mostly mistaken for fear. Love is often mistaken for joy.

## Training

- Data: the `dair-ai/emotion` train Split (16,000 texts) for training; the validation Split (2,000 texts) is only used to monitor accuracy, not to train or select weights.
- 10 epochs, learning rate 2e-5, weight decay 0.01, batch size 32 with 4 gradient accumulation steps, AdamW with a linear schedule (5% warmup), mixed precision, on a Colab GPU.
- The weights of the last epoch are kept. Validation accuracy reaches about 0.93 and is flat over the last epochs.

![Training curves](training_curves.png)

## Limitations

- English only, and trained on short texts: the dataset is made of English tweets labelled from hashtags, so labels are noisy and the Emotions reflect that style of writing.
- Classes are imbalanced (surprise and love are rare), which explains their lower scores.
- Each text gets exactly one Emotion; mixed or neutral feelings are not modelled.
- Not intended for decisions about people: it is a portfolio project, not a validated tool.

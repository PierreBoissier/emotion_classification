# Emotion Classification

Fine-tuning and serving of a text classifier that assigns one of six emotions to short English texts, using the `dair-ai/emotion` dataset.

## Language

**Emotion**:
One of the six categories a text can express: sadness, joy, love, anger, fear, surprise.
_Avoid_: Class name, category, sentiment

**Label**:
The integer id (0-5) of an emotion in the dataset, mapped to its emotion name by `id2label`.
_Avoid_: Target, class

**Split**:
One of the three dataset partitions: train, validation, test.
_Avoid_: Fold, subset

**Classifier**:
A pretrained sentence-embedding model fine-tuned to predict the emotion of a text.
_Avoid_: BERT model (the base model is ALBERT-based, not BERT)

**Checkpoint**:
The saved weights and tokenizer of a fine-tuned classifier, published on the Hugging Face Hub.
_Avoid_: Saved model, trained model

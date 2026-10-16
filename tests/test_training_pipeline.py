import json

import matplotlib
import pandas as pd
import pytest
import torch
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          BertConfig, BertModel, BertTokenizer)

from emotion_classification.config import id2label
from emotion_classification.main import main

matplotlib.use("Agg")

WORDS = ["i", "feel", "so", "happy", "sad", "angry", "scared", "loved", "surprised", "today"]


@pytest.fixture
def tiny_model_dir(tmp_path):
    """A randomly initialised BERT with a 15-word vocabulary, usable offline."""
    model_dir = tmp_path / "tiny-base-model"
    model_dir.mkdir()
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + WORDS
    (model_dir / "vocab.txt").write_text("\n".join(vocab))
    BertTokenizer(str(model_dir / "vocab.txt")).save_pretrained(model_dir)
    config = BertConfig(vocab_size=len(vocab), hidden_size=16, num_hidden_layers=1,
                        num_attention_heads=2, intermediate_size=32, max_position_embeddings=256)
    BertModel(config).save_pretrained(model_dir)
    return str(model_dir)


@pytest.fixture
def tiny_data_dir(tmp_path):
    """Train/validation/test Splits of 12 rows, two per Emotion."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    rows = [{"text": f"i feel so {WORDS[3 + label % 6]} today", "label": label}
            for label in range(6) for _ in range(2)]
    for split in ("train", "val", "test"):
        pd.DataFrame(rows).to_csv(data_dir / f"{split}_df.csv", index=False)
    return str(data_dir)


@pytest.mark.slow
def test_training_run_produces_reloadable_checkpoint_and_evaluation_files(
        tiny_model_dir, tiny_data_dir, tmp_path):
    output_dir = tmp_path / "checkpoint"

    main(config={"model_name_or_path": tiny_model_dir, "batch_size": 2, "num_epochs": 1,
                 "mps": False, "load_trained_model": False},
         data_dir=tiny_data_dir, output_dir=str(output_dir))

    # The Checkpoint can be reloaded and predicts an Emotion
    model = AutoModelForSequenceClassification.from_pretrained(output_dir)
    tokenizer = AutoTokenizer.from_pretrained(output_dir)
    logits = model(**tokenizer("i feel so happy today", return_tensors="pt")).logits
    assert model.config.id2label == id2label
    assert logits.shape == (1, len(id2label))
    assert int(torch.argmax(logits)) in id2label

    # Metrics and confusion matrix are saved as files
    metrics = json.loads((output_dir / "metrics.json").read_text())
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["macro_f1"] <= 1.0
    assert set(metrics["per_emotion"]) == set(id2label.values())
    assert (output_dir / "confusion_matrix.png").stat().st_size > 0

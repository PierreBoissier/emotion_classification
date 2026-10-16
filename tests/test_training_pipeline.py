import json

import pytest
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from emotion_classification.config import id2label
from emotion_classification.main import main

from .conftest import TINY_CONFIG


@pytest.mark.slow
def test_training_run_produces_reloadable_checkpoint_and_evaluation_files(
        tiny_model_dir, tiny_data_dir, tmp_path):
    output_dir = tmp_path / "checkpoint"

    returned = main(config={**TINY_CONFIG, "model_name_or_path": tiny_model_dir},
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
    assert returned == metrics
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert metrics["macro_f1"] == pytest.approx(
        sum(m["f1-score"] for m in metrics["per_emotion"].values()) / len(id2label))
    assert len(metrics["confusion_matrix"]) == len(id2label)
    assert set(metrics["per_emotion"]) == set(id2label.values())
    assert (output_dir / "confusion_matrix.png").stat().st_size > 0

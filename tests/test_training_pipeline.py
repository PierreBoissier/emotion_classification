import json
from pathlib import Path

import pytest
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from emotion_classification.config import id2label
from emotion_classification.evaluate import cli
from emotion_classification.main import evaluate_checkpoint, main

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


def test_evaluating_a_checkpoint_writes_results_without_modifying_it(
        trained_checkpoint, tiny_data_dir, tmp_path):
    results_dir = tmp_path / "results"
    weights = Path(trained_checkpoint) / "model.safetensors"
    weights_before = weights.read_bytes()

    metrics = evaluate_checkpoint(trained_checkpoint, data_dir=tiny_data_dir,
                                  results_dir=str(results_dir), config=TINY_CONFIG)

    assert json.loads((results_dir / "metrics.json").read_text()) == metrics
    assert set(metrics["per_emotion"]) == set(id2label.values())
    assert (results_dir / "confusion_matrix.png").stat().st_size > 0
    assert weights.read_bytes() == weights_before


def test_evaluate_command_line_writes_results(trained_checkpoint, tiny_data_dir, tmp_path):
    results_dir = tmp_path / "results"

    cli([trained_checkpoint, "--data-dir", tiny_data_dir, "--results-dir", str(results_dir)])

    assert set(json.loads((results_dir / "metrics.json").read_text())["per_emotion"]) == set(id2label.values())


def test_evaluating_a_hub_checkpoint_requires_a_results_directory(tiny_data_dir):
    with pytest.raises(ValueError, match="results_dir"):
        evaluate_checkpoint("someone/not-a-local-directory", data_dir=tiny_data_dir)

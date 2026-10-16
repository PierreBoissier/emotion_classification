import matplotlib
import pandas as pd
import pytest
from transformers import BertConfig, BertModel, BertTokenizer

from emotion_classification.main import main

matplotlib.use("Agg")

WORDS = ["i", "feel", "so", "happy", "sad", "angry", "scared", "loved", "surprised", "today"]


def build_tiny_model_dir(root):
    """A randomly initialised BERT with a 15-word vocabulary, usable offline."""
    model_dir = root / "tiny-base-model"
    model_dir.mkdir()
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + WORDS
    (model_dir / "vocab.txt").write_text("\n".join(vocab))
    BertTokenizer(str(model_dir / "vocab.txt")).save_pretrained(model_dir)
    config = BertConfig(vocab_size=len(vocab), hidden_size=16, num_hidden_layers=1,
                        num_attention_heads=2, intermediate_size=32, max_position_embeddings=256)
    BertModel(config).save_pretrained(model_dir)
    return str(model_dir)


def build_tiny_data_dir(root):
    """Train/validation/test Splits of 12 rows, two per Emotion."""
    data_dir = root / "data"
    data_dir.mkdir()
    rows = [{"text": f"i feel so {WORDS[3 + label % 6]} today", "label": label}
            for label in range(6) for _ in range(2)]
    for split in ("train", "val", "test"):
        pd.DataFrame(rows).to_csv(data_dir / f"{split}_df.csv", index=False)
    return str(data_dir)


TINY_CONFIG = {"batch_size": 2, "num_epochs": 1, "mps": False, "load_trained_model": False}


@pytest.fixture
def tiny_model_dir(tmp_path):
    return build_tiny_model_dir(tmp_path)


@pytest.fixture
def tiny_data_dir(tmp_path):
    return build_tiny_data_dir(tmp_path)


@pytest.fixture(scope="session")
def trained_checkpoint(tmp_path_factory):
    """A Checkpoint produced by one tiny training run, shared by the whole test session."""
    root = tmp_path_factory.mktemp("trained")
    output_dir = root / "checkpoint"
    main(config={**TINY_CONFIG, "model_name_or_path": build_tiny_model_dir(root)},
         data_dir=build_tiny_data_dir(root), output_dir=str(output_dir))
    return str(output_dir)

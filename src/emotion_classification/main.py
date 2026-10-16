from .config import model_config, data_path
from .trainer import BertClassifier


def main(config=None, data_dir=data_path, output_dir=None):
    """Train a Classifier, evaluate it on the test Split and save the Checkpoint.

    `config` overrides entries of `model_config`. The Checkpoint, `metrics.json`
    and `confusion_matrix.png` are written to `output_dir` (default: under `models/`).
    Returns the metrics dictionary.
    """
    config = {**model_config, **(config or {})}

    # Initialise the BERT classifier
    bert_classifier = BertClassifier(
        model_name_or_path=config['model_name_or_path'],
        batch_size=config['batch_size'],
        mps=config['mps'],
        load_trained_model=config['load_trained_model'],
        data_dir=data_dir,
        output_dir=output_dir
    )

    # Train the model
    bert_classifier.train(
        num_epochs=config['num_epochs'],
        lr=config['lr'],
        weight_decay=config['weight_decay']
    )

    # Evaluate the model
    return bert_classifier.evaluate()


if __name__ == '__main__':
    main()

import argparse

from .config import data_path
from .main import evaluate_checkpoint


def cli(argv=None):
    """Evaluate a trained Checkpoint on the test Split.

    Example: python -m emotion_classification.evaluate models/paraphrase-albert-small-v2 --results-dir results
    The test Split is read from `data/test_df.csv` (see `python -m emotion_classification.loading_data`).
    """
    parser = argparse.ArgumentParser(description=cli.__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint_dir", help="local directory of the Checkpoint to evaluate")
    parser.add_argument("--data-dir", default=data_path, help="directory containing test_df.csv")
    parser.add_argument("--results-dir", default=None,
                        help="where to write metrics.json and confusion_matrix.png (default: the Checkpoint directory)")
    args = parser.parse_args(argv)
    return evaluate_checkpoint(args.checkpoint_dir, data_dir=args.data_dir, results_dir=args.results_dir)


if __name__ == "__main__":
    cli()

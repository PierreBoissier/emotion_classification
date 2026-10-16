from .config import model_config
from .trainer import BertClassifier


def main():
    # Initialise the BERT classifier
    bert_classifier = BertClassifier(
        model_name_or_path=model_config['model_name_or_path'],
        batch_size=model_config['batch_size'],
        mps=model_config['mps'],
        load_trained_model=model_config['load_trained_model']
    )

    # Train the model
    bert_classifier.train(
        num_epochs=model_config['num_epochs'],
        lr=model_config['lr'],
        weight_decay=model_config['weight_decay']
    )

    # Evaluate the model
    bert_classifier.evaluate()

if __name__ == '__main__':
    main()

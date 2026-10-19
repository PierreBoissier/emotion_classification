import os

# Path to store / load the model and the data
path = os.getcwd()
data_path = path + '/data/'
models_path = path + '/models/'

# Where training saves the Checkpoint
default_checkpoint = os.path.join(models_path, 'paraphrase-albert-small-v2')

# Checkpoint served by the API when none is configured (Hugging Face Hub id)
hub_checkpoint = 'pboissie/paraphrase-albert-small-v2-emotion'

# Model's configuration
model_config = {
    'model_name_or_path': 'sentence-transformers/paraphrase-albert-small-v2',
    'batch_size': 32,
    'mps': False,
    'load_trained_model': False,
    'num_epochs': 10,
    'lr': 2e-5,
    'weight_decay': 0.01
}

# Maximum number of tokens per text, used for truncation in training and inference
max_length = 256

# Dataset labels description
id2label = {
    0 : "sadness",
    1 : "joy",
    2 : "love",
    3 : "anger",
    4 : "fear",
    5 : "surprise"
}

label2id = {l: i for i, l in id2label.items()}

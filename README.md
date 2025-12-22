# Text Classification with BERT

This project aims to train a text classification model using BERT (Bidirectional Encoder Representations from Transformers) to classify texts into different categories.

## Project Structure

```
emotion_classification/
│
├── scripts/
│   ├── config.py
│   ├── data_loader.py
│   ├── loading_data.py
│   ├── main.py
│   ├── trainer.py
│   └── inference.py
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── logs/
│
├── models/
│   └── saved_models/
│
├── notebooks/
│   └── training_notebook.ipynb
│
├── README.md
└── requirements.txt
```

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/votre-utilisateur/emotion_classification.git
   cd emotion_classification
   ```

2. **Create a virtual environment** (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Configuration

Before running the project, make sure to configure the parameters in `scripts/config.py`. Here is an example configuration:

```python
# scripts/config.py
data_path = 'data/processed/'
models_path = 'models/saved_models/'
model_config = {
    'model_name_or_path': 'almanach/camembertv2-base',
    'batch_size': 64,
    'mps': False,
    'load_trained_model': False,
    'num_epochs': 3,
    'lr': 2e-5,
    'weight_decay': 0.01
}
```

## Running Training

To run training with the provided notebook, follow these steps:

1. **Prepare your data**:
   - Place your data in the `data/raw/` folder.
   - Run the data loading script to process it and place it in `data/processed/`.

2. **Run the training notebook**:
   - Open the notebook `notebooks/training_notebook.ipynb`.
   - Follow the instructions in the notebook to execute training.

## Files to Modify for Training Execution

- `scripts/config.py`: Configure model parameters and file paths.
- `scripts/data_loader.py`: Modify data loading functions if necessary.
- `scripts/loading_data.py`: Adjust data loading as needed.
- `scripts/main.py`: Main entry point for training.
- `scripts/trainer.py`: Contains functions to train and evaluate the model.

## Using the API

To use the text classification API, follow these steps:

1. **Start the API**:
   ```bash
   uvicorn scripts.deploy:app --reload
   ```

2. **Send requests to the API**:
   - Use a tool like `curl` or Postman to send requests to the API.
   - Example request:
   ```bash
   curl -X POST "http://127.0.0.1:8000/predict" -H "Content-Type: application/json" -d '{"text":"Example text to classify"}'
   ```

## Deployment

To deploy the application, you can use Docker. Here is an example Dockerfile:

```dockerfile
FROM python:3.8-slim

WORKDIR /app
COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8000
CMD ["uvicorn", "scripts.deploy:app", "--host", "0.0.0.0", "--port", "8000"]
```

1. **Build the Docker image**:
   ```bash
   docker build -t emotion-classification .
   ```

2. **Run the Docker container**:
   ```bash
   docker run -p 8000:8000 emotion-classification
   ```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or fixes.

## License
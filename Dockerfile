FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN useradd --create-home app
WORKDIR /app

# CPU-only PyTorch keeps the image small (the API runs inference on CPU)
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install .

# Checkpoint served by the API: a Hugging Face Hub id (override with --build-arg or -e EMOTION_CHECKPOINT=...)
ARG CHECKPOINT=pboissie/paraphrase-albert-small-v2-emotion
ENV EMOTION_CHECKPOINT=${CHECKPOINT}

USER app
# Download the Checkpoint at build time so the container starts without waiting for the Hub
RUN python -c "import os; from huggingface_hub import snapshot_download; snapshot_download(os.environ['EMOTION_CHECKPOINT'])"

EXPOSE 8000
CMD ["uvicorn", "emotion_classification.api:app", "--host", "0.0.0.0", "--port", "8000"]

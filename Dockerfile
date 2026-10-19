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

USER app
# Download the Checkpoint served by default (config.hub_checkpoint) at build time, so the container
# starts without waiting for the Hub. Serve another one with -e EMOTION_CHECKPOINT=<Hub id or mounted path>.
RUN python -c "from huggingface_hub import snapshot_download; from emotion_classification.config import hub_checkpoint; snapshot_download(hub_checkpoint)"

EXPOSE 8000
CMD ["uvicorn", "emotion_classification.api:app", "--host", "0.0.0.0", "--port", "8000"]

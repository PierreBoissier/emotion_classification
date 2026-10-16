import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel

from .config import default_checkpoint, id2label
from .inference import BertClassifierInference


class TextRequest(BaseModel):
    text: str


def create_app(checkpoint=None):
    """Build the API serving a Checkpoint: a local path or a Hugging Face Hub id.

    Without argument, the Checkpoint is read from the EMOTION_CHECKPOINT environment
    variable, then falls back to the default Checkpoint under `models/`.
    The model is loaded when the application starts.
    """
    @asynccontextmanager
    async def lifespan(app):
        checkpoint_path = checkpoint or os.environ.get("EMOTION_CHECKPOINT") or default_checkpoint
        app.state.inference = BertClassifierInference(checkpoint_path)
        yield

    app = FastAPI(lifespan=lifespan)

    @app.post("/predict")
    async def predict(text_request: TextRequest):
        prediction = app.state.inference.predict(text_request.text)
        return {"prediction": id2label[prediction]}

    @app.post("/predict_proba")
    async def predict_proba(text_request: TextRequest):
        probabilities = app.state.inference.predict_proba(text_request.text)[0]
        return {"probabilities": {id2label[i]: float(p) for i, p in enumerate(probabilities)}}

    return app


app = create_app()

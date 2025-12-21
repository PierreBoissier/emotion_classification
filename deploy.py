from fastapi import FastAPI
from pydantic import BaseModel
from scripts.inference import BertClassifierInference
from scripts.config import models_path, id2label

app = FastAPI()

class TextRequest(BaseModel):
    text: str

# Initialize the model when the application starts
model_path = f"{models_path}paraphrase-albert-small-v2"
inference = BertClassifierInference(model_path)

@app.post("/predict")
async def predict(text_request: TextRequest):
    prediction = inference.predict(text_request.text)
    return {"prediction": id2label[prediction]}

@app.post("/predict_proba")
async def predict_proba(text_request: TextRequest):
    probabilities = inference.predict_proba(text_request.text)
    probabilities = probabilities.tolist()
    probabilities = probabilities[0]

    probabilities_dict = {id2label[0] : probabilities[0],
                          id2label[1] : probabilities[1],
                          id2label[2] : probabilities[2],
                          id2label[3] : probabilities[3],
                          id2label[4] : probabilities[4],
                          id2label[5] : probabilities[5],}
    return {"probabilities": probabilities_dict}

# To execute API :
# uvicorn deploy:app --reload

# To see port 8000 command
# lsof -i :8000

# To kill port that interfere with API execution
# kill -9 <PID>

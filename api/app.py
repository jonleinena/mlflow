from fastapi import FastAPI
import mlflow.pytorch
import torch
import os

app = FastAPI()

mlflow.set_tracking_uri(os.environ(["MLFLOW_TRACKING_URI"]))
model_uri = "models:/MyModel/Production"
model = mlflow.pytorch.load_model(model_uri)

@app.get("/predict")
async def predict(data: list):
    tensor_data = torch.tensor(data)
    with torch.no_grad():
        prediction = model(tensor_data)
    return {"prediction": prediction.tolist()}
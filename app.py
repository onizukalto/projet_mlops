from fastapi import FastAPI
import mlflow.pyfunc
import os

MODEL_URI = os.getenv("MODEL_URI", "models:/Iris@champion")
model = mlflow.pyfunc.load_model(MODEL_URI)

app = FastAPI()

ESPECES = ["setosa", "versicolor", "virginica"]

@app.post("/predict")
def predict(features: list[float]):
    prediction = model.predict([features])[0]
    return {"espece": ESPECES[int(prediction)]}
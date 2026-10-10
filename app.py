import os

import mlflow.pyfunc
from fastapi import FastAPI, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest,
)

MODEL_URI = os.getenv("MODEL_URI", "models:/Iris@champion")
model = mlflow.pyfunc.load_model(MODEL_URI)

app = FastAPI()

ESPECES = ["setosa", "versicolor", "virginica"]

# Plage (min, max) de chaque mesure dans les données d'entraînement Iris
PLAGES = [(4.3, 7.9), (2.0, 4.4), (1.0, 6.9), (0.1, 2.5)]

# --- Les métriques ---
PREDICTIONS = Counter(
    "iris_predictions_total", "Nombre de prédictions par espèce", ["espece"]
)
LATENCE = Histogram(
    "iris_prediction_seconds", "Temps de calcul d'une prédiction"
)
HORS_PLAGE = Counter(
    "iris_inputs_hors_plage_total", "Requêtes avec une mesure jamais vue à l'entraînement"
)


@app.post("/predict")
def predict(features: list[float]):
    # Détection simple de dérive : une mesure sort-elle de la plage connue ?
    if any(not (mn <= x <= mx) for x, (mn, mx) in zip(features, PLAGES)):
        HORS_PLAGE.inc()

    with LATENCE.time():                     # chronomètre la prédiction
        prediction = model.predict([features])[0]

    espece = ESPECES[int(prediction)]
    PREDICTIONS.labels(espece=espece).inc()  # +1 pour cette espèce
    return {"espece": espece}


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
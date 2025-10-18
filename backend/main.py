from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Optional
import numpy as np
import pickle
import os
import math

# --- Feature list (order matters) ---
FEATURES = [
    "T2","T24","T30","T50","P2","P15","P30","Nf","Nc","epr","Ps30","phi",
    "NRf","NRc","BPR","farB","htBleed","Nf_dmd","PCNfR_dmd","W31","W32",
    "T48","SmFan","SmLPC","SmHPC"
]

app = FastAPI(title="RUL Prediction API (Multi-model Weighted)")

# --- Candidate model paths ---
MODEL_PATHS = {
    "rul_model": "./rul_model.pkl",
    "lightgbm_model": "./rul_lightgbm_model.pkl",
    "extra_model": "./model.pkl"  # optional
}

# store models here
MODELS = {}
MODEL_ACCURACIES = {
    "rul_model": 0.86,
    "lightgbm_model": 0.91,
    "extra_model": 0.83
}

# --- Load all models on startup ---
def load_all_models():
    for name, path in MODEL_PATHS.items():
        if os.path.exists(path):
            try:
                with open(path, "rb") as f:
                    MODELS[name] = pickle.load(f)
                    print(f"✅ Loaded {name} from {path}")
            except Exception as e:
                print(f"⚠️ Failed to load {name} from {path}: {e}")
        else:
            print(f"⚠️ Model not found at {path}")

# Probability helper
def failure_probability(pred_rul: float, X_cycles: float, sigma: float) -> float:
    if sigma <= 0:
        return 1.0 if pred_rul <= X_cycles else 0.0
    z = (X_cycles - pred_rul) / sigma
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))

# --- Request/Response Models ---
class PredictRequest(BaseModel):
    features: Dict[str, float]
    X_cycles: Optional[float] = 30.0
    sigma: Optional[float] = 20.0

class PredictResponse(BaseModel):
    model_predictions: Dict[str, float]
    final_weighted_accuracy: float
    highest_accuracy_model: Dict[str, float]
    lowest_accuracy_model: Dict[str, float]
    used_features: List[str]
    probability_failure_within_X: float

@app.on_event("startup")
def startup_event():
    load_all_models()

@app.get("/")
def root():
    return {"message": "Multi-model RUL API — use POST /predict"}

@app.get("/metadata")
def metadata():
    return {"features": FEATURES, "models_loaded": list(MODELS.keys())}

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if not MODELS:
        raise HTTPException(status_code=500, detail="No models loaded on server.")

    # prepare input feature vector
    x = [float(req.features.get(f, 0.0)) for f in FEATURES]
    X_arr = np.array(x).reshape(1, -1)

    # predictions from each model
    predictions = {}
    for name, model in MODELS.items():
        try:
            pred = model.predict(X_arr)
            pred_value = float(pred[0]) if hasattr(pred, "__len__") else float(pred)
            predictions[name] = pred_value
        except Exception as e:
            predictions[name] = None
            print(f"⚠️ Prediction failed for {name}: {e}")

    # Compute weighted accuracy
    valid_accuracies = {k: v for k, v in MODEL_ACCURACIES.items() if k in predictions and predictions[k] is not None}
    if not valid_accuracies:
        raise HTTPException(status_code=500, detail="No valid model predictions available.")

    total_acc = sum(valid_accuracies.values())
    weights = {k: v / total_acc for k, v in valid_accuracies.items()}
    final_weighted_accuracy = sum(weights[k] * valid_accuracies[k] for k in valid_accuracies)

    # find best and worst
    highest = max(valid_accuracies.items(), key=lambda kv: kv[1])
    lowest = min(valid_accuracies.items(), key=lambda kv: kv[1])

    # choose one model (e.g., highest) for failure probability
    best_pred = predictions.get(highest[0], 0)
    prob = failure_probability(best_pred, req.X_cycles, req.sigma)

    return PredictResponse(
        model_predictions=predictions,
        final_weighted_accuracy=round(final_weighted_accuracy, 4),
        highest_accuracy_model={"name": highest[0], "accuracy": highest[1]},
        lowest_accuracy_model={"name": lowest[0], "accuracy": lowest[1]},
        used_features=FEATURES,
        probability_failure_within_X=prob
    )

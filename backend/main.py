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

MODELS: Dict[str, object] = {}
MODEL_ACCURACIES = {
    "rul_model": 0.86,
    "lightgbm_model": 0.91,
    "extra_model": 0.83
}

# --- Load all models ---
def load_all_models():
    loaded = []
    for name, path in MODEL_PATHS.items():
        if os.path.exists(path):
            try:
                with open(path, "rb") as f:
                    MODELS[name] = pickle.load(f)
                    loaded.append(name)
                    print(f"✅ Loaded {name} from {path}")
            except Exception as e:
                print(f"⚠️ Failed to load {name} from {path}: {e}")
        else:
            print(f"⚠️ Model not found at {path}")
    print(f"📦 Models loaded: {loaded or 'None'}")

# --- Probability helper ---
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
    model_predictions: Dict[str, Optional[float]]
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

@app.get("/status")
def status():
    """Check if models are loaded successfully"""
    return {
        "models_loaded": list(MODELS.keys()),
        "model_files_present": {k: os.path.exists(v) for k, v in MODEL_PATHS.items()}
    }

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if not MODELS:
        raise HTTPException(status_code=500, detail="No models loaded on server.")

    # prepare feature vector
    x = [float(req.features.get(f, 0.0)) for f in FEATURES]
    X_arr = np.array(x).reshape(1, -1)

    # collect predictions
    predictions: Dict[str, Optional[float]] = {}
    for name, model in MODELS.items():
        try:
            pred = model.predict(X_arr)
            value = float(pred[0]) if hasattr(pred, "__len__") else float(pred)
            predictions[name] = value
        except Exception as e:
            print(f"⚠️ Prediction failed for {name}: {e}")
            predictions[name] = None

    # filter valid ones
    valid_preds = {k: v for k, v in predictions.items() if v is not None}
    valid_accs = {k: MODEL_ACCURACIES[k] for k in valid_preds.keys() if k in MODEL_ACCURACIES}

    if not valid_preds:
        raise HTTPException(status_code=500, detail="No valid model predictions available.")

    # compute weights
    total_acc = sum(valid_accs.values())
    weights = {k: v / total_acc for k, v in valid_accs.items()}

    # weighted average of accuracies
    final_weighted_accuracy = round(sum(weights[k] * valid_accs[k] for k in valid_accs), 4)

    # best and worst
    highest = max(valid_accs.items(), key=lambda kv: kv[1])
    lowest = min(valid_accs.items(), key=lambda kv: kv[1])

    # probability calculation from best model
    best_pred = valid_preds.get(highest[0], 0.0)
    prob = failure_probability(best_pred, req.X_cycles, req.sigma)

    return PredictResponse(
        model_predictions=predictions,
        final_weighted_accuracy=final_weighted_accuracy,
        highest_accuracy_model={"name": highest[0], "accuracy": highest[1]},
        lowest_accuracy_model={"name": lowest[0], "accuracy": lowest[1]},
        used_features=FEATURES,
        probability_failure_within_X=round(prob, 4)
    )

# main.py bottom
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))

# backend/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Optional
import numpy as np
import pickle
import os
import math

# --- Feature list (order matters) ---
FEATURES = [
    # main features (engine sensors)
    "T2","T24","T30","T50","P2","P15","P30","Nf","Nc","epr","Ps30","phi",
    "NRf","NRc","BPR","farB","htBleed","Nf_dmd","PCNfR_dmd","W31","W32",
    # health-index parameters
    "T48","SmFan","SmLPC","SmHPC"
]

app = FastAPI(title="RUL Prediction API")

MODEL_PATHS = ["/mnt/data/rul_model.pkl", "/mnt/data/model.pkl", "./rul_model.pkl", "./model.pkl"]
_model = None

def load_model():
    global _model
    if _model is not None:
        return _model
    for p in MODEL_PATHS:
        if os.path.exists(p):
            try:
                with open(p, "rb") as f:
                    _model = pickle.load(f)
                    print("Loaded model from", p)
                    return _model
            except Exception as e:
                print("Failed to load model at", p, ":", e)
    print("No model found in candidate paths.")
    return None

# Probability helper using normal residual assumption
def failure_probability(pred_rul: float, X_cycles: float, sigma: float) -> float:
    if sigma <= 0:
        return 1.0 if pred_rul <= X_cycles else 0.0
    z = (X_cycles - pred_rul) / sigma
    # standard normal CDF via erf
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))

# Pydantic request model
class PredictRequest(BaseModel):
    features: Dict[str, float]  # feature name -> value
    X_cycles: Optional[float] = 30.0
    sigma: Optional[float] = 20.0

class PredictResponse(BaseModel):
    predicted_RUL: float
    probability_failure_within_X: float
    used_features: List[str]

@app.on_event("startup")
def startup_event():
    load_model()

@app.get("/metadata")
def metadata():
    return {"features": FEATURES, "notes": "Send JSON POST to /predict with 'features' map."}

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    model = load_model()
    if model is None:
        raise HTTPException(status_code=500, detail="No model loaded on server. Place rul_model.pkl at /mnt/data or model.pkl here.")
    # ensure expected features are provided, fill missing with 0 or raise
    x = []
    missing = []
    for f in FEATURES:
        if f in req.features:
            x.append(float(req.features[f]))
        else:
            # if missing, append 0 and record
            x.append(0.0)
            missing.append(f)
    X_arr = np.array(x, dtype=float).reshape(1, -1)
    try:
        pred = model.predict(X_arr)
        # many sklearn models return array-like
        if hasattr(pred, "__len__"):
            pred_rul = float(pred[0])
        else:
            pred_rul = float(pred)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model prediction failed: {e}")

    prob = failure_probability(pred_rul, req.X_cycles, req.sigma)
    return PredictResponse(predicted_RUL=pred_rul,
                           probability_failure_within_X=prob,
                           used_features=FEATURES)

# Simple root
@app.get("/")
def root():
    return {"message": "RUL Prediction API. GET /metadata or POST /predict"}

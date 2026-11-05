import json
import pickle
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

BASE = Path(__file__).parent

if not (BASE / "model.pkl").exists():
    raise SystemExit("model.pkl not found. Run:  python train.py")

with open(BASE / "model.pkl", "rb") as f:
    model = pickle.load(f)
meta = json.loads((BASE / "metadata.json").read_text())

FEATURES = list(model.feature_names_in_)
scaler = model.named_steps["scale"]
clf = model.named_steps["clf"]

app = FastAPI(title="Tumour Classifier", description="Interactive ML demo")


class PredictRequest(BaseModel):
    values: dict[str, float]


@app.get("/api/info")
def info():
    """Feature ranges (for sliders) and model quality."""
    return meta


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/api/predict")
def predict(req: PredictRequest):
    missing = [f for f in FEATURES if f not in req.values]
    if missing:
        raise HTTPException(422, f"Missing features: {missing}")

    df = pd.DataFrame([{f: req.values[f] for f in FEATURES}])
    proba_benign = float(model.predict_proba(df)[0][1])

    # Explain: scaled value * coefficient = push towards benign (+) / malignant (-)
    scaled = scaler.transform(df)[0]
    contributions = [
        {"feature": f, "effect": round(float(c * s), 3)}
        for f, c, s in zip(FEATURES, clf.coef_[0], scaled)
    ]
    return {
        "label": "Benign" if proba_benign >= 0.5 else "Malignant",
        "probability_benign": round(proba_benign, 4),
        "probability_malignant": round(1 - proba_benign, 4),
        "contributions": contributions,
    }


@app.get("/")
def home():
    return FileResponse(BASE / "static" / "index.html")


app.mount("/static", StaticFiles(directory=BASE / "static"), name="static")

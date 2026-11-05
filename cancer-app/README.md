# Tumour Classifier: interactive ML web app

Trains a logistic regression model on the Breast Cancer Wisconsin dataset (built into scikit-learn) and serves it with FastAPI. Sliders send values to the API and show the prediction live, with each feature's push towards benign or malignant.

**Model:** 6 features (mean radius, texture, area, smoothness, concavity, symmetry), StandardScaler + LogisticRegression.
**Test accuracy:** 89.5% (455 train / 114 test). Confusion matrix `[[38, 4], [8, 64]]` (0 = malignant).

**Live demo:** https://intro-to-ai-lab-works.onrender.com

```
cancer-app/
├── train.py          trains the model, writes model.pkl + metadata.json
├── main.py           FastAPI backend (API + serves the page)
├── requirements.txt
└── static/           index.html, style.css, app.js
```

## Run
```bash
python -m pip install -r requirements.txt
python train.py
python -m uvicorn main:app --reload --port 8000
```

## API
| Method | Path | Purpose |
|---|---|---|
| GET | `/api/info` | Slider ranges and model accuracy |
| POST | `/api/predict` | `{"values": {"mean radius": 14, ...}}` -> label, probabilities, per-feature effects |
| GET | `/api/health` | Health check |

## Deploy on Render
Root directory `cancer-app` · Build `pip install -r requirements.txt && python train.py` · Start `uvicorn main:app --host 0.0.0.0 --port $PORT`

Educational demo only, not for medical use.

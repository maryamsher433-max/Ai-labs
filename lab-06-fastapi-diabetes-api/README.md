# Lab 06: Serving a Model with FastAPI

Loads the trained diabetes model (`Mymdoel.pkl`) and exposes it as a REST API.

| Method | Path | Purpose |
|---|---|---|
| GET | `/` | Status message |
| GET | `/health` | Health check |
| POST | `/predict` | Glucose, BloodPressure, Insulin, BMI, DiabetesPedigreeFunction, Age -> label and probability |

```bash
python -m pip install fastapi uvicorn pandas scikit-learn==1.9.1
python -m uvicorn main:app --reload --port 8000
```
Docs at http://localhost:8000/docs. The finished, deployed version is in [`../diabetes-app`](../diabetes-app/).

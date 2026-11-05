# Introduction to AI: Lab Works

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.9.1-F7931E?logo=scikitlearn&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?logo=jupyter&logoColor=white)

Complete lab portfolio for **Introduction to AI**, BS Robotics & AI, University of Lahore.
It runs from raw data collection to trained models served as live web apps.


## Pipeline covered

```
Scrape/collect  ->  Features  ->  EDA  ->  Train  ->  Evaluate  ->  Serve (API)  ->  Deploy
   Lab 2           Lab 2-3      Lab 2     Lab 4       Lab 4        Lab 6, apps     Render
```

## Contents

| Folder | What it covers | Key tools |
|---|---|---|
| [`lab-02-web-scraping-eda/`](lab-02-web-scraping-eda/) | Scrape blog articles, engineer NLP features, EDA, TF-IDF | requests, BeautifulSoup, spaCy, TextBlob, seaborn |
| [`lab-03-text-image-features/`](lab-03-text-image-features/) | Turn text and images into numeric features | CountVectorizer, Pillow, NumPy |
| [`lab-04-regression-logistic/`](lab-04-regression-logistic/) | Linear and logistic regression, evaluation, cross-validation | scikit-learn, pandas, matplotlib |
| [`lab-06-fastapi-diabetes-api/`](lab-06-fastapi-diabetes-api/) | First version of the diabetes model served as a REST API | FastAPI, pydantic |
| [`lab-07-cnn-kernel-from-scratch/`](lab-07-cnn-kernel-from-scratch/) | CNN forward pass by hand: convolution, ReLU, dense, softmax | NumPy |
| [`cancer-app/`](cancer-app/) | Full-stack ML app with per-feature explanations | FastAPI, JS frontend |
| [`diabetes-app/`](diabetes-app/) | Full-stack ML app | FastAPI, JS frontend |
| [`practice/`](practice/) | Practice datasets and notebooks | pandas |

Labs 1 and 5 are not included in this repository.

## Run locally

```bash
git clone https://github.com/ahtresham-sys/Ai-lab-works.git
cd Ai-lab-works/diabetes-app      # or cancer-app
python -m pip install -r requirements.txt
python train.py                   # cancer-app only
python -m uvicorn main:app --reload --port 8000
```

Open http://localhost:8000 for the app and http://localhost:8000/docs for the API docs.

## Notes
- The `.pkl` models were saved with scikit-learn 1.9.1. Keep that version pinned.
- Educational projects only, not medical advice.

## Author
**Ahtesham**: [GitHub](https://github.com/ahtresham-sys)

# Lab 04: Regression and Logistic Regression

| Notebook | Content |
|---|---|
| `diabetes_logistic_regression.ipynb` | Full workflow: explore, split, scale, train, evaluate, feature importance, 5-fold cross-validation, predict a new patient |
| `diabetes_logistic_simple.ipynb` | Short version of the same model |
| `logistic regression.ipynb` | First logistic regression attempt |

**Diabetes results (held-out test set):** accuracy 0.727, precision 0.625, recall 0.556, ROC AUC 0.817.
**5-fold cross-validation:** accuracy 0.771 ± 0.040, ROC AUC 0.853 ± 0.027.

Datasets: `diabetes.csv`, `housing.csv`, `house_price_regression_dataset.csv`, `Salary_dataset.csv`.
Notebooks read the CSV from the same folder, so run them from this directory.

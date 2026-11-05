# Lab 02: Web Scraping, Feature Engineering and EDA

Builds a small NLP dataset from blog articles and explores it.

**Steps:** scrape title and body text (requests + BeautifulSoup) -> build a DataFrame -> extract features with spaCy (tokens, sentences, entities, nouns) -> sentiment with TextBlob (polarity, subjectivity) -> EDA with histograms, pair plot and TF-IDF top terms.

| File | Purpose |
|---|---|
| `Lab_02_Web_Scraping_and_EDA.ipynb` | Main notebook |
| `articles_raw.csv` / `articles.csv` | Raw scraped data and the cleaned dataset |
| `*.pdf`, `*.docx` | Lab manual and ETL pipeline handout |

```bash
pip install requests beautifulsoup4 lxml pandas spacy matplotlib seaborn textblob scikit-learn
python -m spacy download en_core_web_sm
```

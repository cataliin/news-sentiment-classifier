# Project Overview
A News Sentiment Classifier that processes headlines and predicts sentiment.

## Steps for Running the Program

```bash
python -m venv myenv && source myenv/bin/activate
pip install -r requirements.txt
python setup_nltk.py
python -m src.etl_pipeline
```

## ETL Pipeline Overview

The ETL pipeline processes the raw dataset into a clean, model-ready format.

### **Extract**
- Load the Kaggle dataset from the `data/raw` folder.
- Assign meaningful column names to the data.

### **Transform**
- Apply text cleaning and **POS-aware lemmatization**:
  - *Lemmatization* reduces words to their dictionary form (e.g., `"running"` → `"run"`).
  - *POS-aware* lemmatization first identifies the word’s **part of speech** using **Penn Treebank tags**  
    *(e.g., `('are', 'VBP')` means `"are"` is a verb in present tense)*.
  - A **POS map** converts these tags into **WordNet POS types** so each word is lemmatized accurately based on its role in the sentence.
- Remove stopwords, extra punctuation, and unwanted symbols.

### **Load**
- Save the cleaned dataset to the `data/clean` folder.
- Keep the original dataset unchanged in `data/raw` for reproducibility.
  
## Model Overview

The model is built as a **scikit-learn Pipeline** with two main steps:

### 1. TF-IDF Vectorizer
- Converts cleaned text into a numerical matrix that the model can understand.
- **TF-IDF**:
  - **Term Frequency (TF):** Words that appear more often in a headline get a higher score.
  - **Inverse Document Frequency (IDF):** Words that appear in most of the headlines get a lower weight.
- This is an improved Bag of Words method with an extra step (IDF) for importance weighting.

### 2. Logistic Regression Classifier
- An effective linear model for classification tasks.
- Learns **weights** for each TF-IDF feature that push the prediction toward *positive*, *negative*, or *neutral*.
- Uses `class_weight="balanced"` to reduce bias toward the neutral class, which is the majority one, by giving more importance to the other classes.

### Why a Pipeline?
- By saving both the **TF-IDF Vectorizer** and **Logistic Regression** together in a single pipeline (`model.joblib`), we ensure that:
  - New predictions use the **same preprocessing** as during training.
  - The workflow is consistent and reproducible.

**Data flow:**

Headline → Clean Text → TF-IDF Matrix → Logistic Regression → Sentiment Label

<img width="801" height="433" alt="image" src="https://github.com/user-attachments/assets/2025416d-76e5-42e5-b8ee-93a43b383a16" />


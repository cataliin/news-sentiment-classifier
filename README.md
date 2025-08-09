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

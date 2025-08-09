from . import config
import pandas as pd
from pathlib import Path
from .preprocessing import clean_text, normalize_label

def extract():
    df = pd.read_csv(config.RAW_DIR / 'all-data.csv', encoding='ISO-8859-1', header=None)
    df.columns = ['sentiment', 'headline']
    return df

def transform(df : pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset='headline')
    df['sentiment'] = df['sentiment'].map(normalize_label)
    df['clean_text'] = df['headline'].map(clean_text)

    return df

def load(df: pd.DataFrame) -> Path:
    config.CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    out = config.CLEAN_DIR / "cleaned_news.csv"
    df.to_csv(out, index=False, encoding='utf-8')

    return out

def run() -> Path:
    df = extract()
    df = transform(df)
    return load(df)

if __name__ == "__main__":
    path = run()
    print(f"Wrote cleaned dataset to: {path}")

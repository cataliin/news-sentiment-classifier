from . import config
import pandas as pd
from pathlib import Path
from .preprocessing import clean_text, normalize_label

def extract():
    df = pd.read_csv(config.RAW_DIR / config.DATA_FILE, encoding=config.RAW_ENCODING, header=None)
    df.columns = ['sentiment', 'headline']
    return df

def transform(df : pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset='headline')
    df['sentiment'] = df['sentiment'].map(normalize_label)
    df['clean_text'] = df['headline'].map(clean_text)

    return df

def load(df: pd.DataFrame) -> Path:
    config.CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    out = config.CLEAN_DIR / config.CLEANED_FILE
    df.to_csv(out, index=False, encoding=config.OUTPUT_ENCODING)

    return out

def run() -> Path:
    df = extract()
    df = transform(df)
    return load(df)

if __name__ == "__main__":
    path = run()
    print(f"Wrote cleaned dataset to: {path}")

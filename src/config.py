from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
CLEAN_DIR = ROOT / "data" / "clean"
MODEL_DIR = ROOT / "models"
DATA_FILE = "all-data.csv"
CLEANED_FILE = "cleaned_news.csv"
RAW_ENCODING = "ISO-8859-1"
OUTPUT_ENCODING = "utf-8"
RANDOM_SEED = 42
TEST_SIZE = 0.2


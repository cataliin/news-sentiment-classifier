from . import config
from .preprocessing import clean_text
import sys
import joblib

MODEL_PATH = config.MODEL_DIR / "model.joblib"
model = joblib.load(MODEL_PATH)

def predict_sentiment(txt: str):
    cleaned = clean_text(txt)
    return model.predict([cleaned])[0]

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("You need to add a headline")
        sys.exit(1)
    headline = " ".join(sys.argv[1:])
    prediction = predict_sentiment(headline)
    print("Prediction: ", prediction)
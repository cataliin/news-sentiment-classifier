import sys, pathlib
from pathlib import Path
import streamlit as st
import joblib

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import config
from src.preprocessing import clean_text

# Caching the model so it is created only once and then reused on reruns
@st.cache_resource
def load_model():
    model_path = config.MODEL_DIR / "model.joblib"
    if not model_path.exists():
         if not model_path:
            st.error("Model is not trained, you have to train it first by runnin the train_model.py file")
            st.stop()
    return joblib.load(model_path)

model = load_model()

st.title("News Sentiment Classifier")
headline = st.text_area("Enter a headline:", height=120, placeholder="e.g. Profits are quickly rising after the huge increase in sales")

col1, col2 = st.columns(2)
with col1:
    do_predict = st.button("Predict")
with col2:
    show_probs = st.checkbox("Show probabilities", value=True)

if do_predict:
    raw = (headline or "").strip()
    if not raw:
        st.warning("Please enter a headline.")
    else:
        cleaned = clean_text(raw)
        pred = model.predict([cleaned])[0]
        st.success(f"Prediction: **{pred}**")
        if show_probs and hasattr(model, "predict_proba"):
            probs = model.predict_proba([cleaned])[0]
            classes = model.classes_
            st.write("Probabilities:")
            for cls, p in sorted(zip(classes, probs), key=lambda x: -x[1]):
                st.write(f"- {cls}: {p:.4f}")

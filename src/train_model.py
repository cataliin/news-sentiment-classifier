from . import config
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, classification_report

config.MODEL_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(config.CLEAN_DIR / config.CLEANED_FILE, encoding=config.OUTPUT_ENCODING)

X, y = df["clean_text"], df["sentiment"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=config.TEST_SIZE, stratify=y, random_state=config.RANDOM_SEED
)

pipe = Pipeline([
    ("tfidf", TfidfVectorizer(
        ngram_range=(1, 3),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True)),
    ("clf", LogisticRegression(
        max_iter=1000, 
        class_weight="balanced",
        C=2.0,
        penalty="l2",
        multi_class="ovr"
        )),
])

pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Macro F1:", f1_score(y_test, y_pred, average="macro"))
print("\nClassification report:\n", classification_report(y_test, y_pred))
joblib.dump(pipe, config.MODEL_DIR / "model.joblib")
print("Saved model to", config.MODEL_DIR / "model.joblib")
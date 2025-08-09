import re
import nltk
from nltk.corpus import stopwords, wordnet as wn
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

stop_words -= {'no', 'nor', 'not', 'never'}

_POS_MAP = {
    'J': wn.ADJ,
    'V': wn.VERB,
    'N': wn.NOUN,
    'R': wn.ADV,
}

def _to_wordnet_pos(tag: str):
    return _POS_MAP.get(tag[0], wn.NOUN)

def clean_text(txt: str) -> str:
    txt = txt.lower()
    txt = re.sub(r'[^a-z0-9$%\.\-\s]', ' ', txt)
    txt = re.sub(r'\s+', ' ', txt).strip()

    tokens = nltk.word_tokenize(txt)
    tagged = nltk.pos_tag(tokens)
    
    cleaned =[]
    for word, tag in tagged:
        if word in stop_words:
            continue
        lemma = lemmatizer.lemmatize(word, _to_wordnet_pos(tag))
        cleaned.append(lemma)

    return ' '.join(cleaned)

def normalize_label(y: str) -> str:
    y = y.strip().lower()
    return {
        "neg": "negative",
        "negative": "negative",
        "bearish": "negative",
        "pos": "positive",
        "positive": "positive",
        "bullish": "positive",
        "neutral": "neutral"
    }.get(y, y)

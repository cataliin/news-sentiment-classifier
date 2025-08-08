import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

df = pd.read_csv('all-data.csv', encoding= 'ISO-8859-1', header = None)
df.columns = ['sentiment', 'headline']

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(txt):
    txt = txt.lower()
    txt = re.sub(r'[^a-z\s]', '', txt)
    
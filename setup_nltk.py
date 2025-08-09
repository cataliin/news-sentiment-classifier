import nltk

RESOURCES = [
    "stopwords",
    "punkt",
    "punkt_tab",
    "wordnet",
    "averaged_perceptron_tagger",
    "averaged_perceptron_tagger_eng", 
]

for r in RESOURCES:
    print(f"Downloading {r} ...")
    try:
        nltk.download(r, quiet=False, raise_on_error=True)
    except Exception as e:
        print(f"Failed to download {r}: {e}")

print("Finished NLTK resource setup.")

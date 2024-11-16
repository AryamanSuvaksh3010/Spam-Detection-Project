import pandas as pd
import string
from nltk.stem.porter import PorterStemmer


ps = PorterStemmer()


stopwords = ['i', 'me', 'my', ...]  


def transform_text(text):
    text = text.lower()
    text = text.split()
    text = [ps.stem(word) for word in text if word.isalnum() and word not in stopwords]
    return " ".join(text)


data = pd.read_csv("spam.csv", encoding='latin-1')


data.rename(columns={'v1': 'label', 'v2': 'text'}, inplace=True)


data['processed_text'] = data['text'].apply(transform_text)

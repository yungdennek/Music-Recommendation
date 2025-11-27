import nltk
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import wordpunct_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd


class TextProcessor:
    def __init__(self):
        self.stemmer = PorterStemmer()

    def normalize_text(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        df[column] = df[column].str.lower().replace(r'^\w\s', ' ', regex=True).replace(r'\n', ' ', regex=True)
        return df

    def tokenize(self, text: str) -> str:
        tokens = wordpunct_tokenize(text)
        stemmed = [self.stemmer.stem(w) for w in tokens]
        return " ".join(stemmed)

    def get_tfidf_matrix(self, corpus: list):
        tfidf = TfidfVectorizer(analyzer="word", stop_words="english")
        return tfidf.fit_transform(corpus)

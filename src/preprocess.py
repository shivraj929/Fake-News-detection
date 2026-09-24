"""
Text Preprocessing Module
- Cleans raw text (URLs, special characters, digits, extra whitespace)
- Combines title + text into a single feature
- Fits/transforms TF-IDF vectorizer
"""

import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import config


def clean_text(text):
    """Clean a single text string by removing noise."""
    if not isinstance(text, str):
        return ""
    # Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    # Remove HTML tags
    text = re.sub(r"<.*?>", "", text)
    # Remove special characters and digits
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()
    # Lowercase
    text = text.lower()
    return text


def preprocess_dataframe(df):
    """
    Preprocess the dataframe:
    - Combine title and text columns
    - Clean the combined text
    - Drop rows with empty text after cleaning
    """
    df = df.copy()

    # Combine title and text for richer features
    df["content"] = df["title"].fillna("") + " " + df["text"].fillna("")

    # Clean the combined text
    df["content"] = df["content"].apply(clean_text)

    # Drop rows where content is empty after cleaning
    df = df[df["content"].str.len() > 0].reset_index(drop=True)

    return df


def fit_tfidf(texts):
    """Fit a TF-IDF vectorizer on the given texts and save it."""
    tfidf = TfidfVectorizer(
        max_features=config.MAX_FEATURES,
        stop_words="english",
        ngram_range=(1, 2),
    )
    X = tfidf.fit_transform(texts)

    # Save the fitted vectorizer
    joblib.dump(tfidf, config.TFIDF_PATH)
    print(f"TF-IDF vectorizer saved to {config.TFIDF_PATH}")
    print(f"Vocabulary size: {len(tfidf.vocabulary_)}")

    return tfidf, X


def transform_tfidf(texts):
    """Load a saved TF-IDF vectorizer and transform texts."""
    tfidf = joblib.load(config.TFIDF_PATH)
    X = tfidf.transform(texts)
    return X


if __name__ == "__main__":
    # Quick test
    df = pd.read_csv(config.PROCESSED_DATA, nrows=100)
    df = preprocess_dataframe(df)
    print(f"Preprocessed shape: {df.shape}")
    print(f"Sample cleaned text:\n{df['content'].iloc[0][:200]}...")

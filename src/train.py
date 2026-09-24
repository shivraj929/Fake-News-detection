"""
Model Training Module
- Loads processed data
- Preprocesses and vectorizes text
- Trains 4 classifiers: Logistic Regression, Naive Bayes, SVM, Random Forest
- Saves trained models to disk
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
import joblib
import time

import config
from preprocess import preprocess_dataframe, fit_tfidf


def load_and_prepare_data():
    """Load processed data, preprocess text, and split into train/test."""
    print("=" * 60)
    print("LOADING DATA")
    print("=" * 60)

    df = pd.read_csv(config.PROCESSED_DATA)
    print(f"Dataset shape: {df.shape}")
    print(f"Label distribution:\n{df['label'].value_counts()}\n")

    # Preprocess text
    print("Preprocessing text...")
    df = preprocess_dataframe(df)
    print(f"After preprocessing: {df.shape}")

    # Split into train and test
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        df["content"],
        df["label"],
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=df["label"],
    )

    print(f"\nTrain size: {len(X_train_text)}")
    print(f"Test size:  {len(X_test_text)}")

    # Fit TF-IDF on training data only (to avoid data leakage)
    print("\nFitting TF-IDF vectorizer on training data...")
    tfidf, X_train = fit_tfidf(X_train_text)

    # Transform test data
    X_test = tfidf.transform(X_test_text)

    print(f"Train feature matrix: {X_train.shape}")
    print(f"Test feature matrix:  {X_test.shape}")

    return X_train, X_test, y_train, y_test


def train_models(X_train, y_train):
    """Train all 4 classifiers and return them as a dict."""
    models = {
        "Logistic Regression": {
            "model": LogisticRegression(
                max_iter=config.MAX_ITER,
                random_state=config.RANDOM_STATE,
            ),
            "path": config.LOGISTIC_MODEL,
        },
        "Naive Bayes": {
            "model": MultinomialNB(),
            "path": config.NAIVE_BAYES_MODEL,
        },
        "SVM (LinearSVC)": {
            "model": LinearSVC(
                max_iter=config.MAX_ITER,
                random_state=config.RANDOM_STATE,
            ),
            "path": config.SVM_MODEL,
        },
        "Random Forest": {
            "model": RandomForestClassifier(
                n_estimators=100,
                random_state=config.RANDOM_STATE,
                n_jobs=-1,
            ),
            "path": config.RANDOM_FOREST_MODEL,
        },
    }

    trained_models = {}

    for name, info in models.items():
        print("\n" + "=" * 60)
        print(f"TRAINING: {name}")
        print("=" * 60)

        start = time.time()
        info["model"].fit(X_train, y_train)
        elapsed = time.time() - start

        # Save the trained model
        joblib.dump(info["model"], info["path"])

        print(f"Training time: {elapsed:.2f}s")
        print(f"Model saved to: {info['path']}")

        trained_models[name] = info["model"]

    return trained_models


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_and_prepare_data()
    trained = train_models(X_train, y_train)
    print("\n" + "=" * 60)
    print("ALL MODELS TRAINED AND SAVED SUCCESSFULLY!")
    print("=" * 60)

    # Save test data for evaluation
    import scipy.sparse
    scipy.sparse.save_npz("models/X_test.npz", X_test)
    y_test.to_csv("models/y_test.csv", index=False)
    print("Test data saved for evaluation.")

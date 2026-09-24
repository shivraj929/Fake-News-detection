import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

# Data Paths
FAKE_DATA = os.path.join(PROJECT_ROOT, "data", "raw", "Fake.csv")
TRUE_DATA = os.path.join(PROJECT_ROOT, "data", "raw", "True.csv")
PROCESSED_DATA = os.path.join(PROJECT_ROOT, "data", "processed", "processed_news.csv")

# Model Paths
LOGISTIC_MODEL = os.path.join(MODELS_DIR, "logistic.pkl")
NAIVE_BAYES_MODEL = os.path.join(MODELS_DIR, "naive_bayes.pkl")
SVM_MODEL = os.path.join(MODELS_DIR, "svm.pkl")
RANDOM_FOREST_MODEL = os.path.join(MODELS_DIR, "random_forest.pkl")

TFIDF_PATH = os.path.join(MODELS_DIR, "tfidf.pkl")

# Parameters
TEST_SIZE = 0.2
RANDOM_STATE = 42
MAX_FEATURES = 10000
MAX_ITER = 1000

# Model display names and paths mapping
MODELS = {
    "logistic": {
        "name": "Logistic Regression",
        "path": LOGISTIC_MODEL,
    },
    "naive_bayes": {
        "name": "Naive Bayes",
        "path": NAIVE_BAYES_MODEL,
    },
    "svm": {
        "name": "SVM (LinearSVC)",
        "path": SVM_MODEL,
    },
    "random_forest": {
        "name": "Random Forest",
        "path": RANDOM_FOREST_MODEL,
    },
}


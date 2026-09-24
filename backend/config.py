# Data Paths (relative to backend/)
FAKE_DATA = "../data/raw/Fake.csv"
TRUE_DATA = "../data/raw/True.csv"
PROCESSED_DATA = "../data/processed/processed_news.csv"

# Model Paths (relative to backend/)
LOGISTIC_MODEL = "models/logistic.pkl"
NAIVE_BAYES_MODEL = "models/naive_bayes.pkl"
SVM_MODEL = "models/svm.pkl"
RANDOM_FOREST_MODEL = "models/random_forest.pkl"

TFIDF_PATH = "models/tfidf.pkl"

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

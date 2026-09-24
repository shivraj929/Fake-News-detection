"""
Prediction Module
- Loads the best trained model and TF-IDF vectorizer
- Predicts whether a given news text is FAKE or REAL
"""

import joblib
import config
from preprocess import clean_text


def predict_news(text, model_path=None):
    """
    Predict if a news article is fake or real.

    Args:
        text: The news article text to classify.
        model_path: Path to the model to use. Defaults to Logistic Regression.

    Returns:
        dict with prediction label and confidence.
    """
    if model_path is None:
        model_path = config.LOGISTIC_MODEL

    # Load model and vectorizer
    model = joblib.load(model_path)
    tfidf = joblib.load(config.TFIDF_PATH)

    # Clean and vectorize
    cleaned = clean_text(text)
    features = tfidf.transform([cleaned])

    # Predict
    prediction = model.predict(features)[0]
    label = "REAL" if prediction == 1 else "FAKE"

    result = {
        "prediction": prediction,
        "label": label,
        "model_used": model_path,
    }

    # Get probability if available
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(features)[0]
        result["confidence"] = float(max(proba))
        result["probabilities"] = {
            "FAKE": float(proba[0]),
            "REAL": float(proba[1]),
        }

    return result


if __name__ == "__main__":
    # Example usage
    sample_fake = (
        "BREAKING: Aliens land in Washington DC, President confirms "
        "secret deal with extraterrestrial government for unlimited energy."
    )
    sample_real = (
        "The Federal Reserve raised interest rates by 0.25 percentage points "
        "on Wednesday, as widely expected by economists and market analysts."
    )

    print("=" * 60)
    print("FAKE NEWS DETECTION - PREDICTION DEMO")
    print("=" * 60)

    for label, text in [("Expected FAKE", sample_fake), ("Expected REAL", sample_real)]:
        result = predict_news(text)
        print(f"\n{label}:")
        print(f"  Text: {text[:80]}...")
        print(f"  Prediction: {result['label']}")
        if "confidence" in result:
            print(f"  Confidence: {result['confidence']:.2%}")

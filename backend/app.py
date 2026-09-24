"""
Flask API Backend for Fake News Detection
- POST /api/predict         — classify news text
- POST /api/predict-all     — classify with all 4 models at once
- GET  /api/models          — list available models with metrics
- GET  /api/health          — health check
"""

import os
import sys
import json
# pyrefly: ignore [missing-import]
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import joblib
import pandas as pd

import config
from preprocess import clean_text

# Resolve the frontend directory path
FRONTEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend"))

app = Flask(__name__, static_folder=FRONTEND_DIR)
CORS(app)


# ── Serve Frontend ─────────────────────────────────────────────────────
@app.route("/")
def serve_index():
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.route("/<path:filename>")
def serve_static(filename):
    return send_from_directory(FRONTEND_DIR, filename)

# ── Cache loaded models in memory ──────────────────────────────────────
_model_cache = {}
_tfidf = None


def get_tfidf():
    """Lazy-load the TF-IDF vectorizer."""
    global _tfidf
    if _tfidf is None:
        _tfidf = joblib.load(config.TFIDF_PATH)
    return _tfidf


def get_model(model_key):
    """Lazy-load a model by its config key."""
    if model_key not in _model_cache:
        info = config.MODELS[model_key]
        _model_cache[model_key] = joblib.load(info["path"])
    return _model_cache[model_key]


# ── Load model comparison metrics if available ─────────────────────────
def load_metrics():
    """Load the saved model comparison CSV."""
    metrics_path = os.path.join(config.PROJECT_ROOT, "reports", "model_comparison.csv")
    if os.path.exists(metrics_path):
        df = pd.read_csv(metrics_path)
        return df.to_dict(orient="records")
    return []


# ── API Routes ─────────────────────────────────────────────────────────

@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "models_available": list(config.MODELS.keys())})


@app.route("/api/models", methods=["GET"])
def list_models():
    """Return available models and their evaluation metrics."""
    models = []
    metrics = load_metrics()
    metrics_by_name = {m["Model"]: m for m in metrics}

    for key, info in config.MODELS.items():
        entry = {
            "key": key,
            "name": info["name"],
            "available": os.path.exists(info["path"]),
        }
        # Attach metrics if we have them
        if info["name"] in metrics_by_name:
            m = metrics_by_name[info["name"]]
            entry["accuracy"] = round(m["Accuracy"], 4)
            entry["precision"] = round(m["Precision"], 4)
            entry["recall"] = round(m["Recall"], 4)
            entry["f1_score"] = round(m["F1-Score"], 4)
        models.append(entry)

    # Sort by F1 descending
    models.sort(key=lambda x: x.get("f1_score", 0), reverse=True)
    return jsonify(models)


@app.route("/api/predict", methods=["POST"])
def predict():
    """Predict fake/real for a single model."""
    data = request.get_json()

    if not data or "text" not in data:
        return jsonify({"error": "Missing 'text' field in request body"}), 400

    text = data["text"].strip()
    if not text:
        return jsonify({"error": "Text cannot be empty"}), 400

    model_key = data.get("model", "logistic")
    if model_key not in config.MODELS:
        return jsonify({"error": f"Unknown model '{model_key}'. Available: {list(config.MODELS.keys())}"}), 400

    # Clean and vectorize
    cleaned = clean_text(text)
    tfidf = get_tfidf()
    features = tfidf.transform([cleaned])

    # Predict
    model = get_model(model_key)
    prediction = int(model.predict(features)[0])
    label = "REAL" if prediction == 1 else "FAKE"

    result = {
        "prediction": prediction,
        "label": label,
        "model": config.MODELS[model_key]["name"],
        "model_key": model_key,
        "text_length": len(text),
        "cleaned_length": len(cleaned),
    }

    # Get probability if available
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(features)[0]
        result["confidence"] = round(float(max(proba)), 4)
        result["probabilities"] = {
            "fake": round(float(proba[0]), 4),
            "real": round(float(proba[1]), 4),
        }
    else:
        # For models without predict_proba (e.g., LinearSVC), use decision function
        if hasattr(model, "decision_function"):
            decision = float(model.decision_function(features)[0])
            # Convert decision function to a pseudo-confidence
            import math
            confidence = 1 / (1 + math.exp(-abs(decision)))
            result["confidence"] = round(confidence, 4)
        else:
            result["confidence"] = None

    return jsonify(result)


@app.route("/api/predict-all", methods=["POST"])
def predict_all():
    """Predict using all available models at once."""
    data = request.get_json()

    if not data or "text" not in data:
        return jsonify({"error": "Missing 'text' field in request body"}), 400

    text = data["text"].strip()
    if not text:
        return jsonify({"error": "Text cannot be empty"}), 400

    # Clean and vectorize once
    cleaned = clean_text(text)
    tfidf = get_tfidf()
    features = tfidf.transform([cleaned])

    results = []
    for key, info in config.MODELS.items():
        if not os.path.exists(info["path"]):
            continue

        model = get_model(key)
        prediction = int(model.predict(features)[0])
        label = "REAL" if prediction == 1 else "FAKE"

        entry = {
            "model_key": key,
            "model": info["name"],
            "prediction": prediction,
            "label": label,
        }

        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(features)[0]
            entry["confidence"] = round(float(max(proba)), 4)
            entry["probabilities"] = {
                "fake": round(float(proba[0]), 4),
                "real": round(float(proba[1]), 4),
            }
        elif hasattr(model, "decision_function"):
            import math
            decision = float(model.decision_function(features)[0])
            confidence = 1 / (1 + math.exp(-abs(decision)))
            entry["confidence"] = round(confidence, 4)
        else:
            entry["confidence"] = None

        results.append(entry)

    return jsonify({
        "text_length": len(text),
        "cleaned_length": len(cleaned),
        "results": results,
    })


if __name__ == "__main__":
    print("=" * 60)
    print("  FAKE NEWS DETECTION API")
    print("=" * 60)
    print(f"  Models directory: {config.MODELS_DIR}")
    print(f"  Available models: {list(config.MODELS.keys())}")
    print("=" * 60)
    app.run(debug=True, host="0.0.0.0", port=5000)

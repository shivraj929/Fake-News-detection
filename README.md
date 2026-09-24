<<<<<<< HEAD
# Fake-News-detection
=======
# 🛡️ VerifyAI — Fake News Detection System

An end-to-end Machine Learning web application that detects fake news articles using multiple NLP classifiers (**Random Forest**, **SVM**, **Logistic Regression**, **Multinomial Naive Bayes**). Features a modern dark glassmorphic frontend UI and a Flask REST API backend.

---

## 📊 Model Benchmark Results

Models were trained and evaluated on **44,680 news articles** (80/20 train-test split):

| Model | Accuracy | Precision | Recall | F1-Score | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **🌲 Random Forest** *(Best)* | **99.82%** | **0.9974** | **0.9988** | **0.9981** | ✅ Top Performer |
| **⚡ SVM (LinearSVC)** | **99.63%** | **0.9960** | **0.9962** | **0.9961** | ✅ High Accuracy |
| **📈 Logistic Regression** | **99.14%** | **0.9871** | **0.9948** | **0.9910** | ✅ Reliable Baseline |
| **🎯 Naive Bayes** | **95.01%** | **0.9435** | **0.9519** | **0.9477** | ✅ Fast Probabilistic |

---

## 📁 Repository Structure

```
fake-news-detection/
├── backend/            # Flask REST API server
│   ├── app.py          # Main Flask app & routing
│   ├── config.py       # Backend paths & configuration
│   ├── preprocess.py   # Text preprocessing pipeline
│   └── requirements.txt# Backend dependencies
├── frontend/           # Interactive Web Dashboard
│   ├── index.html      # UI HTML5 template
│   ├── style.css       # Glassmorphism styling & animations
│   └── script.js       # Dynamic AJAX logic & Chart.js rendering
├── src/                # ML Pipeline scripts
│   ├── preprocess.py   # TF-IDF & clean text processing
│   ├── train.py        # Model training script
│   ├── evaluate.py     # Evaluation & benchmark generation
│   ├── predict.py      # CLI inference script
│   └── config.py       # Global pipeline configuration
├── models/             # Pretrained model artifacts (.pkl & TF-IDF)
├── reports/            # Performance metrics (model_comparison.csv)
├── data/               # Raw & processed datasets
├── requirements.txt    # Top-level dependencies
└── README.md           # Project documentation
```

---

## 🚀 Quick Start & Installation

### 1. Clone Repository
```bash
git clone https://github.com/shivraj929/Fake-News-detection.git
cd Fake-News-detection
```

### 2. Set Up Virtual Environment & Install Dependencies
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

pip install -r backend/requirements.txt
```

### 3. Run Web Application & API
```bash
cd backend
python app.py
```
Open your browser and navigate to: **`http://localhost:5000`**

---

## 🔌 REST API Endpoints

- `GET /api/health` — Check server status & model readiness
- `GET /api/models` — Retrieve benchmark metrics for all models
- `POST /api/predict` — Predict fake/real status using a specific model
  ```json
  {
    "text": "WASHINGTON (Reuters) - Congress passed the budget bill...",
    "model": "random_forest"
  }
  ```
- `POST /api/predict-all` — Run ensemble predictions across all 4 models simultaneously
  ```json
  {
    "text": "BREAKING: Viral post claims mysterious object discovered..."
  }
  ```

---

## 🧪 Training & Evaluation

To retrain models or run evaluation from scratch:
```bash
# Evaluate existing models
python src/evaluate.py

# Train all models on processed dataset
python src/train.py
```

---

## 📜 License
Distributed under the MIT License.
>>>>>>> 71d6c74 (Initial commit: Fake News Detection application with trained ML models, Flask backend API, and interactive frontend UI)

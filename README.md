# ShieldAI • AI-Powered Fake Review & Spam Detection Engine

ShieldAI is an intelligent NLP and Machine Learning web application designed to identify fraudulent product reviews, spam bots, promotional clickbait, and rating mismatches with high precision.

---

## 🌟 Key Features

1. **Real-Time Review Classifier**:
   - Classify individual reviews in real time using trained NLP machine learning models.
   - Interactive star rating picker (1 to 5 stars).
   - Live confidence score, probability breakdown, and NLP preprocessed token visualization.
   - Quick preset sample loader (Genuine, Spam Bot, Hype Spam, Rating Mismatch).
   - Export analysis results to JSON.

2. **Batch & CSV Dataset Analyzer**:
   - Drag-and-drop CSV dataset upload with validation.
   - Sample CSV generator and download button.
   - Filterable, searchable table of flagged suspicious reviews with one-click "Inspect" deep-dive.
   - Aggregate KPI metrics (Total reviews, Genuine count, Fake count, Fake percentage).

3. **Heuristic Pattern & Spam Anomaly Engine**:
   - ALL CAPS ratio detection (>35%).
   - Excessive punctuation burst tracking (>=3 `!`).
   - Known spam keyword density matcher ("buy now", "discount", "free gift", etc.).
   - Repetition loop detector (>35% single-word density).
   - Suspicious user posting frequency analyzer.

4. **Sentiment & Rating Cross-Validation**:
   - Computes text sentiment polarity (-1.0 to +1.0).
   - Flags rating–sentiment contradictions (e.g. 5-star rating with negative text).

5. **ML Benchmark & Comparison Suite**:
   - Compares 4 machine learning algorithms:
     - **Multinomial Naive Bayes**
     - **Logistic Regression**
     - **Linear Support Vector Machine (SVM)**
     - **Random Forest Classifier**
   - Displays Accuracy, Precision, Recall, and F1-Score metrics with interactive Chart.js bar charts.
   - One-click "Retrain All Models" endpoint to retrain pipeline on demand.

---

## 📂 Project Architecture

```
major/
├── analytics/
│   ├── __init__.py
│   ├── pattern_detector.py      # Heuristic pattern rules & duplicate detection
│   └── sentiment_analyzer.py    # Sentiment polarity & rating mismatch detection
├── data/
│   └── README.md                  # Dataset download & format instructions
├── model/
│   ├── __init__.py
│   ├── fake_review_model.pkl    # Serialized best ML classifier
│   ├── tfidf_vectorizer.pkl     # Fitted TF-IDF Vectorizer (5,000 max features)
│   ├── metrics.json             # Model benchmark comparison scores
│   └── train.py                 # Multi-model training and evaluation script
├── preprocessing/
│   ├── __init__.py
│   └── text_cleaner.py          # Lowercasing, regex, stopwords, lemmatization
├── static/
│   ├── css/
│   │   └── style.css            # Dark glassmorphic design system
│   └── js/
│       └── app.js               # Frontend controller & Chart.js logic
├── templates/
│   └── index.html               # Multi-tab responsive single-page application
├── app.py                       # Flask web server & REST API endpoints
├── requirements.txt             # Python project dependencies
├── test_all.py                  # End-to-end automated test suite
└── README.md                    # Project documentation
```

---

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.8+ (Tested on Python 3.10 / 3.11 / 3.12)
- pip

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Download & Place Your Dataset
Download a real-world review dataset (Amazon or Flipkart reviews from Kaggle) and save it as:
```
data/reviews_dataset.csv
```
See [`data/README.md`](data/README.md) for detailed format instructions and recommended datasets.

The CSV must contain at least:
- `review_text` — the review content
- `label` — `0` for Genuine, `1` for Fake/Spam

### 4. Train the Models
```bash
python model/train.py
```

### 5. (Optional) Run Automated Tests
```bash
python test_all.py
```

### 6. Start the Application
```bash
python app.py
```

### 7. Access the Web Dashboard
Open your browser and navigate to:
```
http://127.0.0.1:5000
```

---

## 📡 REST API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Serves main UI dashboard |
| `POST` | `/api/predict` | Analyzes single review text & rating |
| `GET` | `/api/batch-analyze` | Analyzes default dataset |
| `POST` | `/api/batch-analyze` | Analyzes uploaded CSV file |
| `GET` | `/api/model-metrics` | Returns benchmark performance metrics |
| `POST` | `/api/retrain` | Retrains all 4 candidate models and updates best model |

---

## 🛡️ Technologies Used
- **Backend**: Python, Flask
- **Machine Learning & NLP**: Scikit-Learn, NLTK, Pandas, NumPy, Joblib
- **Frontend**: HTML5, CSS3 Glassmorphism, Vanilla JavaScript (ES6+), Chart.js, FontAwesome

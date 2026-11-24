import os
import sys
import io
import json
import joblib
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify

# Add root folder to sys.path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from preprocessing.text_cleaner import TextPreprocessor
from analytics.pattern_detector import PatternDetector
from analytics.sentiment_analyzer import SentimentAnalyzer
from model.train import train_and_evaluate, DATA_PATH

app = Flask(__name__, template_folder='templates', static_folder='static', static_url_path='/static')
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

@app.after_request
def add_header(response):
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '-1'
    return response

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model', 'fake_review_model.pkl')
VECTORIZER_PATH = os.path.join(os.path.dirname(__file__), 'model', 'tfidf_vectorizer.pkl')
METRICS_PATH = os.path.join(os.path.dirname(__file__), 'model', 'metrics.json')

preprocessor = TextPreprocessor()
pattern_detector = PatternDetector()
sentiment_analyzer = SentimentAnalyzer()

# Lazy loading variables
model = None
vectorizer = None

def load_artifacts():
    global model, vectorizer
    if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
        print("Model artifacts not found. Training models from dataset...")
        print("(Ensure your dataset CSV is placed at: data/reviews_dataset.csv)")
        train_and_evaluate()
    
    if model is None:
        model = joblib.load(MODEL_PATH)
    if vectorizer is None:
        vectorizer = joblib.load(VECTORIZER_PATH)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/api/predict', methods=['POST'])
def predict_single():
    try:
        load_artifacts()
        data = request.get_json(force=True) or {}
        review_text = data.get('review_text', '').strip()
        rating = data.get('rating', None)
        if rating is not None:
            try:
                rating = int(rating)
            except ValueError:
                rating = None
        user_id = data.get('user_id', None)

        if not review_text:
            return jsonify({"error": "Please enter review text to analyze."}), 400

        # 1. NLP Cleaning
        cleaned_text = preprocessor.clean_text(review_text)

        # 2. Vectorization & Prediction
        tfidf_features = vectorizer.transform([cleaned_text])
        prediction = int(model.predict(tfidf_features)[0]) # 0 = Genuine, 1 = Fake

        # Get probability / confidence score
        confidence = 85.0  # default for models like LinearSVC without predict_proba
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(tfidf_features)[0]
            confidence = round(float(probs[prediction]) * 100, 2)
        elif hasattr(model, "decision_function"):
            dec = model.decision_function(tfidf_features)[0]
            # Sigmoid approximation
            prob_fake = 1 / (1 + np.exp(-dec))
            prob = prob_fake if prediction == 1 else (1 - prob_fake)
            confidence = round(float(prob) * 100, 2)

        is_fake = (prediction == 1)

        # 3. Pattern Detection
        pattern_res = pattern_detector.analyze_single_review(review_text, user_id)

        # 4. Sentiment Analysis
        sentiment_res = sentiment_analyzer.analyze_sentiment(review_text, rating)

        # Boost fake likelihood if multiple severe flags triggered
        final_verdict = "Fake (Spam)" if is_fake or pattern_res["pattern_score"] > 0.6 else "Genuine (Real)"
        is_fake_verdict = (final_verdict == "Fake (Spam)")

        return jsonify({
            "status": "success",
            "verdict": final_verdict,
            "is_fake": is_fake_verdict,
            "confidence_score": confidence,
            "fake_probability": confidence if is_fake_verdict else round(100 - confidence, 2),
            "genuine_probability": round(100 - confidence, 2) if is_fake_verdict else confidence,
            "cleaned_text": cleaned_text,
            "pattern_analysis": pattern_res,
            "sentiment_analysis": sentiment_res
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


def parse_and_normalize_csv(file_source):
    """
    Safely loads and normalizes CSV data from:
    - Werkzeug FileStorage (uploaded via Flask)
    - Raw bytes or io.BytesIO
    - File path string

    Key features:
    1. Multi-encoding support: utf-8, utf-8-sig, cp1252, latin-1, iso-8859-1
    2. Delimiter auto-detection (comma, tab, semicolon)
    3. Handles bad/malformed lines without throwing errors
    4. Auto-detects and maps Amazon/Kaggle/SNAP/Flipkart columns:
       - review text ('reviewText', 'text', 'review_body', 'verified_reviews', 'review', etc.)
       - rating ('overall', 'score', 'stars', 'rating', etc.)
       - user id ('reviewerID', 'UserId', 'customer_id', 'user_id', etc.)
       - review id ('Id', 'asin', 'review_id', etc.)
    5. Filters empty/null review text rows cleanly.
    """
    raw_bytes = None
    if hasattr(file_source, 'read'):
        raw_bytes = file_source.read()
    elif isinstance(file_source, (str, os.PathLike)):
        if not os.path.exists(file_source):
            raise FileNotFoundError(f"File not found: {file_source}")
        with open(file_source, 'rb') as f:
            raw_bytes = f.read()
    elif isinstance(file_source, bytes):
        raw_bytes = file_source

    if not raw_bytes or len(raw_bytes.strip()) == 0:
        raise ValueError("The uploaded CSV file is empty.")

    encodings = ['utf-8', 'utf-8-sig', 'cp1252', 'latin-1', 'iso-8859-1']
    df = None
    last_err = None

    for enc in encodings:
        try:
            df = pd.read_csv(
                io.BytesIO(raw_bytes),
                encoding=enc,
                on_bad_lines='skip',
                sep=None,
                engine='python'
            )
            if df is not None and not df.empty:
                break
        except Exception as e:
            last_err = e

        try:
            df = pd.read_csv(
                io.BytesIO(raw_bytes),
                encoding=enc,
                on_bad_lines='skip',
                engine='c'
            )
            if df is not None and not df.empty:
                break
        except Exception as e:
            last_err = e

    if df is None or df.empty:
        raise ValueError(f"Could not parse CSV file. (Details: {last_err or 'Empty or unreadable format'})")

    # Map Amazon / e-commerce column names to standard schema
    col_mapping = {}
    cleaned_col_map = {
        col: col.strip('\ufeff').strip().lower().replace(' ', '_').replace('.', '_').replace('-', '_')
        for col in df.columns
    }

    TEXT_ALIASES = [
        'review_text', 'reviewtext', 'review_body', 'reviewbody', 'reviews_text',
        'verified_reviews', 'review_content', 'review_comment', 'review_sentence',
        'text', 'review', 'reviews', 'comment', 'comments', 'feedback', 'body',
        'content', 'message', 'description', 'summary', 'review_title', 'title'
    ]

    RATING_ALIASES = [
        'rating', 'ratings', 'overall', 'score', 'stars', 'star_rating',
        'star_ratings', 'user_rating', 'product_rating'
    ]

    USER_ALIASES = [
        'user_id', 'userid', 'reviewerid', 'reviewer_id', 'customer_id',
        'customerid', 'profile_name', 'profilename', 'author', 'user', 'username'
    ]

    ID_ALIASES = [
        'review_id', 'reviewid', 'id', 'asin', 'item_id', 'product_id'
    ]

    # Map Review Text column
    for alias in TEXT_ALIASES:
        matched_orig = next((orig for orig, cln in cleaned_col_map.items() if cln == alias), None)
        if matched_orig:
            col_mapping[matched_orig] = 'review_text'
            break

    if 'review_text' not in col_mapping.values():
        preview_cols = ", ".join([f"'{c}'" for c in df.columns[:8]])
        raise ValueError(
            f"Could not find a review text column. Found columns: [{preview_cols}]. "
            f"Please ensure your CSV includes a text column such as 'reviewText', 'Text', 'review_body', or 'review_text'."
        )

    # Map Rating column
    for alias in RATING_ALIASES:
        matched_orig = next((orig for orig, cln in cleaned_col_map.items() if cln == alias and orig not in col_mapping), None)
        if matched_orig:
            col_mapping[matched_orig] = 'rating'
            break

    # Map User ID column
    for alias in USER_ALIASES:
        matched_orig = next((orig for orig, cln in cleaned_col_map.items() if cln == alias and orig not in col_mapping), None)
        if matched_orig:
            col_mapping[matched_orig] = 'user_id'
            break

    # Map Review ID column
    for alias in ID_ALIASES:
        matched_orig = next((orig for orig, cln in cleaned_col_map.items() if cln == alias and orig not in col_mapping), None)
        if matched_orig:
            col_mapping[matched_orig] = 'review_id'
            break

    df = df.rename(columns=col_mapping)

    # Clean text column - drop missing or purely empty texts
    df['review_text'] = df['review_text'].fillna('').astype(str).str.strip()
    df = df[df['review_text'] != ''].reset_index(drop=True)

    if df.empty:
        raise ValueError("The CSV file contains no non-empty review text rows.")

    return df


@app.route('/api/batch-analyze', methods=['GET', 'POST'])
def batch_analyze():
    try:
        load_artifacts()

        # Check if CSV file uploaded or use local dataset
        df = None
        if request.method == 'POST' and 'file' in request.files:
            file = request.files['file']
            if file.filename != '':
                df = parse_and_normalize_csv(file)

        if df is None:
            if not os.path.exists(DATA_PATH):
                return jsonify({"error": "No dataset found. Please place your CSV at data/reviews_dataset.csv or upload one."}), 400
            df = parse_and_normalize_csv(DATA_PATH)

        if df.empty or 'review_text' not in df.columns:
            return jsonify({"error": "Invalid CSV dataset. Must contain a review text column."}), 400

        # Cap large datasets to 10,000 rows to ensure fast response without UI timeouts
        analyzed_count = min(len(df), 10000)
        if len(df) > 10000:
            df = df.head(10000).copy()

        # Clean all reviews
        cleaned_series = df['review_text'].astype(str).apply(preprocessor.clean_text)
        tfidf_mat = vectorizer.transform(cleaned_series)
        predictions = model.predict(tfidf_mat)

        fake_count = int(np.sum(predictions == 1))
        genuine_count = int(np.sum(predictions == 0))
        fake_percentage = round((fake_count / max(analyzed_count, 1)) * 100, 2)

        # Batch patterns
        batch_patterns = pattern_detector.analyze_batch_patterns(df)

        # Sentiments (first 100 samples)
        sentiments = [sentiment_analyzer.analyze_sentiment(t) for t in df['review_text'].head(100)]
        pos_sent = sum(1 for s in sentiments if s['sentiment'] == 'Positive')
        neg_sent = sum(1 for s in sentiments if s['sentiment'] == 'Negative')
        neu_sent = sum(1 for s in sentiments if s['sentiment'] == 'Neutral')

        # Sample flagged fake reviews
        flagged_samples = []
        df['pred_label'] = predictions
        fake_df = df[df['pred_label'] == 1].head(10)
        for idx, row in fake_df.iterrows():
            raw_rating = row.get('rating')
            parsed_rating = 5
            if pd.notnull(raw_rating):
                try:
                    parsed_rating = int(float(str(raw_rating).strip()))
                    parsed_rating = max(1, min(5, parsed_rating))
                except (ValueError, TypeError):
                    parsed_rating = 5

            rev_id_val = row.get('review_id')
            user_id_val = row.get('user_id')

            flagged_samples.append({
                "review_id": str(rev_id_val if pd.notnull(rev_id_val) and str(rev_id_val).strip() else idx + 1),
                "user_id": str(user_id_val if pd.notnull(user_id_val) and str(user_id_val).strip() else 'Anonymous'),
                "rating": parsed_rating,
                "review_text": str(row['review_text'])
            })

        return jsonify({
            "status": "success",
            "total_reviews": analyzed_count,
            "fake_count": fake_count,
            "genuine_count": genuine_count,
            "fake_percentage": fake_percentage,
            "batch_patterns": batch_patterns,
            "sentiment_summary": {
                "positive": pos_sent,
                "negative": neg_sent,
                "neutral": neu_sent
            },
            "sample_flagged_reviews": flagged_samples
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route('/api/model-metrics', methods=['GET'])
def get_metrics():
    try:
        if not os.path.exists(METRICS_PATH):
            train_and_evaluate()
        with open(METRICS_PATH, 'r') as f:
            data = json.load(f)
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/retrain', methods=['POST'])
def retrain():
    try:
        metrics_data = train_and_evaluate()
        global model, vectorizer
        model = joblib.load(MODEL_PATH)
        vectorizer = joblib.load(VECTORIZER_PATH)
        return jsonify({"status": "success", "message": "Model retrained successfully!", "metrics": metrics_data})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def find_available_port(start_port=5000, max_attempts=20):
    import socket
    for p in range(start_port, start_port + max_attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(('127.0.0.1', p))
                return p
            except OSError:
                continue
    return start_port


if __name__ == '__main__':
    import webbrowser, threading

    # Load or generate model artifacts
    try:
        load_artifacts()
        print("\n  [OK] ML model loaded successfully.")
    except Exception as e:
        print(f"\n  [WARNING] Artifact loading notice: {e}")
        try:
            train_and_evaluate()
            load_artifacts()
            print("  [OK] Successfully trained and loaded model artifacts.")
        except Exception as err:
            print(f"  [ERROR] Could not train models: {err}")

    port = find_available_port(5000)
    url = f"http://127.0.0.1:{port}"

    print("\n" + "=" * 55)
    print(f"  ShieldAI server is running at {url}")
    print("  Opening browser automatically...")
    print("=" * 55 + "\n")
    threading.Timer(1.5, lambda: webbrowser.open(url)).start()
    app.run(host='127.0.0.1', port=port, debug=False, use_reloader=False)

import json
import os
import io
import pandas as pd
from app import app, load_artifacts
from preprocessing.text_cleaner import TextPreprocessor
from analytics.pattern_detector import PatternDetector
from analytics.sentiment_analyzer import SentimentAnalyzer
from model.train import DATA_PATH

def test_full_system():
    print("=" * 60)
    print("SHIELDAI - FULL SYSTEM END-TO-END VERIFICATION")
    print("=" * 60)

    # 0. Check dataset exists
    if not os.path.exists(DATA_PATH):
        print(f"\n  [WARNING] Dataset not found at: {DATA_PATH}")
        print("  Please place your Amazon/Flipkart reviews CSV there first.")
        print("  Required columns: 'review_text' and 'label' (0=Genuine, 1=Fake)")
        print("  Skipping model-dependent tests.\n")
        dataset_available = False
    else:
        dataset_available = True

    # 1. Test Text Preprocessor
    print("\n[1/5] Testing Text Preprocessor...")
    preprocessor = TextPreprocessor()
    sample_text = "CHECK OUT THIS AMAZING PRODUCT AT https://deal.com/buy! Best quality ever!!!"
    cleaned = preprocessor.clean_text(sample_text)
    print(f"  Raw:     '{sample_text}'")
    print(f"  Cleaned: '{cleaned}'")
    assert len(cleaned) > 0, "Cleaned text should not be empty"
    assert "http" not in cleaned, "URLs should be removed"
    print("  [OK] NLP Preprocessor passed.")

    # 2. Test Pattern Detector
    print("\n[2/5] Testing Pattern Detector...")
    pd_detector = PatternDetector()
    spam_review = "BUY NOW BUY NOW BEST PRODUCT EVER 90% DISCOUNT CLICK PROFILE LIMITED OFFER!!!!!"
    pattern_res = pd_detector.analyze_single_review(spam_review)
    print(f"  Caps ratio: {pattern_res['caps_ratio']}%")
    print(f"  Exclamation count: {pattern_res['exclamation_count']}")
    print(f"  Pattern flags: {pattern_res['flags']}")
    assert len(pattern_res['flags']) >= 2, "Spam review should trigger multiple pattern flags"
    print("  [OK] Pattern Detector passed.")

    # 3. Test Sentiment Analyzer & Mismatch
    print("\n[3/5] Testing Sentiment Analyzer...")
    sent_analyzer = SentimentAnalyzer()
    mismatch_review = "Worst product ever, complete garbage, broke in 1 minute."
    sent_res = sent_analyzer.analyze_sentiment(mismatch_review, rating=5)
    print(f"  Text Sentiment: {sent_res['sentiment']}")
    print(f"  Polarity Score: {sent_res['polarity_score']}")
    print(f"  Rating Mismatch Flag: {sent_res['rating_mismatch']}")
    assert sent_res['sentiment'] == "Negative", "Review should be negative"
    assert sent_res['rating_mismatch'] is True, "5-star rating with negative text should trigger mismatch"
    print("  [OK] Sentiment & Mismatch Analyzer passed.")

    # 4. Test Flask Test Client & Prediction API
    print("\n[4/5] Testing Flask API Endpoints...")

    if not dataset_available:
        print("  [SKIPPED] No dataset — cannot test prediction and batch APIs.")
        print("  Place your dataset CSV and re-run tests.")
        print("\n" + "=" * 60)
        print("PARTIAL TESTS PASSED (NLP + Pattern + Sentiment).")
        print("Download a dataset and re-run for full verification.")
        print("=" * 60)
        return

    load_artifacts()
    client = app.test_client()

    # 4a. Home Page Route
    res_home = client.get('/')
    assert res_home.status_code == 200, "Home route should return 200"
    assert b"ShieldAI" in res_home.data, "Home page must contain ShieldAI"
    print("  [OK] GET / (Home Page) passed.")

    # 4b. Prediction API - Genuine Review
    genuine_payload = {
        "review_text": "The laptop battery lasts around 8 hours with normal coding and browsing. Build quality is solid.",
        "rating": 4,
        "user_id": "TechBuyer_402"
    }
    res_pred_gen = client.post('/api/predict', json=genuine_payload)
    data_gen = json.loads(res_pred_gen.data)
    print(f"  Prediction (Genuine): verdict='{data_gen['verdict']}', fake_prob={data_gen['fake_probability']}%")
    assert res_pred_gen.status_code == 200, "Prediction endpoint should return 200"
    assert data_gen['is_fake'] is False, "Genuine sample should be classified as genuine"

    # 4c. Prediction API - Spam Review
    spam_payload = {
        "review_text": "BEST PRODUCT EVER EVER CREATED BUY NOW IMMEDIATELY DONT MISS OUT BEST DEAL CLICK HERE FOR DISCOUNT!!!!!!",
        "rating": 5,
        "user_id": "SpamBot_99"
    }
    res_pred_spam = client.post('/api/predict', json=spam_payload)
    data_spam = json.loads(res_pred_spam.data)
    print(f"  Prediction (Spam): verdict='{data_spam['verdict']}', fake_prob={data_spam['fake_probability']}%")
    assert res_pred_spam.status_code == 200, "Prediction endpoint should return 200"
    assert data_spam['is_fake'] is True, "Spam sample should be classified as fake"
    print("  [OK] POST /api/predict passed.")

    # 4d. Model Metrics API
    res_metrics = client.get('/api/model-metrics')
    data_metrics = json.loads(res_metrics.data)
    assert res_metrics.status_code == 200, "Metrics endpoint should return 200"
    assert "models" in data_metrics, "Metrics must include model performance table"
    print(f"  Active Best Model: {data_metrics['best_model']}")
    print("  [OK] GET /api/model-metrics passed.")

    # 5. Test Batch Analysis API (Default & Custom CSV)
    print("\n[5/5] Testing Batch Analysis Endpoint...")
    res_batch_get = client.get('/api/batch-analyze')
    data_batch = json.loads(res_batch_get.data)
    assert res_batch_get.status_code == 200, "Batch GET should return 200"
    print(f"  Total Dataset Reviews: {data_batch['total_reviews']}")
    print(f"  Fake Count:            {data_batch['fake_count']} ({data_batch['fake_percentage']}%)")
    print(f"  Genuine Count:         {data_batch['genuine_count']}")

    # Batch Upload Test with In-Memory CSV (Standard format)
    csv_content = """review_id,user_id,rating,review_text
1,User1,5,BEST PRODUCT BUY NOW DISCOUNT CLICK HERE!!!!!!
2,User2,4,Very nice build and fast delivery.
3,User3,1,Total scam broke on arrival.
"""
    csv_bytes = io.BytesIO(csv_content.encode('utf-8'))
    res_batch_post = client.post(
        '/api/batch-analyze',
        data={'file': (csv_bytes, 'test_upload.csv')},
        content_type='multipart/form-data'
    )
    data_upload = json.loads(res_batch_post.data)
    assert res_batch_post.status_code == 200, f"Batch CSV upload should return 200: {data_upload}"
    assert data_upload['total_reviews'] == 3, "Should analyze exactly 3 uploaded rows"
    print("  [OK] POST /api/batch-analyze (Standard CSV) passed.")

    # Batch Upload Test with Amazon SNAP dataset format (reviewText, overall, reviewerID)
    amazon_snap_csv = """reviewerID,asin,reviewerName,helpful,reviewText,overall,summary
AMZ_USR1,B001,ShopperA,[2,3],Amazing headphones with crystal clear sound and deep bass.,5.0,Great Sound
AMZ_USR2,B001,ShopperB,[0,0],BUY NOW LIMITED OFFER 90% DISCOUNT CLICK PROFILE FREE SHIPPING!!!!!,5.0,Super deal
AMZ_USR3,B002,ShopperC,[1,1],Broke on first use very poor plastic quality and terrible support.,1.0,Do not buy
"""
    res_snap = client.post(
        '/api/batch-analyze',
        data={'file': (io.BytesIO(amazon_snap_csv.encode('utf-8')), 'amazon_snap.csv')},
        content_type='multipart/form-data'
    )
    data_snap = json.loads(res_snap.data)
    assert res_snap.status_code == 200, f"Amazon SNAP upload should return 200: {data_snap}"
    assert data_snap['total_reviews'] == 3, "Should analyze 3 Amazon reviews"
    print("  [OK] POST /api/batch-analyze (Amazon SNAP 'reviewText' / 'overall') passed.")

    # Batch Upload Test with Amazon Fine Food format (Text, Score, UserId) + CP1252 encoding
    amazon_ff_csv = """Id,ProductId,UserId,ProfileName,Score,Time,Summary,Text
1,B00001,USR_FOOD_1,Foodie,4.0,1303862400,Good flavor,"It’s really tasty and fresh, highly recommend!"
2,B00002,USR_FOOD_2,SpamGuy,5.0,1303862400,Win free items,"FREE GIFT CARDS CLICK HERE BUY NOW DISCOUNT NOW NOW NOW!!!!!"
"""
    res_ff = client.post(
        '/api/batch-analyze',
        data={'file': (io.BytesIO(amazon_ff_csv.encode('cp1252')), 'amazon_reviews.csv')},
        content_type='multipart/form-data'
    )
    data_ff = json.loads(res_ff.data)
    assert res_ff.status_code == 200, f"Amazon Fine Food / CP1252 upload should return 200: {data_ff}"
    assert data_ff['total_reviews'] == 2, "Should analyze 2 Amazon Fine Food reviews"
    print("  [OK] POST /api/batch-analyze (Amazon Fine Food 'Text' / 'Score' + CP1252) passed.")

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED! ShieldAI SYSTEM IS 100% OPERATIONAL.")
    print("=" * 60)

if __name__ == "__main__":
    test_full_system()

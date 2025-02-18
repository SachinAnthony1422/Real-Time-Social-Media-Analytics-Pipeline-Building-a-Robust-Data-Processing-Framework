from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import pickle  # For loading the model
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from flask_cors import CORS
import os  # For checking if files exist

app = Flask(__name__)
CORS(app)  # ✅ Enable CORS for React communication

@app.route('/')
def home():
    return "✅ Flask API is Running! Available endpoints: /predict_sentiment, /predict_engagement, /predict_hashtag_cluster, /get_data"

# ✅ Load dataset (Check if file exists)
csv_file = "phase3_results.csv"
if os.path.exists(csv_file):
    df = pd.read_csv(csv_file)
    print("✅ Dataset loaded successfully!")
else:
    df = pd.DataFrame()  # Empty DataFrame if file is missing
    print("❌ Warning: Dataset file not found!")

# ✅ Load trained engagement model safely
model_file = "engagement_model.pkl"
rf_model = None
if os.path.exists(model_file):
    with open(model_file, "rb") as file:
        rf_model = pickle.load(file)
    print("✅ Model loaded successfully!")
else:
    print("❌ Model file not found! Ensure 'engagement_model.pkl' is in the backend folder.")

# ✅ Initialize VADER sentiment analyzer
sentiment_analyzer = SentimentIntensityAnalyzer()
# ✅ API for Sentiment Analysis
@app.route('/predict_sentiment', methods=['POST'])
def predict_sentiment():
    data = request.json
    text = data.get("text", "")

    if not text.strip():
        return jsonify({"error": "No text provided"}), 400

    sentiment_score = sentiment_analyzer.polarity_scores(text)["compound"]
    sentiment_label = "Positive" if sentiment_score > 0.05 else ("Negative" if sentiment_score < -0.05 else "Neutral")

    return jsonify({"sentiment_score": sentiment_score, "sentiment_label": sentiment_label})                                                                                                                    # ✅ Train KMeans for Hashtag Clustering (Pre-trained for speed)
vectorizer = TfidfVectorizer()
sample_hashtags = ["#AI", "#Python", "#DataScience", "#MachineLearning", "#DeepLearning"]
X_sample = vectorizer.fit_transform(sample_hashtags)

kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
kmeans.fit(X_sample)  # ✅ Fit KMeans on sample data

@app.route('/predict_hashtag_cluster', methods=['POST'])
def predict_hashtag_cluster():
    data = request.json
    hashtags = data.get("hashtags", "").strip()

    if not hashtags:
        return jsonify({"error": "No hashtags provided"}), 400

    try:
        X = vectorizer.transform([hashtags])
        cluster = kmeans.predict(X)[0]
        return jsonify({"cluster": int(cluster)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ✅ API for Engagement Prediction
@app.route('/predict_engagement', methods=['POST'])
def predict_engagement():
    data = request.json
    likes = data.get("likes", 0)
    shares = data.get("shares", 0)
    comments = data.get("comments", 0)

    # Validate input (Ensure values are numbers)
    try:
        likes, shares, comments = float(likes), float(shares), float(comments)
    except ValueError:
        return jsonify({"error": "Likes, shares, and comments must be numeric"}), 400

    if rf_model is None:
        return jsonify({"error": "Model not loaded. Please train and save the model first!"}), 500

    try:
        features = np.array([[likes, shares, comments]])
        predicted_engagement = rf_model.predict(features)[0]
        return jsonify({"predicted_engagement": float(predicted_engagement)})  # Convert NumPy float to Python float
    except Exception as e:
        return jsonify({"error": str(e)}), 500



# ✅ API to View Dataset (Check if CSV is Loaded)
@app.route('/get_data', methods=['GET'])
def get_data():
    if df.empty:
        return jsonify({"error": "Dataset not loaded"}), 500
    return jsonify(df.head(10).to_dict(orient="records"))  # Return first 10 rows as JSON array

# ✅ Flask Server Start
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

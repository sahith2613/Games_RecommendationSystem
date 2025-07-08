# 🎮 Infinity Recs – Game Recommendation System

Infinity Recs is a content-based game recommendation system built with Python, Flask, and machine learning techniques. It helps users find similar games based on genre, platform, rating, and release year using fuzzy matching and a TF-IDF-based similarity model.

---

## 🚀 Features

- 🔍 Fuzzy matching of game titles using `fuzzywuzzy`
- 🧠 Content-based similarity using TF-IDF vectors
- 📅 Boosted recommendations based on release year
- 🌐 REST API built with Flask and Flask-CORS
- 🔌 Easy integration with frontend via JSON API

---

## 📁 Project Structure
frontend/ # React or other frontend UI (not included here)
└── ... # (Not tracked in this push)
/backend/
├── app.py # Main Flask application
├── games.csv # Game metadata dataset
├── tfidf_vectorizer.pkl # Pretrained TF-IDF vectorizer
├── similarity_matrix.pkl # Precomputed cosine similarity matrix

---

## 🔧 Setup Instructions

### 1. Clone the repository
git clone https://github.com/sahith2613/Games_RecommendationSystem.git
cd Games_RecommendationSystem/backend


2. Install Python dependencies

pip install -r requirements.txt
If requirements.txt is missing, install manually:

pip install flask pandas numpy scikit-learn fuzzywuzzy python-Levenshtein joblib flask-cors
3. Run the Flask API

python app.py
Server will run on http://127.0.0.1:5000




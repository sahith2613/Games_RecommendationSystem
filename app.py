import numpy as np 
import pandas as pd
from fuzzywuzzy import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
import joblib
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load data
df = pd.read_csv('C:/Users/ASUS/Desktop/Games/games.csv', encoding='latin1')
df.fillna('', inplace=True)

games_tfidf = joblib.load('C:/Users/ASUS/Desktop/Games/tfidf_vectorizer.pkl')
games_similarity_matrix = joblib.load('C:/Users/ASUS/Desktop/Games/similarity_matrix.pkl')

def extract_year(date_str): 
    try:
        return datetime.strptime(date_str, "%d-%m-%Y").year
    except:
        return None

df["release_year"] = df["released"].apply(extract_year)
df["combined"] = df["genres"] + " " + df["platforms"]

def recommend_content_based_games(query, initial_n=20, final_n=20, date_weight=0.4):
    try:
        def find_closest_game(query, titles):
            scores = [(title, fuzz.token_sort_ratio(query.lower(), title.lower())) for title in titles]
            best_match, score = max(scores, key=lambda x: x[1])
            if score < 50:
                print(f"No close match found for '{query}'")
                return None
            return best_match

        closest_title = find_closest_game(query, df["name"])
        if not closest_title:
            return []

        game_idx = df[df["name"] == closest_title].index[0]
        matched_year = df.loc[game_idx, "release_year"]
        if matched_year is None:
            return []

        sim_scores = []
        for idx, sim in enumerate(games_similarity_matrix[game_idx]):
            if idx == game_idx:
                continue
            release_year = df.loc[idx, "release_year"]
            year_score = 1.0 if release_year is not None and release_year >= matched_year else 0.0
            combined_score = (1 - date_weight) * sim + date_weight * year_score
            sim_scores.append((idx, combined_score))

        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        top_indices = [idx for idx, score in sim_scores[:final_n]]

        recommendations = df.iloc[top_indices][[
            "name", "genres", "platforms", "rating", "released", "cover_image", "game_link"
        ]].copy()

        recommendations["item_id"] = recommendations.index.astype(str)
        recommendations = recommendations.rename(columns={
            "name": "title",
            "cover_image": "image_url",
            "game_link": "link"
        })

        recommendations = recommendations[[
            "item_id", "title", "genres", "platforms", "rating", "released", "image_url", "link"
        ]]

        return recommendations.to_dict('records')
    except Exception as e:
        print(f"Error in game recommendation: {e}")
        return []

@app.route('/recommend', methods=['GET'])
@app.route('/api/recommend', methods=['GET'])
def recommend():
    title = request.args.get('title')
    if not title:
        return jsonify({"error": "Title is required"}), 400

    recommendations = recommend_content_based_games(title)
    print(f"Recommendations: {len(recommendations)} items")
    return jsonify(recommendations)

if __name__ == '__main__': 
    app.run(debug=True, port=5000)

# Importer les bibliothèques nécessaires
from flask import Flask, render_template, request, send_from_directory
import pickle
import pandas as pd
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Charger le modèle et le vectoriseur à partir des fichiers pickle
with open('sentiment_model.pkl', 'rb') as model_file:
    loaded_model = pickle.load(model_file)

with open('tfidfvectorizer.pkl', 'rb') as vectorizer_file:
    loaded_vectorizer = pickle.load(vectorizer_file)

# Charger le dataset pour la recommandation de films
data = pd.read_csv('top_1000_IMDb_movies.csv')
tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(data["about"])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
index_sim = pd.Series(data.index, index=data['name']).drop_duplicates()

# Charger le dataset des sentiments
sentiment_data = pd.read_csv('imdb_movies_sentiments_svm_with_predictions.csv')

# Fonction pour analyser les sentiments
def analyze_sentiment(comment):
    comment_tfidf = loaded_vectorizer.transform([comment])
    sentiment = loaded_model.predict(comment_tfidf)[0]
    return sentiment

# Ajoutez cette fonction pour obtenir les détails du film à partir de TMDb
def get_movie_details(movie_title):
    base_url = "https://api.themoviedb.org/3/search/movie"
    api_key = "eac6c6cdd7ad8d756970feeb9b6fb6b3"  #  clé API TMDb
    params = {
        'api_key': api_key,
        'query': movie_title,
    }

    response = requests.get(base_url, params=params)
    data = response.json()

    if data['results']:
        movie_details = data['results'][0]
        return movie_details
    else:
        return None

# Fonction pour obtenir des recommandations en utilisant la similarité cosine
def get_recommendations(title):
    idx = index_sim[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    movie_indices = [i[0] for i in sim_scores]
    return list(data['name'].iloc[movie_indices].values)

def analyze_sentiments_and_recommend_with_percentage(title, positive_threshold, negative_threshold):
    recommended_movies = get_recommendations(title)

    positive_recommendations = []

    for movie in recommended_movies:
        movie_details = get_movie_details(movie)

        if movie_details:
            # Ajoutez le détail du film à la liste des recommandations
            positive_recommendations.append({
                'title': movie_details['title'],
                'poster_path': movie_details['poster_path'],
            })

    return positive_recommendations
@app.route('/', methods=['GET', 'POST'])
def home():
    predicted_sentiment = None
    recommendations_sentiment = None
    positive_recommendations = None
    positive_recommendation_message = None  # Ajoutez cette ligne
    film_name = None  # Ajoutez cette ligne

    # Récupérer la liste des noms de films à partir du dataset
    movie_list = data['name'].tolist()

    if request.method == 'POST':
        action = request.form['action']

        if action == 'predict':
            # Code pour la prédiction de sentiments
            comment = request.form['comment']
            predicted_sentiment = analyze_sentiment(comment)

        elif action == 'recommend':
            # Code pour la recommandation de films avec analyse de sentiment
            film_name = request.form['film_name']
            positive_recommendations = analyze_sentiments_and_recommend_with_percentage(film_name, positive_threshold=80, negative_threshold=15)
            positive_recommendation_message = f"Positive Recommendations for '{film_name}':"  # Ajoutez cette ligne

    return render_template('index.html', predicted_sentiment=predicted_sentiment, film_name=film_name, recommendations_sentiment=recommendations_sentiment, positive_recommendations=positive_recommendations, positive_recommendation_message=positive_recommendation_message, movie_list=movie_list)


if __name__ == '__main__':
    app.run(debug=True)


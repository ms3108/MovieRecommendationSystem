import pandas as pd
from flask import Flask, render_template, request
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import coo_matrix
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load MovieLens dataset
movies = pd.read_csv("data/movies.csv")  # Replace with the correct path to your movies dataset
ratings = pd.read_csv("data/ratings.csv")  # Replace with the correct path to your ratings dataset

# Filter top N movies and users (for optimization)
top_movies = ratings["movieId"].value_counts().head(1000).index
top_users = ratings["userId"].value_counts().head(5000).index

# Filter ratings to include only top movies and users
filtered_ratings = ratings[
    (ratings["movieId"].isin(top_movies)) & (ratings["userId"].isin(top_users))
    ]

# Create a sparse matrix for user-movie interactions
movie_user_matrix_sparse = coo_matrix(
    (filtered_ratings["rating"], (filtered_ratings["movieId"], filtered_ratings["userId"]))
).tocsr()

# Train KNN model using cosine similarity
model_knn = NearestNeighbors(metric='cosine', algorithm='brute')
model_knn.fit(movie_user_matrix_sparse)

# Get movie titles by movieId for easy lookup
movie_titles = pd.Series(movies['title'].values, index=movies['movieId']).to_dict()


# Function for content-based filtering (using genres)
def content_based_filtering(movie_id, top_n=5):
    movie = movies[movies['movieId'] == movie_id]
    if movie.empty:
        return []

    movie_genres = movie['genres'].values[0].split('|')
    genre_based_recommendations = []

    for index, row in movies.iterrows():
        # Compare genres of each movie to the input movie
        movie_genres2 = row['genres'].split('|')
        common_genres = set(movie_genres).intersection(set(movie_genres2))

        # If there is any genre match, add it to recommendations
        if common_genres:
            genre_based_recommendations.append(row['movieId'])

    # Return top N recommendations based on genre similarity
    return movie_titles[movie_titles.index.isin(genre_based_recommendations)].tolist()[:top_n]


# Hybrid Recommendation System combining KNN and Content-based filtering
def hybrid_recommendation(user_input, top_n=5):
    # Collaborative Filtering: KNN-based recommendations
    knn_recommendations = collaborative_filtering(user_input)

    # Content-Based Filtering: Genre-based recommendations
    content_recommendations = content_based_filtering(user_input)

    # Combine recommendations
    final_recommendations = list(set(knn_recommendations + content_recommendations))[:top_n]

    return final_recommendations


# Collaborative Filtering: KNN-based recommendation function
def collaborative_filtering(movie_name, top_n=5):
    matched_movies = movies[movies['title'].str.contains(movie_name, case=False, na=False)]

    if matched_movies.empty:
        return []

    movie_id = matched_movies.iloc[0]['movieId']

    # Find the movie's vector in the sparse matrix
    movie_vector = movie_user_matrix_sparse[movie_id].toarray()

    # Find the most similar movies
    distances, indices = model_knn.kneighbors(movie_vector, n_neighbors=top_n + 1)  # +1 to skip the movie itself

    recommended_movies = []
    for i in range(1, len(indices[0])):
        recommended_movie_id = indices[0][i]
        recommended_movie_name = movie_titles.get(recommended_movie_id)
        recommended_movies.append(recommended_movie_name)

    return recommended_movies


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/recommend', methods=['POST'])
def recommend():
    # Get the movie name from the user input
    movie_name = request.form.get('movie_name')

    # Get recommendations using the hybrid system
    recommended_movies = hybrid_recommendation(movie_name, top_n=5)

    if not recommended_movies:
        return render_template('index.html', recommendations=[],
                               error="No movies found or no recommendations available!")

    return render_template('index.html', recommendations=recommended_movies, movie_name=movie_name)


if __name__ == '__main__':
    app.run(debug=True)

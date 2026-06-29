import sqlite3
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

class MovieRecommender:
    def __init__(self):
        self.conn = sqlite3.connect('data/movies.db')
        self.movies = pd.read_sql("SELECT * FROM movies", self.conn)
        self.ratings = pd.read_sql("SELECT * FROM ratings", self.conn)

    def get_user_ratings(self, user_id):
        """SQL Join example"""
        query = """
        SELECT m.title, r.rating
        FROM ratings r
        JOIN movies m ON r.movieId = m.movieId
        WHERE r.userId = ?
        ORDER BY r.rating DESC
        """
        return pd.read_sql(query, self.conn, params=(user_id,))
    
    def collaborative_filtering(self, user_id, top_n=5):
        """User-based Collaborative Filtering"""
        user_items = self.ratings.pivot_table(index='userId', columns='movieId', values='rating').fillna(0)

        user_sim = cosine_similarity(user_items)
        user_sim_df = pd.DataFrame(user_sim, index=user_items.index, columns=user_items.index)

        if user_id not in user_sim_df.index:
            return ["New user - Try content-based recommendations"]

        similar_users = user_sim_df[user_id].sort_values(ascending=False)[1:11].index
        similar_ratings = self.ratings[self.ratings['userId'].isin(similar_users)]
        weighted = similar_ratings.groupby('movieId')['rating'].mean()


        user_watched = self.ratings[self.ratings['userId'] == user_id]['movieId']
        recommendations = weighted.drop(user_watched, errors='ignore').sort_values(ascending=False)

        rec_movies = recommendations.head(top_n).index.tolist()
        return self.movies[self.movies['movieId'].isin(rec_movies)]['title'].tolist()

    def content_based_recommendation(self, movie_title, top_n=5):
        """Vectorization with TF-IDF"""
        tfidf = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf.fit_transform(self.movies['genres'].fillna(''))

        idx = self.movies[self.movies['title'].str.contains(movie_title, case=False, na=False)].index
        if len(idx) == 0:
            return ["Movie not found"]
        idx = idx[0]

        cosine_sim = cosine_similarity(tfidf_matrix[idx:idx+1], tfidf_matrix).flatten()
        similar_indices = cosine_sim.argsort()[-top_n-1:-1][::-1]

        return self.movies.iloc[similar_indices]['title'].tolist()

    def hybrid_recommendation(self, user_id, movie_title=None, top_n=5):
        """Hybrid: Collaborative + Content_based"""
        collab_recs = self.collaborative_filtering(user_id, top_n*2)

        if movie_title:
            content_recs = self.content_based_recommendation(movie_title, top_n*2)
        else:
            user_top = self.get_user_ratings(user_id).head(1)
            if not user_top.empty:
                seed_movie = user_top.iloc[0]['title']
                content_recs = self.content_based_recommendation(seed_movie, top_n*2)

            else:
                content_recs = []

        all_recs = list(dict.fromkeys(collab_recs + content_recs))
        return all_recs[:top_n]
    def close(self):
        self.conn.close()

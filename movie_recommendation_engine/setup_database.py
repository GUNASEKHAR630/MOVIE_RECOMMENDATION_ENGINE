import sqlite3
import pandas as pd

def setup_db():
    conn = sqlite3.connect('data/movies.db')
    cursor = conn.cursor()

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS movies (
        movieId INTEGER PRIMARY KEY,
        title TEXT,
        genres TEXT               
    )
    ''')
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS ratings (
        userId INTEGER,
        movieId INTEGER,
        ratings REAL,
        timestamp INTEGER,
        FOREIGN KEY (movieId) REFERENCES movies(movieId)               
    )               
    ''')

    movies_df = pd.read_csv('data/movies.csv')
    ratings_df = pd.read_csv('data/ratings.csv')

    movies_df.to_sql('movies', conn, if_exists='replace',index=False)
    ratings_df.to_sql('ratings', conn, if_exists='replace', index=False)

    print("Database setup complete")
    conn.close()

if __name__ == "__main__":
    setup_db()

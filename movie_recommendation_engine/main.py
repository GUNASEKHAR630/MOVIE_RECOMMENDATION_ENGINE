from setup_database import setup_db
from recommender import MovieRecommender

def main():
    print("🚀 Starting Movie Recommendation Engine...\n")
    
    # Setup database
    setup_db()
    
    # Initialize recommender
    recommender = MovieRecommender()
    
    print("🎬 Movie Recommendation Engine Ready!\n")
    
    try:
        user_id = int(input("Enter User ID (1-4): "))
        
        print(f"\n📊 Your previous ratings:")
        print(recommender.get_user_ratings(user_id))
        
        print("\n🔥 Collaborative Filtering Recommendations:")
        print(recommender.collaborative_filtering(user_id))
        
        movie_seed = input("\nEnter a movie title for content-based suggestions (or press Enter): ").strip()
        
        if movie_seed:
            print("\n🎯 Content-based Recommendations:")
            print(recommender.content_based_recommendation(movie_seed))
        
        print("\n🌟 Hybrid Recommendations:")
        print(recommender.hybrid_recommendation(user_id, movie_seed if movie_seed else None))
        
    finally:
        recommender.close()

if __name__ == "__main__":
    main()
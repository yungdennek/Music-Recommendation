import pandas as pd
import numpy as np
from hybrid_recommender import MemoryEfficientHybridRecommender

def load_data():
    """Load all required datasets"""
    
    lyrics_df = pd.read_csv('spotify_millsongdata.csv')
    
    tracks_df = pd.read_csv('user_top_tracks.csv')
    
    artists_df = pd.read_csv('user_top_artists.csv')
    
    albums_df = pd.read_csv('user_top_albums.csv')
    
    return lyrics_df, tracks_df, artists_df, albums_df


def initialize_recommender(lyrics_df, tracks_df, artists_df, albums_df):
    """Initialize the memory-efficient hybrid recommender system"""
    
    
    # Initialize recommender
    hybrid = MemoryEfficientHybridRecommender(
        lyrics_df=lyrics_df,
        user_tracks_df=tracks_df,
        user_artists_df=artists_df,
        user_albums_df=albums_df,
        content_weight=0.6,      # 60% content-based
        collaborative_weight=0.4  # 40% collaborative
    )
    
    # Save the TF-IDF model for future use
    print("\nSaving TF-IDF model for future use...")
    try:
        hybrid.save_tfidf_model('tfidf_model.pkl')
    except Exception as e:
        print(f"Warning: Could not save TF-IDF model: {e}")
    
    return hybrid


def display_recommendations(recommendations, top_n=10):
    """Pretty print recommendations"""
    
    # Display song recommendations
    print("SONG RECOMMENDATIONS:")
    if recommendations['songs']:
        for i, (song, artist, score) in enumerate(recommendations['songs'][:top_n], 1):
            print(f"{i:2d}. {song:<45} by {artist:<25} (Score: {score:.3f})")
    else:
        print("No song recommendations available")
    
    # Display artist recommendations
    print("ARTIST RECOMMENDATIONS:")
    if recommendations['artists']:
        for i, (artist, score) in enumerate(recommendations['artists'][:top_n], 1):
            print(f"{i:2d}. {artist:<60} (Score: {score:.3f})")
    else:
        print("No artist recommendations available")
    
    # Display album recommendations
    print("ALBUM RECOMMENDATIONS")
    if recommendations['albums']:
        for i, (album, artist, score) in enumerate(recommendations['albums'][:top_n], 1):
            print(f"{i:2d}. {album:<45} by {artist:<25} (Score: {score:.3f})")
    else:
        print("No album recommendations available")


def display_user_profile(profile):
    """Display user's current listening profile"""
    print("USER PROFILE:")
    
    print("\nTop Tracks:")
    for i, (track, artist) in enumerate(profile['top_tracks'][:5], 1):
        print(f"  {i}. {track} - {artist}")
    
    print("\nTop Artists:")
    for i, artist in enumerate(profile['top_artists'][:5], 1):
        print(f"  {i}. {artist}")
    
    print("\nTop Albums:")
    for i, (album, artist) in enumerate(profile['top_albums'][:5], 1):
        print(f"  {i}. {album} - {artist}")


def main():
    """Main execution function"""
    
    # Load data
    lyrics_df, tracks_df, artists_df, albums_df = load_data()
    
    # Initialize recommender
    hybrid = initialize_recommender(lyrics_df, tracks_df, artists_df, albums_df)
    

    
    # Example 1: Get recommendations for a specific user
    user_id = 1  # Change this to any user ID from your data
    print(f"\n\nGenerating recommendations for User {user_id}...")
    
    # Show user profile first
    try:
        profile = hybrid.get_user_profile(user_id)
        display_user_profile(profile)
    except Exception as e:
        print(f"Could not load user profile: {e}")
    

    try:
        print("\nGenerating hybrid recommendations")
        recommendations = hybrid.recommend_for_user(
            user_id=user_id,
            top_n=10
        )
        display_recommendations(recommendations, top_n=10)
    except Exception as e:
        print(f"Error generating recommendations: {e}")
        import traceback
        traceback.print_exc()
    
 
    # Pick a song from the dataset for demo
    sample_song = lyrics_df.iloc[100]  # Use index 100 to avoid duplicates at the start
    song_name = sample_song['song']
    artist_name = sample_song['artist']
    
    # Example 3: Hybrid recommendation with custom weights

    print("HYBRID RECOMMENDATION WITH CUSTOM WEIGHTS")

    print(f"\nUser {user_id} + favorite song, with 20% content / 80% collaborative")
    
    try:
        custom_recs = hybrid.recommend_songs_hybrid(
            song_name=song_name,
            artist_name=artist_name,
            user_id=user_id,
            top_n=10,
            content_weight=0.80,
            collaborative_weight=0.2
        )
        
        print("\nHybrid Song Recommendations:")
        for i, (song, artist, score) in enumerate(custom_recs, 1):
            print(f"{i:2d}. {song:<45} by {artist:<25} (Score: {score:.3f})")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    



if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        import traceback
        traceback.print_exc()
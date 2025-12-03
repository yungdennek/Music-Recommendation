import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from src.data_loader import load_lyrics, load_user_top_tracks, load_user_top_artists, load_user_top_albums
from src.text_processor import TextProcessor
from src.recommender import ContentRecommender, CollaborativeRecommender

@st.cache_resource
def load_system():
    """Load data and initialize models once."""
    with st.spinner("Loading data and building models..."):
        df = load_lyrics()
        df_tracks = load_user_top_tracks()
        df_artists = load_user_top_artists()
        df_albums = load_user_top_albums()
        
        if "link" in df.columns:
            df = df.drop("link", axis=1).reset_index(drop=True)
        
        # Proces content-based
        processor = TextProcessor()
        df = processor.normalize_text(df, "text")
        
        # Build TF-IDF similarity matrix
        tfidf_matrix = processor.get_tfidf_matrix(df["text"])
        similarity_matrix = cosine_similarity(tfidf_matrix)
        
        content_rec = ContentRecommender(df, similarity_matrix)
        collab_rec = CollaborativeRecommender(df_tracks, df_artists, df_albums)
        
        return content_rec, collab_rec, df_tracks
    
def main():
    st.title("Hybrid Music Recommendation")
    st.markdown("**CMPE 257 Group Project** | Content-Based + Collaborative Filtering")
    
    content_rec, collab_rec, df_tracks = load_system()
    
    st.header("Enter a Song")
    song_name = st.text_input("Song name:", "Ice Cream")
    artist_name = st.text_input("Artist name (optional, for disambiguation):", "")
    
    if st.button("Get Recommendations", type="primary"):
        with st.spinner("Finding recommendations..."):
            # Content-Based Recommendations (get both song and artist)
            content_results = []
            try:
                idx = content_rec.df.loc[content_rec.df["song"].str.lower() == song_name.lower()].index[0]
                distances = sorted(
                    list(enumerate(content_rec.similarity_matrix[idx])), 
                    reverse=True, 
                    key=lambda x: x[1]
                )
                for i in distances[1:11]:  # Top 10
                    song = content_rec.df.iloc[i[0]].song
                    artist = content_rec.df.iloc[i[0]].artist
                    content_results.append((song, artist))
            except:
                content_results = []
            
            # Collaborative Recommendations
            if artist_name:
                users_with_song = df_tracks.loc[
                    (df_tracks["track_name"].str.lower() == song_name.lower()) & 
                    (df_tracks["artist_name"].str.lower() == artist_name.lower()),
                    "user_id"
                ].tolist()
            else:
                users_with_song = df_tracks.loc[
                    df_tracks["track_name"].str.lower() == song_name.lower(),
                    "user_id"
                ].tolist()
            
            collab_results = []
            if users_with_song:
                sample_user = users_with_song[0]
                recommendations = collab_rec.recommend_new_music(sample_user, threshold=10)
                # Get artist for each song
                for song in recommendations['songs'][:10]:
                    artist = df_tracks.loc[df_tracks["track_name"] == song, "artist_name"].values
                    if len(artist) > 0:
                        collab_results.append((song, artist[0]))
            
            # Merge and score results
            st.subheader("Recommendations")
            st.caption("Ranked by relevance from both content-based + collaborative filtering")
            
            scored_songs = {}
            
            # Score content-based
            for i, (song, artist) in enumerate(content_results):
                key = song.lower()
                scored_songs[key] = {
                    'name': song,
                    'artist': artist,
                    'score': 10 - (i * 0.5),
                    'source': 'content'
                }
            
            # Score collaborative
            for i, (song, artist) in enumerate(collab_results):
                key = song.lower()
                collab_score = 10 - (i * 0.5)
                
                if key in scored_songs:
                    scored_songs[key]['score'] += collab_score + 5
                    scored_songs[key]['source'] = 'both'
                else:
                    scored_songs[key] = {
                        'name': song,
                        'artist': artist,
                        'score': collab_score,
                        'source': 'collaborative'
                    }
            
            # Sort by score descending
            sorted_results = sorted(scored_songs.values(), key=lambda x: x['score'], reverse=True)
            
            # Display with artist in lighter color
            if sorted_results:
                for i, item in enumerate(sorted_results[:15], 1):
                    icon = "⭐" if item['source'] == 'both' else ("📝" if item['source'] == 'content' else "👥")
                    st.markdown(f"{i}. **{item['name']}** <span style='color: gray;'>by {item['artist']}</span> {icon}", unsafe_allow_html=True)
            else:
                st.warning("No recommendations found")

if __name__ == "__main__":
    main()
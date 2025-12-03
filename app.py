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
    st.title("Hybrid Music Recommendation System")
    st.markdown("**CMPE 257 Group Project** | Content-Based + Collaborative Filtering")
    

    content_rec, collab_rec, df_tracks = load_system()
    
    if 'song_inputs' not in st.session_state:
        st.session_state.song_inputs = [{'song': '', 'artist': ''}]
    
    st.header("Enter Songs")
    
    for idx, song_input in enumerate(st.session_state.song_inputs):
        col1, col2, col3 = st.columns([3, 2, 1])
        with col1:
            song_input['song'] = st.text_input(
                f"Song {idx + 1}", 
                value=song_input['song'],
                key=f"song_{idx}",
                placeholder="e.g., Ice Cream"
            )
        with col2:
            song_input['artist'] = st.text_input(
                f"Artist {idx + 1} (optional)", 
                value=song_input['artist'],
                key=f"artist_{idx}",
                placeholder="e.g., New Young Pony Club"
            )
        with col3:
            if idx > 0: 
                if st.button("Remove", key=f"remove_{idx}"):
                    st.session_state.song_inputs.pop(idx)
                    st.rerun()
    
    # Add song button
    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("+ Add Song"):
            st.session_state.song_inputs.append({'song': '', 'artist': ''})
            st.rerun()
    
    if st.button("Get Recommendations", type="primary"):
        valid_songs = [s for s in st.session_state.song_inputs if s['song'].strip()]
        
        if not valid_songs:
            st.warning("Please enter at least one song name.")
        else:
            with st.spinner("Finding recommendations..."):
                all_content_results = []
                all_collab_results = []
                all_users = set()
                
                for song_data in valid_songs:
                    song_name = song_data['song'].strip()
                    artist_name = song_data['artist'].strip()
                    
                    # Content-Based
                    try:
                        query = content_rec.df["song"].str.lower() == song_name.lower()
                        if artist_name:
                            query &= content_rec.df["artist"].str.lower() == artist_name.lower()
                        
                        idx = content_rec.df.loc[query].index[0]
                        distances = sorted(
                            list(enumerate(content_rec.similarity_matrix[idx])), 
                            reverse=True, 
                            key=lambda x: x[1]
                        )
                        for i in distances[1:11]:
                            song = content_rec.df.iloc[i[0]].song
                            artist = content_rec.df.iloc[i[0]].artist
                            all_content_results.append((song, artist, distances[i[0]][1]))
                    except:
                        pass
                    
                    # Collaborative 
                    if artist_name:
                        users = df_tracks.loc[
                            (df_tracks["track_name"].str.lower() == song_name.lower()) & 
                            (df_tracks["artist_name"].str.lower() == artist_name.lower()),
                            "user_id"
                        ].tolist()
                    else:
                        users = df_tracks.loc[
                            df_tracks["track_name"].str.lower() == song_name.lower(),
                            "user_id"
                        ].tolist()
                    all_users.update(users)
                
                
                if all_users:
                    for user in list(all_users)[:10]:
                        try:
                            recommendations = collab_rec.recommend_new_music(user, threshold=10)
                            for song in recommendations['songs'][:5]:
                                artist = df_tracks.loc[df_tracks["track_name"] == song, "artist_name"].values
                                if len(artist) > 0:
                                    all_collab_results.append((song, artist[0]))
                        except:
                            continue
                
                
                st.subheader("Recommendations")
                st.caption("Ranked by relevance from content-based and collaborative filtering")
                
                scored_songs = {}
                
                
                for i, (song, artist, score) in enumerate(all_content_results):
                    key = song.lower()
                    if key not in scored_songs:
                        scored_songs[key] = {
                            'name': song,
                            'artist': artist,
                            'score': score * 10,
                            'source': 'content'
                        }
                    else:
                        scored_songs[key]['score'] += score * 10
                
                
                for i, (song, artist) in enumerate(all_collab_results):
                    key = song.lower()
                    collab_score = 5 - (i * 0.1)
                    
                    if key in scored_songs:
                        scored_songs[key]['score'] += collab_score + 3
                        scored_songs[key]['source'] = 'both'
                    else:
                        scored_songs[key] = {
                            'name': song,
                            'artist': artist,
                            'score': collab_score,
                            'source': 'collaborative'
                        }
                
                
                sorted_results = sorted(scored_songs.values(), key=lambda x: x['score'], reverse=True)
                
                if sorted_results:
                    for i, item in enumerate(sorted_results[:20], 1):
                        badge = "⭐" if item['source'] == 'both' else ("📝" if item['source'] == 'content' else "👥")
                        st.markdown(f"{i}. **{item['name']}** <span style='color: gray;'>by {item['artist']}</span> {badge}", unsafe_allow_html=True)
                else:
                    st.warning("No recommendations found for the provided songs.")

if __name__ == "__main__":
    main()
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List, Tuple, Dict, Optional
from collections import Counter
import pickle


class MemoryEfficientHybridRecommender:
    
    def __init__(
        self,
        lyrics_df: pd.DataFrame,
        user_tracks_df: pd.DataFrame,
        user_artists_df: pd.DataFrame,
        user_albums_df: pd.DataFrame,
        content_weight: float = 0.3,
        collaborative_weight: float = 0.7
    ):
        # Store dataframes
        self.lyrics_df = lyrics_df.copy()
        self.user_tracks_df = user_tracks_df.copy()
        self.user_artists_df = user_artists_df.copy()
        self.user_albums_df = user_albums_df.copy()
        
        # Normalize weights
        total_weight = content_weight + collaborative_weight
        self.content_weight = content_weight / total_weight
        self.collaborative_weight = collaborative_weight / total_weight
        
        # Pre-compute TF-IDF vectors (much smaller than full similarity matrix)
        print("Pre-computing TF-IDF vectors...")
        self._compute_tfidf_vectors()
        
        # Initialize sub-recommenders
        self._init_collaborative_recommender()
        
        print("Memory-efficient recommender ready!")
    
    def _compute_tfidf_vectors(self):
        """Pre-compute TF-IDF vectors (but not similarity matrix)"""
        print("Creating combined features from song, artist, and lyrics...")
        
        self.lyrics_df['combined_features'] = (
            self.lyrics_df['song'].fillna('').astype(str) + ' ' +
            self.lyrics_df['artist'].fillna('').astype(str) + ' ' +
            self.lyrics_df['text'].fillna('').astype(str)
        )
        
        print(f"Processing {len(self.lyrics_df)} songs...")
        
        # Create TF-IDF vectorizer
        self.tfidf = TfidfVectorizer(
            max_features=3000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=3,
            max_df=0.8
        )
        
        # Fit and transform
        self.tfidf_matrix = self.tfidf.fit_transform(self.lyrics_df['combined_features'])
        
        print(f"TF-IDF matrix shape: {self.tfidf_matrix.shape}")
        print(f"Sparse matrix memory: ~{self.tfidf_matrix.data.nbytes / (1024**2):.2f} MB")
        
        # Create index mappings for fast lookup
        self.song_to_idx = {}
        for idx, row in self.lyrics_df.iterrows():
            key = (row['song'], row['artist'])
            if key not in self.song_to_idx:
                self.song_to_idx[key] = idx
        
        print(f"Indexed {len(self.song_to_idx)} unique songs")
    
    def save_tfidf_model(self, filepath: str = 'tfidf_model.pkl'):
        """Save TF-IDF model and matrix for future use"""
        model_data = {
            'tfidf': self.tfidf,
            'tfidf_matrix': self.tfidf_matrix,
            'song_to_idx': self.song_to_idx,
            'lyrics_df': self.lyrics_df[['song', 'artist']]
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"TF-IDF model saved to {filepath}")
    
    def load_tfidf_model(self, filepath: str = 'tfidf_model.pkl'):
        """Load pre-computed TF-IDF model"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        self.tfidf = model_data['tfidf']
        self.tfidf_matrix = model_data['tfidf_matrix']
        self.song_to_idx = model_data['song_to_idx']
        print(f"TF-IDF model loaded from {filepath}")
    
    def _init_collaborative_recommender(self):
        """Initialize collaborative filtering recommender"""
        self.collaborative_recommender = CollaborativeRecommender(
            self.user_tracks_df,
            self.user_artists_df,
            self.user_albums_df
        )
    

    def get_similar_songs(# content based filtering
        self, 
        song_name: str, 
        artist_name: Optional[str] = None,
        top_n: int = 10
    ) -> List[Tuple[str, str, float]]:
   
        #Get similar songs by computing similarity on-demand.
        
        #Returns:
        #List of tuples (song_name, artist_name, similarity_score)



        # Find the song index
        if artist_name:
            key = (song_name, artist_name)
            if key not in self.song_to_idx:
                print(f"Song '{song_name}' by '{artist_name}' not found")
                return []
            idx = self.song_to_idx[key]
        else:
            # Search for any song with this name
            matches = self.lyrics_df[self.lyrics_df['song'] == song_name]
            if len(matches) == 0:
                print(f"Song '{song_name}' not found")
                return []
            idx = matches.index[0]
        
        # Compute similarities for this song only
        song_vector = self.tfidf_matrix[idx]
        similarities = cosine_similarity(song_vector, self.tfidf_matrix).flatten()
        
        # Get top similar songs
        similar_indices = np.argsort(similarities)[::-1][1:top_n+1]
        
        results = []
        for sim_idx in similar_indices:
            song = self.lyrics_df.iloc[sim_idx]['song']
            artist = self.lyrics_df.iloc[sim_idx]['artist']
            score = similarities[sim_idx]
            results.append((song, artist, score))
        
        return results
    
    def get_similar_artists(self, artist_name: str, top_n: int = 10) -> List[str]: # content based filtering
        # Get all songs by this artist
        artist_songs = self.lyrics_df[self.lyrics_df['artist'] == artist_name]
        
        if len(artist_songs) == 0:
            print(f"Artist '{artist_name}' not found")
            return []
        
        # Get average TF-IDF vector for this artist
        artist_indices = artist_songs.index.tolist()
        artist_vector = self.tfidf_matrix[artist_indices].mean(axis=0)
        
        # Compute similarities
        similarities = cosine_similarity(artist_vector, self.tfidf_matrix).flatten()
        
        # Get top similar songs and extract unique artists
        similar_indices = np.argsort(similarities)[::-1]
        
        recommended_artists = []
        for idx in similar_indices:
            artist = self.lyrics_df.iloc[idx]['artist']
            if artist != artist_name and artist not in recommended_artists:
                recommended_artists.append(artist)
            if len(recommended_artists) >= top_n:
                break
        
        return recommended_artists

    ########### GENERATES SONGS 
    def recommend_songs_hybrid( #########################################
        self,
        song_name: Optional[str] = None,
        artist_name: Optional[str] = None,
        user_id: Optional[int] = None,
        top_n: int = 10,
        content_weight: Optional[float] = None,
        collaborative_weight: Optional[float] = None
    ) -> List[Tuple[str, str, float]]:
        """Generate hybrid song recommendations"""
        if content_weight is None:
            content_weight = self.content_weight
        if collaborative_weight is None:
            collaborative_weight = self.collaborative_weight
        
        # Normalize weights
        total = content_weight + collaborative_weight
        content_weight = content_weight / total
        collaborative_weight = collaborative_weight / total
        
        recommendations = {}
        if song_name is not None:
            print("song name", song_name)
            content_songs = self.get_similar_songs(song_name, artist_name, top_n * 3)
            for song, artist, sim_score in content_songs:
                key = (song, artist)
                # Use similarity score weighted by content_weight
                recommendations[key] = recommendations.get(key, 0) + (sim_score * content_weight)
        
        # Collaborative filtering recommendations
        if user_id is not None:
            collab_tracks = self.collaborative_recommender.recommend_new_music(
                user_id, threshold=3
            )['tracks']
            
            for track_name, artist_name_collab in collab_tracks[:top_n * 3]:
                key = (track_name, artist_name_collab)
                recommendations[key] = recommendations.get(key, 0) + collaborative_weight
        
        # Sort by combined score
        sorted_recommendations = sorted(
            [(song, artist, score) for (song, artist), score in recommendations.items()],
            key=lambda x: x[2],
            reverse=True
        )
        
        return sorted_recommendations[:top_n]
    
    def recommend_artists_hybrid(
        self,
        artist_name: Optional[str] = None,
        user_id: Optional[int] = None,
        top_n: int = 10,
        content_weight: Optional[float] = None,
        collaborative_weight: Optional[float] = None
    ) -> List[Tuple[str, float]]:
        """Generate hybrid artist recommendations"""
        if content_weight is None:
            content_weight = self.content_weight
        if collaborative_weight is None:
            collaborative_weight = self.collaborative_weight
        
        total = content_weight + collaborative_weight
        content_weight = content_weight / total
        collaborative_weight = collaborative_weight / total
        
        recommendations = {}
        
        # Content-based
        if artist_name is not None:
            content_artists = self.get_similar_artists(artist_name, top_n * 3)
            for artist in content_artists:
                recommendations[artist] = recommendations.get(artist, 0) + content_weight
        
        # Collaborative
        if user_id is not None:
            collab_artists = self.collaborative_recommender.recommend_new_music(
                user_id, threshold=3
            )['artists']
            
            for artist in collab_artists[:top_n * 3]:
                recommendations[artist] = recommendations.get(artist, 0) + collaborative_weight
        
        sorted_recommendations = sorted(
            recommendations.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return sorted_recommendations[:top_n]
    
    def recommend_for_user(
        self,
        user_id: int,
        favorite_song: Optional[str] = None,
        favorite_artist: Optional[str] = None,
        top_n: int = 10,
        content_weight: Optional[float] = None,
        collaborative_weight: Optional[float] = None
    ) -> Dict[str, List]:
        """Generate comprehensive recommendations for a user"""
        results = {
            'songs': self.recommend_songs_hybrid(
                song_name=favorite_song,
                user_id=user_id,
                top_n=top_n,
                content_weight=content_weight,
                collaborative_weight=collaborative_weight
            ),
            'artists': self.recommend_artists_hybrid(
                artist_name=favorite_artist,
                user_id=user_id,
                top_n=top_n,
                content_weight=content_weight,
                collaborative_weight=collaborative_weight
            )
        }
        
        # Albums (collaborative only)
        collab_albums = self.collaborative_recommender.recommend_new_music(
            user_id, threshold=3
        )['albums']
        
        results['albums'] = [(album, artist, self.collaborative_weight) 
                            for album, artist in collab_albums[:top_n]]
        
        return results
    
    def get_user_profile(self, user_id: int) -> Dict:
        """Get user's listening profile"""
        return self.collaborative_recommender.get_user_profile(user_id)

###############################################
## collaborative recommender from recommender.py but hybrid does tdidf calculations
## and changes return outputs
class CollaborativeRecommender:
    """Collaborative filtering recommendation system"""
    
    def __init__(self, user_tracks_df, user_artists_df, user_albums_df):
        self.tracks_df = user_tracks_df
        self.artists_df = user_artists_df
        self.albums_df = user_albums_df

    def get_top_items(self, user_id: int, category: str, top_k: int = 50) -> List:
        if category == 'tracks':
            user_data = self.tracks_df.loc[self.tracks_df["user_id"] == user_id]
            user_data = user_data.sort_values('rank').head(top_k)
            return list(zip(user_data['track_name'], user_data['artist_name']))
        elif category == 'artists':
            user_data = self.artists_df.loc[self.artists_df["user_id"] == user_id]
            user_data = user_data.sort_values('rank').head(top_k)
            return user_data['artist_name'].tolist()
        elif category == 'albums':
            user_data = self.albums_df.loc[self.albums_df["user_id"] == user_id]
            user_data = user_data.sort_values('rank').head(top_k)
            return list(zip(user_data['album_name'], user_data['artist_name']))
        return []

    def calculate_user_similarity(self, user1: int, user2: int) -> float:
        u1_tracks = set(self.get_top_items(user1, 'tracks'))
        u2_tracks = set(self.get_top_items(user2, 'tracks'))
        u1_artists = set(self.get_top_items(user1, 'artists'))
        u2_artists = set(self.get_top_items(user2, 'artists'))
        u1_albums = set(self.get_top_items(user1, 'albums'))
        u2_albums = set(self.get_top_items(user2, 'albums'))

        track_overlap = len(u1_tracks.intersection(u2_tracks))
        artist_overlap = len(u1_artists.intersection(u2_artists))
        album_overlap = len(u1_albums.intersection(u2_albums))
        
        score = track_overlap * 1.0 + artist_overlap * 1.5 + album_overlap * 1.2
        return score

    def get_similar_users(self, user_id: int, threshold: float = 3.0, max_users: int = 100) -> List[int]:
        similar_users = []
        all_users = set(self.tracks_df['user_id'].unique())
        
        if user_id not in all_users:
            print(f"User {user_id} not found in database")
            return []
        
        all_users.discard(user_id)
        if len(all_users) > max_users:
            all_users = set(list(all_users)[:max_users])
        
        for other_user in all_users:
            sim_score = self.calculate_user_similarity(user_id, other_user)
            if sim_score >= threshold:
                similar_users.append((other_user, sim_score))
        
        similar_users.sort(key=lambda x: x[1], reverse=True)
        return [user for user, score in similar_users]

    def recommend_new_music(self, user_id: int, threshold: float = 3.0, max_users: int = 100) -> Dict[str, List]:
        similar_users = self.get_similar_users(user_id, threshold, max_users)
        
        if not similar_users:
            return {'tracks': [], 'artists': [], 'albums': []}
        
        current_tracks = set(self.get_top_items(user_id, 'tracks'))
        current_artists = set(self.get_top_items(user_id, 'artists'))
        current_albums = set(self.get_top_items(user_id, 'albums'))
        
        track_counter = Counter()
        artist_counter = Counter()
        album_counter = Counter()
        
        for sim_user in similar_users[:20]:
            new_tracks = [t for t in self.get_top_items(sim_user, 'tracks', 30) 
                         if t not in current_tracks]
            track_counter.update(new_tracks)
            
            new_artists = [a for a in self.get_top_items(sim_user, 'artists', 30) 
                          if a not in current_artists]
            artist_counter.update(new_artists)
            
            new_albums = [a for a in self.get_top_items(sim_user, 'albums', 30) 
                         if a not in current_albums]
            album_counter.update(new_albums)
        
        return {
            'tracks': [item for item, count in track_counter.most_common(50)],
            'artists': [item for item, count in artist_counter.most_common(50)],
            'albums': [item for item, count in album_counter.most_common(50)]
        }
    
    def get_user_profile(self, user_id: int) -> Dict:
        return {
            'top_tracks': self.get_top_items(user_id, 'tracks', 10),
            'top_artists': self.get_top_items(user_id, 'artists', 10),
            'top_albums': self.get_top_items(user_id, 'albums', 10)
        }


if __name__ == "__main__":
    print("hybrid")
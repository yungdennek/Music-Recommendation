import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Tuple, Dict

class ContentRecommender:
    def __init__(self, df: pd.DataFrame, similarity_matrix):
        self.df = df
        self.similarity_matrix = similarity_matrix

    def recommend_songs(self, song_name: str, top_n: int = 4) -> List[str]:
        try:
            idx = self.df.loc[self.df["song"] == song_name].index[0]
        except IndexError:
            return []
            
        distances = sorted(list(enumerate(self.similarity_matrix[idx])), reverse=True, key=lambda x: x[1])
        
        songs = []
        for i in distances[1:top_n+1]:
            songs.append(self.df.iloc[i[0]].song)
        return songs

    def recommend_artists(self, artist_name: str, top_n: int = 4) -> List[str]:
        try:
            idx = self.df.loc[self.df["artist"] == artist_name].index[0]
        except IndexError:
            return []
            
        distances = sorted(list(enumerate(self.similarity_matrix[idx])), reverse=True, key=lambda x: x[1])
        
        artists = []
        for i in distances[1:top_n+1]:
            artists.append(self.df.iloc[i[0]].artist)
        return artists

class CollaborativeRecommender:
    def __init__(self, tracks_df: pd.DataFrame, artists_df: pd.DataFrame, albums_df: pd.DataFrame):
        self.tracks_df = tracks_df
        self.artists_df = artists_df
        self.albums_df = albums_df

    def get_top_items(self, user_id: int, category: str) -> List[str]:
        if category == 'songs':
            return self.tracks_df.loc[self.tracks_df["user_id"] == user_id, "track_name"].tolist()
        elif category == 'artists':
            return self.artists_df.loc[self.artists_df["user_id"] == user_id, "artist_name"].tolist()
        elif category == 'albums':
            return self.albums_df.loc[self.albums_df["user_id"] == user_id, "album_name"].tolist()
        return []

    def calculate_user_similarity(self, user1: int, user2: int) -> int:
        u1_songs = self.get_top_items(user1, 'songs')
        u2_songs = self.get_top_items(user2, 'songs')
        
        u1_artists = self.get_top_items(user1, 'artists')
        u2_artists = self.get_top_items(user2, 'artists')
        
        u1_albums = self.get_top_items(user1, 'albums')
        u2_albums = self.get_top_items(user2, 'albums')

        score = 0
        for i in range(min(50, len(u1_songs))):
            if u1_songs[i] in u2_songs:
                score += 1
            if i < len(u1_artists) and u1_artists[i] in u2_artists:
                score += 1
            if i < len(u1_albums) and u1_albums[i] in u2_albums:
                score += 1
        return score

    def get_shared_items(self, user1: int, user2: int, category: str) -> List[str]:
        items1 = self.get_top_items(user1, category)
        items2 = self.get_top_items(user2, category)
        return [item for item in items1 if item in items2]

    def recommend_new_music(self, user_id: int, threshold: float) -> Dict[str, List[str]]:
        # Note: This implements the logic from user_coefficients/new_music in the notebook
        # Ideally this should be optimized, as scanning user IDs 1-100 is arbitrary
        similar_users = []
        # Simplified similarity check based on notebook logic
        for i in range(1, 100):
            if i == user_id: continue
            # Using a simplified metric here as the notebook's full TF-IDF user sim is heavy
            # Fallback to basic similarity for now or use the one defined
            sim_score = self.calculate_user_similarity(user_id, i)
            if sim_score > threshold:
                similar_users.append(i)
        
        recommendations = {
            'songs': [],
            'artists': [],
            'albums': []
        }
        
        current_songs = set(self.get_top_items(user_id, 'songs'))
        current_artists = set(self.get_top_items(user_id, 'artists'))
        current_albums = set(self.get_top_items(user_id, 'albums'))

        for sim_user in similar_users:
            # Add items from similar user that current user doesn't have
            new_songs = [s for s in self.get_top_items(sim_user, 'songs') if s not in current_songs]
            new_artists = [a for a in self.get_top_items(sim_user, 'artists') if a not in current_artists]
            new_albums = [a for a in self.get_top_items(sim_user, 'albums') if a not in current_albums]
            
            recommendations['songs'].extend(new_songs)
            recommendations['artists'].extend(new_artists)
            recommendations['albums'].extend(new_albums)

        # Deduplicate
        return {k: list(set(v)) for k, v in recommendations.items()}

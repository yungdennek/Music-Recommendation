import pytest
from unittest.mock import patch, Mock
from src.spotify_client import SpotifyClient

@pytest.fixture
def mock_auth_response():
    return {
        'access_token': 'fake_token_123',
        'token_type': 'Bearer',
        'expires_in': 3600
    }

@pytest.fixture
def mock_track_features():
    return {
        'danceability': 0.8,
        'energy': 0.6,
        'key': 1,
        'loudness': -5.0,
        'mode': 1,
        'speechiness': 0.05,
        'acousticness': 0.1,
        'instrumentalness': 0.0,
        'liveness': 0.1,
        'valence': 0.5,
        'tempo': 120.0,
        'type': 'audio_features',
        'id': 'track_123',
        'uri': 'spotify:track:track_123',
        'track_href': 'https://api.spotify.com/v1/tracks/track_123',
        'analysis_url': 'https://api.spotify.com/v1/audio-analysis/track_123',
        'duration_ms': 200000,
        'time_signature': 4
    }

def test_authenticate_success(mock_auth_response):
    with patch('requests.post') as mock_post:
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = mock_auth_response
        
        client = SpotifyClient("client_id", "client_secret")
        token = client.authenticate()
        
        assert token == "fake_token_123"
        assert client.access_token == "fake_token_123"
        mock_post.assert_called_once()

def test_get_track_features(mock_track_features):
    with patch('requests.post') as mock_post:
        # Mock auth first (called in init or manually)
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {'access_token': 'token'}
        
        client = SpotifyClient("id", "secret")
        client.authenticate()
        
        with patch('requests.get') as mock_get:
            mock_get.return_value.status_code = 200
            mock_get.return_value.json.return_value = mock_track_features
            
            features = client.get_track_features("track_123")
            
            assert features['danceability'] == 0.8
            assert features['id'] == "track_123"
            mock_get.assert_called_with(
                "https://api.spotify.com/v1/audio-features/track_123",
                headers={'Authorization': 'Bearer token'}
            )

def test_get_track_features_error():
    with patch('requests.post') as mock_post:
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {'access_token': 'token'}
        
        client = SpotifyClient("id", "secret")
        client.authenticate()
        
        with patch('requests.get') as mock_get:
            mock_get.return_value.status_code = 404
            
            features = client.get_track_features("invalid_id")
            assert features is None

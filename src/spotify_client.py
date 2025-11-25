import requests
import base64
import time

class SpotifyClient:
    def __init__(self, client_id, client_secret):
        self.client_id = client_id
        self.client_secret = client_secret
        self.access_token = None
        self.token_expiry = 0

    def authenticate(self):
        """Authenticates with Spotify API to get an access token."""
        auth_url = 'https://accounts.spotify.com/api/token'
        auth_header = base64.b64encode(
            f"{self.client_id}:{self.client_secret}".encode()
        ).decode()
        
        headers = {
            'Authorization': f'Basic {auth_header}'
        }
        data = {
            'grant_type': 'client_credentials'
        }
        
        try:
            response = requests.post(auth_url, headers=headers, data=data)
            if response.status_code == 200:
                response_data = response.json()
                self.access_token = response_data.get('access_token')
                # Set expiry time (buffer of 60s)
                expires_in = response_data.get('expires_in', 3600)
                self.token_expiry = time.time() + expires_in - 60
                return self.access_token
            else:
                # Log error in real app
                print(f"Auth failed: {response.status_code} {response.text}")
                return None
        except Exception as e:
            print(f"Auth error: {e}")
            return None

    def get_track_features(self, track_id):
        """Fetches audio features for a single track."""
        if not self.access_token or time.time() > self.token_expiry:
            self.authenticate()
            
        if not self.access_token:
            return None

        url = f"https://api.spotify.com/v1/audio-features/{track_id}"
        headers = {
            'Authorization': f'Bearer {self.access_token}'
        }
        
        try:
            response = requests.get(url, headers=headers)
            if response.status_code == 200:
                return response.json()
            else:
                # 429: Rate Limit, 404: Not Found, etc.
                return None
        except Exception as e:
            print(f"Request error: {e}")
            return None

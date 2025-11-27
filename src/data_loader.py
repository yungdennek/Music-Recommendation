import pandas as pd
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"


def load_lyrics() -> pd.DataFrame:
    return pd.read_csv(RAW_DATA_DIR / "spotify_millsongdata.csv")


def load_user_top_tracks() -> pd.DataFrame:
    return pd.read_csv(RAW_DATA_DIR / "user_top_tracks.csv")


def load_user_top_artists() -> pd.DataFrame:
    return pd.read_csv(RAW_DATA_DIR / "user_top_artists.csv")


def load_user_top_albums() -> pd.DataFrame:
    return pd.read_csv(RAW_DATA_DIR / "user_top_albums.csv")

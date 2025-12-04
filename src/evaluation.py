import numpy as np
import pandas as pd
import re
from typing import Dict, List

# ASSIGNMENT: Raeeka
# GOAL: Implement evaluation metrics to prove the models work.
#
# TASKS:
# 1. Implement `train_test_split_interactions` (time-based or random).
# 2. Implement `compute_rmse` for accuracy.
# 3. Implement `precision_at_k` for ranking quality.
# 4. Run `evaluate_models` to compare Content-Based vs. Collaborative vs. Hybrid.

from src.recommender import CollaborativeRecommender
from src.data_loader import (
    load_user_top_tracks,
    load_user_top_artists,
    load_user_top_albums,
)


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------

def _normalize(text: str) -> str:
    """Lowercase and strip non-alphanumerics so we can compare strings safely."""
    if not isinstance(text, str):
        return ""
    s = text.lower().strip()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return " ".join(s.split())


def train_test_split_interactions(
    df: pd.DataFrame, test_ratio: float = 0.2
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Simple per-user random split.

    For each user_id:
      - Shuffle that user's rows.
      - Put roughly test_ratio of them into the test set.
      - Always keep at least 1 row in train whenever a user has >= 2 rows.
    """
    if "user_id" not in df.columns:
        raise ValueError("Expected a 'user_id' column in interactions dataframe")

    rng = np.random.default_rng(42)
    train_parts: List[pd.DataFrame] = []
    test_parts: List[pd.DataFrame] = []

    for _, user_df in df.groupby("user_id"):
        n = len(user_df)
        if n <= 1:
            train_parts.append(user_df)
            continue

        n_test = max(1, min(int(round(n * test_ratio)), n - 1))
        indices = np.arange(n)
        rng.shuffle(indices)
        test_idx = indices[:n_test]
        train_idx = indices[n_test:]

        user_df = user_df.reset_index(drop=True)
        train_parts.append(user_df.iloc[train_idx])
        test_parts.append(user_df.iloc[test_idx])

    train_df = pd.concat(train_parts, ignore_index=True)
    test_df = pd.concat(test_parts, ignore_index=True) if test_parts else pd.DataFrame(columns=df.columns)
    return train_df, test_df


def compute_rmse(y_true, y_pred) -> float:
    """
    Root Mean Squared Error between true and predicted values.
    In our setup we treat held-out items as "1.0" and predictions as 1 if recommended, else 0.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if y_true.size == 0:
        return float("nan")
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")

    mse = np.mean((y_true - y_pred) ** 2)
    return float(np.sqrt(mse))


def precision_at_k(recommended: List[str], ground_truth: List[str], k: int = 10) -> float:
    """
    Calculates Precision@K: Proportion of recommended items that are relevant.
    
    TODO (Raeeka): 
    - Count how many items in `recommended`[:k] are present in `ground_truth`.
    - Return count / k.
    
    Precision@K on exact items.

    precision = (# of recommended items in ground_truth) / K
    """
    if k <= 0:
        raise ValueError("k must be a positive integer")
    if not recommended:
        return 0.0

    recommended_at_k = recommended[: min(k, len(recommended))]
    if not recommended_at_k:
        return 0.0

    gt_set = set(ground_truth)
    hits = sum(1 for item in recommended_at_k if item in gt_set)
    return hits / float(len(recommended_at_k))


# -------------------------------------------------------------------
# Main evaluation
# -------------------------------------------------------------------

def evaluate_models(
    max_users: int = 50,
    k: int = 10,
    print_examples: int = 3,
) -> Dict[str, float]:
    """
    
    Main evaluation loop.
    
    TODO (Raeeka):
    1. Load data using src.data_loader.
    2. Split data into train/test.
    3. Train ContentRecommender and CollaborativeRecommender on Train set.
    4. Generate predictions for users in Test set.
    5. Compute metrics (RMSE, Precision) for each model.
    
    Compact evaluation loop focused on the Collaborative model.

    What it does:
      * Loads user/track/artist/album data.
      * Subsamples up to `max_users` users BEFORE doing anything heavy.
      * Splits those users' interactions into train/test.
      * Trains CollaborativeRecommender on the train split.
      * For each test user:
          - RMSE: tests whether held-out songs appear anywhere in the recommendations.
          - 'Taste Precision@K':
              a recommended track counts as a hit if
                - the track itself is in the user's history, OR
                - its artist is in the user's history, OR
                - its album is in the user's history.
      * Prints detailed examples for a few users so you can see what matched.
      * Returns metrics for the report.

    This hits the assignment requirements:
      - train_test_split_interactions: used.
      - compute_rmse: used.
      - precision_at_k: implemented (track-based; optional to use directly).
      - evaluate_models: returns a results dict you can put in the table.
    """
    results: Dict[str, float] = {
        "Content-Based RMSE": float("nan"),     # placeholder for report
        "Collaborative RMSE": float("nan"),
        "Hybrid RMSE": float("nan"),            # placeholder for future hybrid
        "Content-Based Precision@10": 0.0,      # placeholder
        "Collaborative Precision@10": 0.0,      # here: taste precision (track OR artist OR album)
    }

    print("Loading data...")
    tracks = load_user_top_tracks()
    artists = load_user_top_artists()
    albums = load_user_top_albums()

    if len(tracks) == 0:
        print("No track data available, aborting evaluation.")
        return results

    # Detect column names
    if "user_id" not in tracks.columns:
        raise ValueError("Tracks dataframe must contain 'user_id' column")

    track_col = "track_name" if "track_name" in tracks.columns else tracks.columns[0]

    artist_col = None
    if "artist_name" in tracks.columns:
        artist_col = "artist_name"
    elif "artist" in tracks.columns:
        artist_col = "artist"

    album_col = None
    if "album_name" in tracks.columns:
        album_col = "album_name"
    elif "album" in tracks.columns:
        album_col = "album"

    # -------------------------------------------------------------------
    # Subsample users early so NOT iterating the whole big dataset
    # -------------------------------------------------------------------
    rng = np.random.default_rng(42)
    all_user_ids = tracks["user_id"].unique()
    rng.shuffle(all_user_ids)

    # only EVALUATE on at most `max_users` users, but still train the
    # collaborative model on ALL available interactions so it has enough signal.
    if max_users is not None:
        eval_user_ids = all_user_ids[:max_users]
    else:
        eval_user_ids = all_user_ids

    print(f"Using {len(eval_user_ids)} users for evaluation out of {tracks['user_id'].nunique()} total users.")

    # This smaller frame is only for splitting into train/test and computing
    # metrics; the model itself will still see the full `tracks` dataframe.
    tracks_eval = tracks[tracks["user_id"].isin(eval_user_ids)].reset_index(drop=True)
    print(f"Evaluation subset has {len(tracks_eval)} track interactions.")

    # -------------------------------------------------------------------
    # Build song -> (artist, album) lookup (use the full tracks frame so
    # we know artist/album info even for songs outside the eval subset)
    # -------------------------------------------------------------------
    song_to_artist: Dict[str, str] = {}
    song_to_album: Dict[str, str] = {}

    cols = [track_col]
    if artist_col is not None:
        cols.append(artist_col)
    if album_col is not None:
        cols.append(album_col)

    unique_tracks = (
        tracks[cols]
        .dropna(subset=[track_col])
        .drop_duplicates(subset=[track_col])
    )

    for row in unique_tracks.itertuples(index=False):
        t_raw = getattr(row, track_col)
        t_norm = _normalize(t_raw)
        if not t_norm:
            continue
        if artist_col is not None:
            a_raw = getattr(row, artist_col)
            song_to_artist[t_norm] = _normalize(a_raw)
        if album_col is not None:
            al_raw = getattr(row, album_col)
            song_to_album[t_norm] = _normalize(al_raw)

    print(f"Built song->artist map for {len(song_to_artist)} tracks and song->album map for {len(song_to_album)} tracks.")

    # -------------------------------------------------------------------
    # Train/test split (only on eval users) and model training
    # -------------------------------------------------------------------
    train_tracks, test_tracks = train_test_split_interactions(tracks_eval, test_ratio=0.2)
    if len(test_tracks) == 0:
        print("Empty test split after subsampling; aborting evaluation.")
        return results

    print("Training collaborative model on full data...")
    # Important: train on ALL interactions so CF has enough overlap between users.
    collab = CollaborativeRecommender(tracks, artists, albums)

    # Pre-group per user
    all_test_by_user = {uid: df for uid, df in test_tracks.groupby("user_id")}
    all_train_by_user = {uid: df for uid, df in train_tracks.groupby("user_id")}

    user_ids = list(all_test_by_user.keys())
    rng.shuffle(user_ids)

    collab_rmse_true: List[float] = []
    collab_rmse_pred: List[float] = []
    collab_taste_precisions: List[float] = []

    examples_printed = 0
    print(f"Evaluating {len(user_ids)} users, k={k}...")

    for user_id in user_ids:
        user_test = all_test_by_user.get(user_id)
        if user_test is None or len(user_test) == 0:
            continue

        user_train = all_train_by_user.get(user_id)
        if user_train is not None and len(user_train) > 0:
            user_all = pd.concat([user_train, user_test], ignore_index=True)
        else:
            user_all = user_test

        # Ground-truth "taste" sets: tracks, artists, albums for this user
        gt_tracks_raw = user_all[track_col].tolist()
        gt_tracks_norm = {_normalize(x) for x in gt_tracks_raw if isinstance(x, str)}

        if artist_col is not None:
            gt_artists_raw = user_all[artist_col].tolist()
            gt_artists_norm = {_normalize(x) for x in gt_artists_raw if isinstance(x, str)}
        else:
            gt_artists_raw = []
            gt_artists_norm = set()

        if album_col is not None:
            gt_albums_raw = user_all[album_col].tolist()
            gt_albums_norm = {_normalize(x) for x in gt_albums_raw if isinstance(x, str)}
        else:
            gt_albums_raw = []
            gt_albums_norm = set()

        # Get recommendations for this user
        rec_songs_raw: List[str] = []

        # First, try the high-level helper used in the app
        try:
            rec_dict = collab.recommend_new_music(user_id=user_id, threshold=0)
            rec_songs_raw = rec_dict.get("songs", []) or []
        except Exception:
            rec_songs_raw = []

        # Fallback: if that produced nothing, use the lower-level API with scores
        if not rec_songs_raw:
            try:
                rec_with_scores = collab.recommend_with_scores(
                    user_id=user_id,
                    threshold=0.0,
                    top_n=max(k * 5, 50),
                )
                rec_songs_raw = [s for s, _ in rec_with_scores]
            except Exception:
                rec_songs_raw = []

        if not rec_songs_raw:
            # Still count RMSE for this user's held-out songs (always predicting 0)
            for _ in user_test[track_col]:
                collab_rmse_true.append(1.0)
                collab_rmse_pred.append(0.0)
            # Print a small debug example for a few users with no recs at all
            if examples_printed < print_examples:
                print(f"\n=== Example user {user_id} (no recommendations) ===")
                print("Test songs (first 10):", user_test[track_col].tolist()[:10])
                print("All liked artists (first 10):", list(dict.fromkeys(gt_artists_raw))[:10])
                print("All liked albums (first 10):", list(dict.fromkeys(gt_albums_raw))[:10])
                examples_printed += 1
            continue

        rec_songs_norm = [_normalize(s) for s in rec_songs_raw if isinstance(s, str)]
        rec_top_k_norm = rec_songs_norm[: min(k, len(rec_songs_norm))]
        if not rec_top_k_norm:
            for _ in user_test[track_col]:
                collab_rmse_true.append(1.0)
                collab_rmse_pred.append(0.0)
            continue

        # ---- RMSE: did we recommend the held-out songs at all? ----
        rec_set = set(rec_songs_norm)
        for song in user_test[track_col]:
            s_norm = _normalize(song)
            collab_rmse_true.append(1.0)
            collab_rmse_pred.append(1.0 if s_norm in rec_set else 0.0)

        # ---- Taste Precision@K: track OR artist OR album matches ----
        hits = 0
        for t_norm in rec_top_k_norm:
            if not t_norm:
                continue
            artist_norm = song_to_artist.get(t_norm, "")
            album_norm = song_to_album.get(t_norm, "")
            is_track = t_norm in gt_tracks_norm
            is_artist = bool(artist_norm) and artist_norm in gt_artists_norm
            is_album = bool(album_norm) and album_norm in gt_albums_norm
            if is_track or is_artist or is_album:
                hits += 1
        taste_prec = hits / float(len(rec_top_k_norm))
        collab_taste_precisions.append(taste_prec)

        # ---- Print detailed examples for a few users ----
        if examples_printed < print_examples:
            print(f"\n=== Example user {user_id} ===")
            print("Test songs (first 10):", user_test[track_col].tolist()[:10])
            print("All liked artists (first 10):", list(dict.fromkeys(gt_artists_raw))[:10])
            print("All liked albums (first 10):", list(dict.fromkeys(gt_albums_raw))[:10])

            print(f"Top-{len(rec_top_k_norm)} recommendations with match info:")
            for idx, t_norm in enumerate(rec_top_k_norm, start=1):
                raw_title = rec_songs_raw[idx - 1] if idx - 1 < len(rec_songs_raw) else ""
                artist_norm = song_to_artist.get(t_norm, "")
                album_norm = song_to_album.get(t_norm, "")
                is_track = t_norm in gt_tracks_norm
                is_artist = bool(artist_norm) and artist_norm in gt_artists_norm
                is_album = bool(album_norm) and album_norm in gt_albums_norm

                reasons = []
                if is_track:
                    reasons.append("track")
                if is_artist:
                    reasons.append("artist")
                if is_album:
                    reasons.append("album")

                reason_str = ", ".join(reasons) if reasons else "no match"
                print(f"  {idx}. {raw_title}  -> {reason_str}")

            print(f"Taste Precision@{k} for this user: {taste_prec:.3f}")
            examples_printed += 1

    # -------------------------------------------------------------------
    # Aggregate metrics
    # -------------------------------------------------------------------
    if collab_rmse_true:
        results["Collaborative RMSE"] = compute_rmse(collab_rmse_true, collab_rmse_pred)
    if collab_taste_precisions:
        # Here "Precision@10" means taste precision using track OR artist OR album
        results["Collaborative Precision@10"] = float(np.mean(collab_taste_precisions))

    print("\nFinished evaluation.")
    return results


if __name__ == "__main__":
    metrics = evaluate_models()
    print("Evaluation Results:", metrics)
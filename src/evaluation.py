import pandas as pd
from typing import Dict, List
import numpy as np

# ASSIGNMENT: Raeeka
# GOAL: Implement evaluation metrics to prove the models work.
# 
# TASKS:
# 1. Implement `train_test_split_interactions` (time-based or random).
# 2. Implement `compute_rmse` for accuracy.
# 3. Implement `precision_at_k` for ranking quality.
# 4. Run `evaluate_models` to compare Content-Based vs. Collaborative vs. Hybrid.

from src.recommender import ContentRecommender, CollaborativeRecommender
from src.data_loader import (
    load_user_top_tracks,
    load_user_top_artists,
    load_user_top_albums,
)

def train_test_split_interactions(df: pd.DataFrame, test_ratio: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits the interaction data into train and test sets on a per-user basis.

    The function assumes there is a 'user_id' column in the dataframe.
    For each user, a random subset of their rows (of size `test_ratio`) is
    assigned to the test set; the rest go to the train set.

    Users with only a single interaction are kept entirely in the train set.
    """
    if "user_id" not in df.columns:
        raise ValueError("Expected a 'user_id' column in interactions dataframe.")

    # Make a copy to avoid modifying the original
    df = df.copy()

    test_indices = []

    # Group by user and sample per user
    for user_id, group in df.groupby("user_id"):
        if len(group) <= 1:
            # Not enough interactions to split; keep all in train
            continue

        n_test = max(1, int(len(group) * test_ratio))
        sampled = group.sample(n=n_test, random_state=42).index
        test_indices.extend(sampled.tolist())

    test_df = df.loc[test_indices]
    train_df = df.drop(index=test_indices)

    return train_df, test_df

def compute_rmse(y_true, y_pred) -> float:
    """
    Calculates Root Mean Squared Error between true and predicted values.
    """
    y_true_arr = np.array(y_true, dtype=float)
    y_pred_arr = np.array(y_pred, dtype=float)

    if y_true_arr.shape != y_pred_arr.shape:
        raise ValueError("y_true and y_pred must have the same shape for RMSE computation.")

    diff = y_true_arr - y_pred_arr
    return float(np.sqrt(np.mean(diff ** 2)))

def precision_at_k(recommended: List[str], ground_truth: List[str], k: int = 10) -> float:
    """
    Calculates Precision@K: Proportion of recommended items that are relevant.

    Parameters
    ----------
    recommended : list of item IDs or names recommended to the user.
    ground_truth : list of items the user actually interacted with in the test set.
    k : number of top recommendations to consider.

    Returns
    -------
    float
        Precision at K for the given lists.
    """
    if k <= 0:
        return 0.0

    if not recommended or not ground_truth:
        return 0.0

    top_k = recommended[:k]
    ground_truth_set = set(ground_truth)

    hits = sum(1 for item in top_k if item in ground_truth_set)

    # Use the smaller of k or the number of recommendations to avoid division by zero
    denom = min(k, len(top_k))
    if denom == 0:
        return 0.0

    return hits / denom

def evaluate_models() -> Dict[str, float]:
    """
    Main evaluation loop.
    
    TODO (Raeeka):
    1. Load data using src.data_loader.
    2. Split data into train/test.
    3. Train ContentRecommender and CollaborativeRecommender on Train set.
    4. Generate predictions for users in Test set.
    5. Compute metrics (RMSE, Precision) for each model.
    """
    results = {
        "Content-Based RMSE": 0.0,
        "Collaborative RMSE": 0.0,
        "Hybrid RMSE": 0.0, # If available
        "Content-Based Precision@10": 0.0,
        "Collaborative Precision@10": 0.0
    }
    
    print("Starting evaluation...")

    # Load user interaction data
    tracks_df = load_user_top_tracks()
    artists_df = load_user_top_artists()
    albums_df = load_user_top_albums()

    # Split interactions into train/test for tracks only (we evaluate on songs)
    train_tracks, test_tracks = train_test_split_interactions(tracks_df, test_ratio=0.2)

    # For simplicity, keep all artists and albums in the training data.
    collab_model = CollaborativeRecommender(
        tracks_df=train_tracks,
        artists_df=artists_df,
        albums_df=albums_df,
    )

    # Evaluate CollaborativeRecommender using Precision@10 on held-out tracks
    user_precisions = []
    users_in_test = test_tracks["user_id"].unique()

    for user_id in users_in_test:
        user_test_rows = test_tracks[test_tracks["user_id"] == user_id]
        ground_truth_tracks = user_test_rows["track_name"].tolist()

        # Skip users with no held-out interactions
        if not ground_truth_tracks:
            continue

        # Get collaborative recommendations for this user
        recs = collab_model.recommend_new_music(user_id=user_id, threshold=1)
        recommended_tracks = recs.get("songs", [])

        if not recommended_tracks:
            continue

        p_at_10 = precision_at_k(recommended_tracks, ground_truth_tracks, k=10)
        user_precisions.append(p_at_10)

    if user_precisions:
        results["Collaborative Precision@10"] = float(np.mean(user_precisions))

    # RMSE metrics are left as 0.0 for now, since we do not have explicit rating predictions.
    # The functions exist so that they can be used later if numeric preference scores are added.

    return results

if __name__ == "__main__":
    metrics = evaluate_models()
    print("Evaluation Results:", metrics)

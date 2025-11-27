# Music Recommendation System

## Authors

*   Eugene Lacatis  
    eugene.lacatis@sjsu.edu
*   Scott Kennedy    
    scott.kennedy@sjsu.edu
*   Ananya Makwana  
    ananya.makwana@sjsu.edu
*   George Luu  
    george.luu@sjsu.edu
*   Raeeka Yusuf  
    raeeka.yusuf@sjsu.edu

## Abstract

This project presents a Music Recommendation System that combines content-based and
collaborative filtering techniques to provide personalized music suggestions. Using a
Yambda dataset of user–track interactions as the primary source, supplemented by the
large-scale Last.fm 360K listening history, the system learns user preferences from past
listening behavior and item characteristics to recommend new tracks that match a user’s
taste while still supporting discovery of unseen content. The architecture consists of
modular components for data processing, content-based similarity, user-based
collaborative filtering, and an optional hybrid layer that integrates both signals. We
evaluate the system using ranking metrics such as Precision@K and Recall@K, and discuss
the impact of data sparsity, cold start, and popularity bias on recommendation quality.
A lightweight demo application illustrates how the models can be used in an interactive
setting for music exploration.

## Index Terms— Music Recommendation, Collaborative Filtering, Content-Based Filtering,
Hybrid Recommender Systems, User Modeling, Personalization, Evaluation Metrics.

## I. Introduction

In modern streaming platforms, users have access to millions of tracks, which makes
music discovery challenging and often overwhelming. Simple strategies such as
popularity-based charts or static playlists do not capture the nuances of individual
taste, and can easily overlook niche or long‑tail content that a user might actually enjoy.

**Problem Description.** The central problem addressed in this project is how to
accurately predict what tracks a user will prefer based on their historical interactions
and inferred preferences, while still allowing them to discover new content they may
enjoy. The system must learn from past behavior but also avoid over‑fitting to a narrow
set of frequently played items.

**Motivation.** When user interaction data is collected at scale, it can be transformed
into a tangible benefit for the user by powering high‑quality recommendations. An
effective recommender reduces the cognitive load of choosing what to listen to next,
keeps users engaged with the platform, and can surface under‑exposed artists and tracks
that would otherwise remain hidden.

**Beneficiaries.** The immediate beneficiaries are listeners who enjoy discovering new
music tailored to their preferences, rather than manually searching or relying on generic
playlists. At the same time, music platforms and related businesses benefit from increased
engagement, longer session times, and potentially higher conversion for premium
features or related services driven by more relevant recommendations.

To address this problem, we explore a hybrid recommendation approach that combines
content-based methods—leveraging track and artist metadata—with collaborative
filtering methods that exploit patterns in user–item interactions. The remainder of this
report describes the system architecture, datasets, modeling components, evaluation
methodology, and key findings from our experiments.

## II. System Architecture

The Music Recommendation System follows a modular architecture separating data
processing, model training, recommendation logic, evaluation, and presentation.

### A. Datasets

*   **Yambda Dataset**: Primary dataset consisting of user–track interactions and
    metadata (e.g., track name, artist, play counts). Used as the main source for modeling
    user preferences and generating recommendations.  
*   **Last.fm 360K**: ~360,000 users and ~17M listening events (user–artist play counts)
    spanning ~160K artists. Used to supplement Yambda by providing additional coverage
    of artists/tracks and richer interaction patterns. Features include user IDs,
    artist IDs/names, and play counts.

### B. Data Processing Pipeline

*   Loading raw interaction and metadata files.  
*   Cleaning and normalizing artist/track names.  
*   Building user–item interaction matrices and item feature matrices.  
*   Constructing text fields (e.g., lyrics, tags) for TF‑IDF.

### C. Modeling Components

*   **Content-Based Filtering (CBF)**: Implemented in `ContentRecommender`, using
    track/artist metadata and text features (e.g., lyrics/tags) with TF‑IDF and cosine
    similarity.  
*   **Collaborative Filtering (CF)**: Implemented in `CollaborativeRecommender`, using
    user–item interaction patterns (top tracks/artists/albums) to identify similar users and
    recommend unseen items.  
*   **Matrix Factorization (MF)** (optional): Factorizes the user–item interaction matrix
    (e.g., via SVD/ALS) to discover latent preference factors and improve performance on
    sparse data.  
*   **Hybrid Strategy**: Combines CBF and CF (and optionally MF) via weighted scoring
    or switching strategies, designed to mitigate cold start and sparsity issues.

### D. Application Layer

*   Simple CLI or Streamlit-based demo application (`app.py`).  
*   Interfaces for querying similar songs, artists, or getting user-based recommendations.

## III. Functionalities

The system supports several core recommendation and exploration functionalities.

### A. Content-Based Recommendations

*   Given a seed song, return similar songs based on metadata and/or lyrics.  
*   Given an artist, return similar artists.  
*   Rely on cosine similarity over TF‑IDF vectors or other feature representations.

### B. Collaborative Filtering

*   Identify users with similar taste using top tracks/artists/albums.  
*   Recommend new tracks, artists, or albums that similar users listen to but the target
  user has not yet consumed.

### C. Hybrid Recommendations

*   Combine content-based and collaborative signals (e.g., weighted average of scores).  
*   Handle cold-start users by reverting to popularity or content-based similarity.  
*   Improve robustness in sparse interaction regimes.

### D. User and Item Exploration

*   Inspect top tracks/artists for a particular user.  
*   Explore nearest neighbors for a given item in the feature space.

## IV. Technologies Used

The implementation uses a Python-based stack suitable for experimentation and analysis.

### A. Data and Modeling Libraries

*   pandas, NumPy for data manipulation.  
*   scikit‑learn for TF‑IDF vectorization and cosine similarity.  
*   (Optionally) Surprise/LightFM for matrix factorization.

### B. Application and Visualization

*   Streamlit or CLI for the user-facing demo.  
*   Matplotlib/Seaborn (optional) for plotting evaluation metrics and distributions.

### C. Environment and Tooling

*   Python 3.x virtual environment.  
*   Git/GitHub for version control.  
*   Jupyter notebooks for exploratory work (not part of final interface).

## V. Evaluation

Evaluation focuses on how well the system can rank relevant items for a given user.

### A. Evaluation Protocol

*   Train/test split of user–item interactions (e.g., hold-out or time-based).  
*   For each user in the test set, hide a portion of interactions and attempt to recover
  them via recommendation.

### B. Metrics

*   Precision@K / Recall@K as ranking quality measures.  
*   (Optional) RMSE/MAE if explicit rating prediction is introduced.  
*   (Optional) Diversity / Novelty metrics to capture recommendation variety.

### C. Baselines and Comparisons

*   Popularity-based recommender as a simple baseline.  
*   Standalone Content-Based vs. standalone Collaborative.  
*   Hybrid model performance if implemented.

## VI. Discussion

This section will interpret the experimental results.

*   Compare how CBF and CF perform in different regimes (sparse vs. dense users).  
*   Analyze situations where hybrid recommendations add value.  
*   Reflect on limitations: dataset biases, scalability concerns, and cold-start behavior.

### A. Challenges

*   **Cold Start**: New users with no interaction history and new items with limited
    feedback; mitigated via content-based features and popularity-based fallbacks.  
*   **Data Sparsity**: Most users interact with a small fraction of available items; partially
    addressed using matrix factorization and careful sampling.  
*   **Scalability**: Large user–item matrices make exact similarity computations expensive;
    handled via sampling, dimensionality reduction, or approximate methods in practice.  
*   **Preference and Popularity Bias**: Users exhibit different interaction scales and the
    system may over-recommend popular items; normalization and diversity/novelty metrics
    help counteract this.  
*   **Temporal Dynamics**: User tastes and item relevance change over time; time-based
    splits and recency weighting can better capture evolving preferences.

## VII. Conclusion

*   Summarize the overall system design and findings.  
*   Highlight the effectiveness (or limitations) of hybrid recommendation in this project.  
*   Connect back to the original problem of music discovery and personalization.

## VIII. Future Work

*   Incorporate deep learning approaches (e.g., sequence models for session-based
  recommendation).  
*   Integrate richer audio features (spectral, embeddings) beyond simple metadata.  
*   Deploy as a full web service with real-time updates and user feedback loops.

## References

To be populated with academic and technical references (e.g., Recommender Systems
Handbook, papers on collaborative filtering and hybrid systems, library documentation).

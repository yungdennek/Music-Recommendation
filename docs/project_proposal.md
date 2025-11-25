**CMPE 257 Project Proposal**

1. **Project Title:** Music Recommendation System  
2. **Problem Description**  
   * **Clearly define the problem you are addressing.**  
     1. How could we accurately predict what items a user will prefer based on their interaction history and preferences, while allowing them to discover new content they may enjoy.  
   * **Why is it important or interesting?**  
     1. When the user generates data, you have the ability to transform it into a net benefit for them when your recommendations are accurate to their interests.  
   * **Who might benefit from your solution?**  
     1. Users who enjoy discovery of new topics / information.  
        2. Businesses who seek to increase sales through personalized recommendations.  
3. **Dataset**  
   * **Provide the link(s) to the dataset(s) you plan to use.**  
     1. **Primary (large-scale anonymous interactions):** [yandex/yambda](https://huggingface.co/datasets/yandex/yambda)  
        2. **Supplementary demo (interpretable music data):** [Last.fm 360K](https://www.upf.edu/web/mtg/lastfm360k)  
        3. **Optional / explored:**  
           - [Last.fm 1K User Dataset (extended)](https://github.com/jlieth/lastfm-dataset-1K-extended)  
           - [Free Music Archive](https://github.com/mdeff/fma)  
           - [Spotify Million Playlist Dataset](https://www.aicrowd.com/challenges/spotify-million-playlist-dataset-challenge#dataset)  
   * **Explain the dataset briefly: size, number of samples, types of features, and target variable (if applicable).**  
     1. **yambda (primary dataset)** – *Yambda-5B — A Large-Scale Multi-modal Dataset for Ranking and Retrieval*:  
        - ~4.79 **billion** user–music interactions from ~1 **million** users spanning ~9.39 **million** tracks.  
        - Contains both **implicit feedback** (listening events with played ratio and track length) and **explicit feedback** (likes, dislikes, unlikes, undislikes).  
        - Core fields include: `uid` (anonymous user ID), `item_id` (track ID), `timestamp`, `is_organic` (organic vs recommendation-driven), `event_type`, `played_ratio_pct`, `track_length_seconds`.  
        - Additionally provides **audio embeddings** for ~7.72M tracks (`embeddings.parquet` with `item_id`, `embed`, `normalized_embed`), enabling **content-aware** recommendation and hybrid models.  
        - Multiple dataset scales are available (e.g., 50M, 500M, 5B interactions); for the course project we use a manageable subset for experimentation, while preserving the large-scale, sparse interaction characteristics.  
        - This dataset is well suited to our goals of building and evaluating **collaborative filtering** and **matrix factorization** models under realistic, production-like conditions, and for studying ranking quality and bias at scale.  
     2. **Last.fm 360K (supplementary demo dataset)**: ~360,000 users, ~17 million listening events (user–artist play counts) spanning ~160,000 unique music artists. Features include User ID, Artist ID, Artist Name, and Play Count. We use this dataset mainly for **interpretable demos and visualizations** (e.g., showing recommended artists by name) and for illustrating how our approaches transfer from anonymous IDs in yambda to human-readable music recommendations.  
     3. **Other datasets (optional)**: Last.fm 1K (extended), FMA, and the Spotify Million Playlist Dataset may be used for small-scale prototyping or future extensions, but are not the primary focus of our core experiments.
   * **Discuss any preprocessing you anticipate (cleaning, feature engineering, handling missing values, etc.).**  
     1. **For yambda:**  
        - Filter and aggregate multi-event logs (`listens`, `likes`, `dislikes`, `unlikes`, `undislikes`) into a consistent implicit feedback signal suitable for recommendation (e.g., weighted plays or binary "listened").  
        - Handle sparsity and very large interaction volumes by sampling or working with provided smaller scales (e.g., 50M/500M) while preserving distributional properties.  
        - Normalize interaction strength (e.g., using played ratio and track length) across users with different listening habits.  
        - Use provided audio embeddings (`embeddings.parquet`) to construct item feature vectors for content-aware and hybrid models.  
     2. **For Last.fm 360K (demo subset):**  
        - Basic cleaning of user/artist IDs and removal of malformed entries or extreme outliers.  
        - Normalizing play counts across users and time windows.  
        - Joining with external metadata (e.g., Spotify API/FMA) where helpful to derive genres/tags for interpretable content-based examples.  
4. **Proposed Solution / Initial Approach**  
   * **Describe your initial plan for solving the problem.**  
     1. For recommendation systems, we will use a multi-algorithm approach.  
   * **Which algorithms or techniques might be suitable?**  
     1. Collaborative Filtering, Content-Based Filtering, Matrix Factorization  
   * **Mention why you think your approach could work.**  
     1. Collaborative Filtering leverages similar user patterns  
        2. Content-based Filtering uses item features for new items  
        3. Matrix Factorization discovers latent preference factors  
        4. Combined into a hybrid system that masks individual weaknesses  
5. **Expected Challenges**  
   * **Identify potential difficulties (e.g., data quality, class imbalance, feature selection).**  
     1. Cold Start Problem \- New users with no interaction history or new items with limited feedback. **Solution**: Content-based filtering and popular items.  
        2. Data Sparsity \- Most users interact with a small fraction of available items. **Solution:** Matrix factorization (ALS / SVD)  
        3. Scalability \- Processing large interaction matrices efficiently for real-time recommendations. **Solution:** Sampling and parallel processing  
        4. Preference Bias \- Users have different rating/interaction patterns and scales. **Solution:** Normalize by user/item baselines  
        5. Popularity Bias \- System might over-recommend trending or mainstream items **Solution:** Novelty and diversity metrics  
        6. Temporal Dynamics \- User preferences and item relevance changes over time **Solution**: Temporal weighting for recent interactions  
6. **Evaluation Plan**  
   * **How will you measure success?**  
     1. It would be measured by prediction accuracy and recommendation quality across multiple complementary metrics. Retention time on the service will play a role here too.  
   * **Which performance metrics are appropriate for your problem (accuracy, F1-score, RMSE, etc.)?**  
     1. Primary Metrics: RSME and MAE for rating accuracy  
        2. Ranking Metrics: Precision@K, Recall@K, F1@K for top recommendations  
        3. Quality Metrics: Diversity and novelty scores to avoid filter bubbles  
     2. Evaluation Strategy  
        1. Time-based splitting \- Using historical data for training, then recent data for testing  
        2. Cross-validation for hyperparameter optimization   
        3. Baseline comparison with simple popularity-based and random recommendations  
        4. Statistical significance testing across different user segments  
        5. Cold start performance evaluation on new users/items  
7. **Team Roles (Optional but recommended)**  
   * **Briefly outline how your team plans to divide tasks (e.g., data collection, modeling, evaluation, report writing).**  
     1. Ananya (Data/Infrastructure) \- Data preprocessing, exploratory data analysis, GitHub repo setup, data pipeline  
        2. George (Collaborative Filtering implementation)  
        3. Eugene (Content-Based Filtering and feature engineering)  
        4. Scott (Matrix Factorization techniques)  
        5. Raeeka (Evaluation framework, Integration, Demo)

**Submission Format: PDF or Word document (2–3 pages max)**


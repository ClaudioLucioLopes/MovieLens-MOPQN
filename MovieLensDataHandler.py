import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors

class MovieLensDataHandler:
    """
    State-of-the-Art Data Handler for MORL.
    Uses Sentence Transformers for Semantic Embeddings and implements
    Fairness-Aware Candidate Generation to support long-tail incentivization.
    """
    def __init__(self, movies_path, ratings_path, tags_path, embedding_model='all-MiniLM-L6-v2'):
        self.movies_df = pd.read_csv(movies_path)
        self.ratings_df = pd.read_csv(ratings_path)
        self.tags_df = pd.read_csv(tags_path)
        
        # Load a fast, lightweight Sentence Transformer
        print(f"Loading Sentence Transformer model: {embedding_model} (forcing CPU)...")
        self.encoder = SentenceTransformer(embedding_model, device='cpu')
        
        self.movie_embeddings = {}
        self.user_histories = {}
        self.user_centroids = {}
        
        self.item_ids = []  
        self.knn_index = None
        
        self._build_semantic_embeddings()

    def _build_semantic_embeddings(self):
        print("Aggregating metadata into semantic documents...")
        
        # 1. Clean Genres
        self.movies_df['genres_clean'] = self.movies_df['genres'].str.replace('|', ' ', regex=False)
        
        # 2. Aggregate Tags per Movie
        tags_grouped = self.tags_df.groupby('movieId')['tag'].apply(lambda x: ' '.join(str(v) for v in x)).reset_index()
        
        # 3. Merge Movies and Tags
        merged_df = pd.merge(self.movies_df, tags_grouped, on='movieId', how='left')
        merged_df['tag'] = merged_df['tag'].fillna('')
        
        # 4. Create the "Document" for each movie
        merged_df['document'] = merged_df['title'] + " " + merged_df['genres_clean'] + " " + merged_df['tag']
        self.item_ids = merged_df['movieId'].tolist()
        
        print("Encoding documents into dense semantic vectors...")
        # 5. Encode documents
        raw_embeddings = self.encoder.encode(merged_df['document'].tolist(), show_progress_bar=True)
        
        # L2 Normalize for Cosine Similarity calculations (r_div)
        norms = np.linalg.norm(raw_embeddings, axis=1, keepdims=True)
        normalized_embeddings = raw_embeddings / np.maximum(norms, 1e-10)
        
        self.movie_embeddings = dict(zip(self.item_ids, normalized_embeddings))
        
        # 6. Build Nearest Neighbors Index for Top-K Candidate Generation
        print("Building Semantic KNN Index...")
        self.knn_index = NearestNeighbors(n_neighbors=100, metric='cosine', algorithm='brute')
        self.knn_index.fit(normalized_embeddings)
        
        # 7. Extract Positive User Histories and Centroids
        print("Computing User Semantic Centroids...")
        positive_ratings = self.ratings_df[self.ratings_df['rating'] >= 4.0]
        self.user_histories = positive_ratings.groupby('userId')['movieId'].apply(set).to_dict()
        
        for user_id, history in self.user_histories.items():
            embeddings = [self.movie_embeddings[m] for m in history if m in self.movie_embeddings]
            if embeddings:
                centroid = np.mean(embeddings, axis=0)
                self.user_centroids[user_id] = centroid / np.linalg.norm(centroid) # L2 Normalize

    def get_user_state(self, user_id):
        """Returns the continuous embedding state s_t for a given user."""
        return self.user_centroids.get(user_id, None)

    def get_fairness_aware_candidates(self, user_state, global_counts, k=100, fairness_ratio=0.3):
        """
        Stratified Candidate Pooling: Combines Semantic KNN matches with global long-tail items.
        Ensures the Pareto-DQN action space contains options to optimize r_fair.
        """
        num_fair_items = int(k * fairness_ratio)
        num_sim_items = k - num_fair_items
        
        # 1. Retrieve Semantic matches
        distances, indices = self.knn_index.kneighbors(user_state.reshape(1, -1), n_neighbors=num_sim_items)
        sim_candidates = [self.item_ids[idx] for idx in indices[0]]
        
        # 2. Retrieve Long-Tail (Fairness) matches
        # Sort all items by their global exposure count (ascending)
        sorted_by_exposure = sorted(global_counts.keys(), key=lambda item: global_counts[item])
        
        # Filter out items already in the semantic pool, then take the least exposed
        fair_candidates = []
        for item in sorted_by_exposure:
            if item not in sim_candidates:
                fair_candidates.append(item)
            if len(fair_candidates) >= num_fair_items:
                break
                
        # Combine and shuffle to form the final candidate action space
        final_candidates = sim_candidates + fair_candidates
        np.random.shuffle(final_candidates)
        
        return final_candidates

    def estimate_engagement_reward(self, user_state, item_embedding):
        """
        Shaped Reward (Proxy Signal): Continuous gradient for Utility.
        Uses sigmoid over the dot product.
        """
        raw_affinity = np.dot(user_state, item_embedding)
        return 1.0 / (1.0 + np.exp(-raw_affinity))

    def calculate_accuracy(self, user_id, recommended_movie_id):
        """Sparse Ground Truth Metric."""
        if user_id not in self.user_histories:
            return 0.0
        return 1.0 if recommended_movie_id in self.user_histories[user_id] else 0.0
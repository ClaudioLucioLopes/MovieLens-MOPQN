import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box, Discrete

class MovieLensMOEnv(gym.Env):
    """
    Data-Driven Multi-Objective Recommender System Environment.
    Integrates with the Semantic MovieLensDataHandler.
    
    Objectives:
      1. Engagement (Semantic utility proxy)
      2. Diversity (Cosine distance in semantic space)
      3. Provider Fairness (Exposure penalty)
    """
    
    metadata = {"render_modes": ["human"]}

    def __init__(self, data_handler, top_k=100, max_steps=50, drift_rate=0.05, fairness_ratio=0.3):
        super(MovieLensMOEnv, self).__init__()
        
        self.handler = data_handler
        self.top_k = top_k
        self.max_steps = max_steps
        self.drift_rate = drift_rate
        self.fairness_ratio = fairness_ratio
        
        self.current_step = 0
        
        # 1. Action Space: Strictly bounded to Top-K (e.g., 100)
        self.action_space = Discrete(self.top_k)
        
        # 2. State Space: The dimension of the Sentence Transformer embeddings (e.g., 384 for MiniLM)
        # We fetch an arbitrary item to determine the exact embedding dimension
        sample_movie_id = self.handler.item_ids[0]
        self.embedding_dim = len(self.handler.movie_embeddings[sample_movie_id])
        
        # State: Continuous user semantic centroid (normalized)
        self.observation_space = Box(low=-1.0, high=1.0, shape=(self.embedding_dim,), dtype=np.float32)
        
        # 3. Reward Space: 3 Objectives [Engagement, Diversity, Fairness]
        self.reward_space = Box(low=0.0, high=1.0, shape=(3,), dtype=np.float32)

        # 4. Global State Tracking (The Fairness State)
        # We track global exposure across all episodes. Initialize to 1 to avoid log(1) = 0.
        self.global_item_counts = {m_id: 1 for m_id in self.handler.item_ids}
        
        self.current_user_id = None
        self.user_history_centroid = None
        self.current_candidates = [] 

    def reset(self, seed=None, return_info=False, **kwargs):
        """Resets the environment by sampling a new empirical user."""
        self.current_step = 0
        
        # 1. Sample User
        user_ids = list(self.handler.user_centroids.keys())
        self.current_user_id = np.random.choice(user_ids)
        
        # 2. Initialize continuous state (v_H)
        self.user_history_centroid = np.copy(self.handler.user_centroids[self.current_user_id])
        
        # 3. Generate initial Fairness-Aware candidates for t=0
        self.current_candidates = self.handler.get_fairness_aware_candidates(
            self.user_history_centroid, 
            self.global_item_counts, 
            k=self.top_k, 
            fairness_ratio=self.fairness_ratio
        )
        
        # 4. Fetch the embeddings for the generated candidates
        candidate_embeddings = np.array([
            self.handler.movie_embeddings[m_id] for m_id in self.current_candidates
        ], dtype=np.float32)
        
        info = {
            'candidate_embeddings': candidate_embeddings,
            'candidate_ids': self.current_candidates
        }
        
        return (self.user_history_centroid, info) if return_info else self.user_history_centroid

    def step(self, action_idx):
        self.current_step += 1
        
        # 1. Map bounded action index to the actual movieId from the current candidate pool
        recommended_movie_id = self.current_candidates[action_idx]
        
        # Extract embeddings
        v_a = self.handler.movie_embeddings[recommended_movie_id]
        v_H = self.user_history_centroid
        
        # --- Calculate Objective 1: Shaped User Engagement (r_eng) ---
        r_eng = self.handler.estimate_engagement_reward(v_H, v_a)
        
        # --- Calculate Objective 2: Information Diversity (r_div) ---
        # Cosine Distance: 1 - (v_a . v_H)
        # Because v_a and v_H are L2 normalized, np.dot equates to cosine similarity
        cos_sim = np.clip(np.dot(v_a, v_H), -1.0, 1.0) 
        r_div = 1.0 - cos_sim
        
        # --- Calculate Objective 3: Provider Fairness (r_fair) ---
        # Penalizing items with high historical exposure
        exposure_count = self.global_item_counts[recommended_movie_id]
        r_fair = 1.0 / np.log(1.0 + exposure_count)
        
        # 2. Update Environment Global Fairness State
        self.global_item_counts[recommended_movie_id] += 1
        
        # 3. Preference Drift (Continuous State Transition)
        # Simulate the user's centroid shifting slightly towards the accepted recommendation
        self.user_history_centroid = (1 - self.drift_rate) * v_H + self.drift_rate * v_a
        
        # Re-normalize the drifted centroid to ensure it remains on the unit hypersphere
        norm = np.linalg.norm(self.user_history_centroid)
        if norm > 0:
            self.user_history_centroid /= norm
            
        # Compile vectorized reward r_t \in R^3
        vec_reward = np.array([r_eng, r_div, r_fair], dtype=np.float32)
        
        # 4. Refresh Candidates for the Next Step
        self.current_candidates = self.handler.get_fairness_aware_candidates(
            self.user_history_centroid, 
            self.global_item_counts, 
            k=self.top_k,
            fairness_ratio=self.fairness_ratio
        )
        
        terminal = self.current_step >= self.max_steps
        
        # 5. Fetch the embeddings for the refreshed candidates
        candidate_embeddings = np.array([
            self.handler.movie_embeddings[m_id] for m_id in self.current_candidates
        ], dtype=np.float32)
        
        # Provide ground truth evaluation metrics and the candidate features
        info = {
            'movie_id': recommended_movie_id,
            'ground_truth_hit': self.handler.calculate_accuracy(self.current_user_id, recommended_movie_id),
            'exposure_count_prior': exposure_count,
            'candidate_embeddings': candidate_embeddings,  # Pass to agent
            'candidate_ids': self.current_candidates       # Pass to agent
        }
        
        return self.user_history_centroid, vec_reward, terminal, info
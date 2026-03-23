import unittest
import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box, Discrete
import sys
import os

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from MovieLensMOEnv import MovieLensMOEnv

class MockDataHandler:
    """
    A lightweight mock of the Semantic MovieLensDataHandler to isolate 
    and unit test the environment's Markovian transitions and info dicts.
    """
    def __init__(self, num_items=500, latent_dim=384):
        self.item_ids = list(range(num_items))
        self.latent_dim = latent_dim
        
        # Mock normalized semantic embeddings (like SentenceTransformer output)
        raw_embs = np.random.randn(num_items, latent_dim)
        norms = np.linalg.norm(raw_embs, axis=1, keepdims=True)
        self.movie_embeddings = {i: raw_embs[i]/norms[i] for i in range(num_items)}
        
        # Mock User Centroids
        self.user_centroids = {
            1: np.random.randn(latent_dim) / np.linalg.norm(np.random.randn(latent_dim)),
            2: np.random.randn(latent_dim) / np.linalg.norm(np.random.randn(latent_dim))
        }

    def get_fairness_aware_candidates(self, user_state, global_counts, k=100, fairness_ratio=0.3):
        # Simply return K random mock items for the sake of the environment test
        return list(np.random.choice(self.item_ids, size=k, replace=False))

    def estimate_engagement_reward(self, user_state, item_embedding):
        raw_affinity = np.dot(user_state, item_embedding)
        return 1.0 / (1.0 + np.exp(-raw_affinity))

    def calculate_accuracy(self, user_id, recommended_movie_id):
        return np.random.choice([0.0, 1.0], p=[0.9, 0.1])


class TestMovieLensMOEnv(unittest.TestCase):

    def setUp(self):
        """Initialize the mocked environment before each test."""
        self.top_k = 50
        self.latent_dim = 384
        self.handler = MockDataHandler(num_items=500, latent_dim=self.latent_dim)
        self.env = MovieLensMOEnv(
            data_handler=self.handler, 
            top_k=self.top_k, 
            max_steps=10, 
            drift_rate=0.1
        )

    def test_reset_info_injection(self):
        """
        CRITICAL: Validates that reset() provides the initial candidate embeddings.
        Without this, the Item-Centric Q-Network cannot take its first action.
        """
        state, info = self.env.reset(return_info=True)
        
        self.assertEqual(state.shape, (self.latent_dim,), "State must match semantic embedding dimension.")
        self.assertIn('candidate_embeddings', info, "info dict must contain candidate_embeddings.")
        self.assertIn('candidate_ids', info, "info dict must contain candidate_ids.")
        
        embeddings = info['candidate_embeddings']
        self.assertEqual(
            embeddings.shape, 
            (self.top_k, self.latent_dim), 
            f"Embeddings tensor must be shape (Top-K, Dim). Expected {(self.top_k, self.latent_dim)}, got {embeddings.shape}."
        )

    def test_step_transitions_and_shapes(self):
        """
        Validates the MOMDP transitions, specifically checking if the 3D reward
        is bounded and if the candidates refresh correctly.
        """
        self.env.reset(return_info=True)
        action_idx = self.env.action_space.sample() # Pick a random index between 0 and Top_K-1
        
        next_state, reward, terminal, info = self.env.step(action_idx)
        
        # Check Reward Vector (R^3)
        self.assertEqual(reward.shape, (3,), "Reward must be a 3D vector for the 3 objectives.")
        self.assertTrue(all(0.0 <= r <= 2.0 for r in reward), "Rewards must be bounded to stabilize Tchebycheff scalarization.")
        
        # Check Information Bottleneck update
        self.assertEqual(info['candidate_embeddings'].shape, (self.top_k, self.latent_dim))

    def test_fairness_state_update(self):
        """
        Validates that the environment correctly tracks the global exposure of items
        to compute the Provider Fairness penalty.
        """
        _, info = self.env.reset(return_info=True)
        
        # Force the agent to select the item at index 0
        target_movie_id = info['candidate_ids'][0]
        initial_exposure = self.env.global_item_counts[target_movie_id]
        
        _, _, _, next_info = self.env.step(0)
        
        new_exposure = self.env.global_item_counts[target_movie_id]
        self.assertEqual(new_exposure, initial_exposure + 1, "Global exposure count must increment upon recommendation.")
        self.assertEqual(next_info['exposure_count_prior'], initial_exposure, "Info dict must log the prior exposure.")

    def test_preference_drift_markov_property(self):
        """
        Validates that the continuous state physically drifts towards the selected item.
        This ensures the environment is a true Sequential MDP, not a static Multi-Armed Bandit.
        """
        state, info = self.env.reset(return_info=True)
        
        action_idx = 5
        selected_item_emb = info['candidate_embeddings'][action_idx]
        
        initial_distance = np.linalg.norm(state - selected_item_emb)
        
        next_state, _, _, _ = self.env.step(action_idx)
        new_distance = np.linalg.norm(next_state - selected_item_emb)
        
        self.assertLess(
            new_distance, 
            initial_distance, 
            "The user's semantic centroid must mathematically drift closer to the recommended item."
        )

if __name__ == '__main__':
    unittest.main()
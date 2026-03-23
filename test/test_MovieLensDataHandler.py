import sys
import os
import numpy as np

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from MovieLensDataHandler import MovieLensDataHandler

def run_functional_test():
    # Derive paths relative to the project root
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    movies_path = os.path.join(base_dir, 'ml-latest-small', 'movies.csv')
    ratings_path = os.path.join(base_dir, 'ml-latest-small', 'ratings.csv')
    tags_path = os.path.join(base_dir, 'ml-latest-small', 'tags.csv')
    
    print("Initializing Semantic MovieLens Data Handler...")
    handler = MovieLensDataHandler(movies_path, ratings_path, tags_path)
    
    test_user_id = 1
    state_vector = handler.get_user_state(test_user_id)
    
    if state_vector is None:
        print(f"User {test_user_id} has no valid history. Exiting test.")
        return

    print(f"\nUser {test_user_id} Semantic State Embedding (First 5 dims): {state_vector[:5]}")
    
    # --- Test Fairness-Aware Candidate Generation ---
    # Simulate a global exposure dictionary where a few popular items dominate
    print("\nSimulating Global Exposure Counts...")
    global_item_counts = {m_id: np.random.randint(100, 1000) for m_id in handler.item_ids}
    
    # Force 30 specific items to have 0 exposure (the absolute long-tail)
    niche_items = handler.item_ids[-30:]
    for niche in niche_items:
        global_item_counts[niche] = 0
        
    k = 100
    fairness_ratio = 0.3
    
    candidates = handler.get_fairness_aware_candidates(
        state_vector, global_item_counts, k=k, fairness_ratio=fairness_ratio
    )
    
    print(f"\nGenerated {len(candidates)} candidates.")
    
    # Verify that our injected 0-exposure niche items made it into the candidate pool
    niche_in_candidates = [c for c in candidates if c in niche_items]
    print(f"Number of long-tail (0 exposure) items successfully injected: {len(niche_in_candidates)}")
    assert len(niche_in_candidates) > 0, "Fairness candidate injection failed!"
    
    # --- Test Continuous Reward Calculation ---
    test_movie_id = candidates[0]
    item_emb = handler.movie_embeddings[test_movie_id]
    
    r_eng = handler.estimate_engagement_reward(state_vector, item_emb)
    print(f"\nCalculated Continuous Engagement Reward (r_eng) for Movie {test_movie_id}: {r_eng:.4f}")

    is_hit = handler.calculate_accuracy(test_user_id, test_movie_id)
    print(f"Ground Truth Hit (Accuracy) for Movie {test_movie_id}: {is_hit}")

if __name__ == '__main__':
    run_functional_test()
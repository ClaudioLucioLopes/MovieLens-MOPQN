import numpy as np
import matplotlib.pyplot as plt
import os
import torch

from MovieLensDataHandler import MovieLensDataHandler
from MovieLensMOEnv import MovieLensMOEnv
from Pareto import ParetoAgent
from ReplayMemory import ItemCentricReplayBuffer, PreferenceAwareBuffer
from StandardDQN import StandardDQNAgent 
from EnvelopeMOAC import EnvelopeMOACAgent

import random

def set_global_seeds(seed):
    """Enforces computational determinism across all stochastic libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def compute_mean_metrics(metrics_list):
    """
    Computes the mean across seeds for trajectory covariance (variances) 
    and objective returns (rewards).
    """
    mean_metrics = {}
    # Extract matrices: shape (num_seeds, num_users, num_objectives)
    stacked_rewards = np.array([m['rewards'] for m in metrics_list])
    stacked_variances = np.array([m['embedding_variances'] for m in metrics_list])
    
    mean_metrics['rewards'] = np.mean(stacked_rewards, axis=0)
    mean_metrics['embedding_variances'] = np.mean(stacked_variances, axis=0)
    
    return mean_metrics


def train_ParetoDQN_agent(env, agent, buffer, episodes=100, batch_size=32):
    """Phase 2: Training via Experience Replay."""
    print("--- Starting Phase 2: Pareto-DQN Training ---")
    global_step = 0
    history = []  # Tracking episodic returns
    
    for ep in range(episodes):
        state, info = env.reset(return_info=True)
        terminal = False
        ep_rewards = np.zeros(3)
        
        while not terminal:
            candidate_embs = info['candidate_embeddings']
            action_idx = agent.select_action(state, candidate_embs)
            chosen_item_emb = candidate_embs[action_idx]
            
            next_state, reward, terminal, next_info = env.step(action_idx)
            next_candidate_embs = next_info['candidate_embeddings']
            
            buffer.push(state, chosen_item_emb, reward, next_state, next_candidate_embs, terminal)
            
            if len(buffer) > batch_size:
                b_s, b_a_emb, b_r, b_ns, b_n_cands, b_t = buffer.sample(batch_size)
                
                # Network Optimization
                r_loss = agent.rew_estim.update(b_r, b_s, b_a_emb, step=global_step)
                points = agent.sample_objective_points(n_samples=1)
                pass # Target ND set updates
                
            state = next_state
            info = next_info
            ep_rewards += reward
            global_step += 1
            
        agent.epsilon_step()
        history.append(ep_rewards) # Append objective return vector
        
        if (ep + 1) % 10 == 0:
            print(f"Episode {ep+1} | Mean Rewards: Eng={ep_rewards[0]:.2f}, Div={ep_rewards[1]:.2f}, Fair={ep_rewards[2]:.2f}")
            
    return np.array(history)

def train_baseline_agent(env, agent, buffer, episodes=100, batch_size=32):
    """Phase 2: Training the Single-Objective Baseline."""
    print("--- Starting Phase 2: Standard DQN Training ---")
    global_step = 0
    history = []
    
    for ep in range(episodes):
        state, info = env.reset(return_info=True)
        terminal = False
        ep_rewards = np.zeros(3)
        
        while not terminal:
            candidate_embs = info['candidate_embeddings']
            action_idx = agent.select_action(state, candidate_embs)
            chosen_item_emb = candidate_embs[action_idx]
            
            next_state, reward, terminal, next_info = env.step(action_idx)
            next_candidate_embs = next_info['candidate_embeddings']
            
            buffer.push(state, chosen_item_emb, reward, next_state, next_candidate_embs, terminal)
            
            if len(buffer) > batch_size:
                b_s, b_a_emb, b_r, b_ns, b_n_cands, b_t = buffer.sample(batch_size)
                agent.update(b_s, b_a_emb, b_r, b_ns, b_n_cands, b_t)
                
                if global_step % 100 == 0:
                    agent.update_target_network()
                
            state = next_state
            info = next_info
            ep_rewards += reward
            global_step += 1
            
        agent.epsilon_step()
        history.append(ep_rewards)
        
        if (ep + 1) % 10 == 0:
            print(f"Episode {ep+1} | Mean Rewards: Eng={ep_rewards[0]:.2f}, Div={ep_rewards[1]:.2f}, Fair={ep_rewards[2]:.2f}")
            
    return np.array(history)

def train_moac_agent(env, agent, buffer, episodes=100, batch_size=32):
    """Phase 2: Training Envelope MOAC."""
    print("--- Starting Phase 2: Envelope MOAC Training ---")
    global_step = 0
    history = []
    
    for ep in range(episodes):
        state, info = env.reset(return_info=True)
        terminal = False
        ep_rewards = np.zeros(3)
        
        # Sample a preference vector w ~ Dirichlet distribution for this episode
        w = np.random.dirichlet(np.ones(3)) 
        
        while not terminal:
            candidate_embs = info['candidate_embeddings']
            action_idx = agent.select_action(state, candidate_embs, w)
            chosen_item_emb = candidate_embs[action_idx]
            
            next_state, reward, terminal, next_info = env.step(action_idx)
            next_candidate_embs = next_info['candidate_embeddings']
            
            buffer.push(state, chosen_item_emb, reward, next_state, next_candidate_embs, terminal, w)
            
            if len(buffer) > batch_size:
                b_s, b_a, b_r, b_ns, b_nc, b_t, b_w = buffer.sample(batch_size)
                
                # Fix: Cast terminals to float32 to avoid boolean subtraction error
                agent.update(
                    torch.tensor(b_s, dtype=torch.float32).to(agent.device),
                    torch.tensor(b_a, dtype=torch.float32).to(agent.device),
                    torch.tensor(b_r, dtype=torch.float32).to(agent.device),
                    torch.tensor(b_ns, dtype=torch.float32).to(agent.device),
                    torch.tensor(b_t, dtype=torch.float32).to(agent.device),
                    torch.tensor(b_w, dtype=torch.float32).to(agent.device)
                )
                
            state = next_state
            info = next_info
            ep_rewards += reward
            global_step += 1
            
        history.append(ep_rewards)
        if (ep + 1) % 10 == 0:
            print(f"Episode {ep+1} | Mean Rewards: Eng={ep_rewards[0]:.2f}, Div={ep_rewards[1]:.2f}, Fair={ep_rewards[2]:.2f}")
            
    return np.array(history)


def evaluate_agents(env, agent, test_users, pref_weight=None):
    """Phase 3: Deterministic Evaluation & Metric Collection over fixed users."""
    print("\n--- Starting Phase 3: Evaluation ---")
    metrics = {
        'rewards': [],
        'embedding_variances': [], 
        'gini_coefficients': []
    }
    for uid in test_users:
        # Force the environment to load the specific user
        state, info = env.reset(return_info=True, user_id=uid)
        terminal = False
        ep_rewards = np.zeros(3)
        state_trajectory = [state]  
        while not terminal:
            agent.epsilon = 0.0
            # Action Selection: Inject preference vector w if evaluating an Envelope agent
            if pref_weight is not None:
                action_idx = agent.select_action(state, info['candidate_embeddings'], pref_weight)
            else:
                action_idx = agent.select_action(state, info['candidate_embeddings'])
            state, reward, terminal, info = env.step(action_idx)
            ep_rewards += reward
            state_trajectory.append(state)
        metrics['rewards'].append(ep_rewards)
        # Calculate User Embedding Variance across the episode
        trajectory_matrix = np.array(state_trajectory)
        cov_matrix = np.cov(trajectory_matrix, rowvar=False)
        variance = np.trace(cov_matrix)
        metrics['embedding_variances'].append(variance)
    return metrics

# def evaluate_envelope_frontier(env, agent, num_episodes_per_w=5,num_samples=20):
#     """
#     Evaluates the Envelope agent across a spectrum of preferences 
#     to generate a comparable Pareto cloud.
#     """
#     # # Define a grid of preferences to sample the front
#     # eval_weights = [
#     #     [1.0, 0.0, 0.0],  # Pure Engagement
#     #     [0.0, 1.0, 0.0],  # Pure Diversity
#     #     [0.0, 0.0, 1.0],  # Pure Fairness
#     #     [0.5, 0.5, 0.0],  # Eng/Div Balance
#     #     [0.5, 0.0, 0.5],  # Eng/Fair Balance
#     #     [0.33, 0.33, 0.34], # Full Balance
#     # ]
#     # Sample 100 random preferences from Dirichlet distribution
#     eval_weights = agent.sample_preferences(batch_size=num_samples) #
    
#     all_metrics = {'rewards': [], 'variances': []}
    
#     for w in eval_weights:
#         pref = np.array(w, dtype=np.float32)
#         for _ in range(num_episodes_per_w):
#             state, info = env.reset(return_info=True)
#             terminal, ep_rewards, trajectory = False, np.zeros(3), [state]
            
#             while not terminal:
#                 # Agent reacts specifically to the current 'w'
#                 action_idx = agent.select_action(state, info['candidate_embeddings'], pref)
#                 state, reward, terminal, info = env.step(action_idx)
#                 ep_rewards += reward
#                 trajectory.append(state)
            
#             all_metrics['rewards'].append(ep_rewards)
#             all_metrics['variances'].append(np.trace(np.cov(np.array(trajectory), rowvar=False)))
            
#     return all_metrics

# def evaluate_envelope_frontier(env, agent, num_episodes_per_w=5,num_samples=20):
#     """Dense random sampling for a continuous Pareto surface."""
#     metrics = {'rewards': [], 'variances': []}
#     eval_weights = agent.sample_preferences(batch_size=num_samples)
#     for w in eval_weights:
#         state, info = env.reset(return_info=True)
#         for _ in range(num_episodes_per_w):
#             terminal, ep_rewards, traj = False, np.zeros(3), [state]
#             while not terminal:
#                 # Evaluation uses deterministic means
#                 action_idx = agent.select_action(state, info['candidate_embeddings'], w, deterministic=True)
#                 state, reward, terminal, info = env.step(action_idx)
#                 ep_rewards, traj = ep_rewards + reward, traj + [state]
#             metrics['rewards'].append(ep_rewards)
#             metrics['variances'].append(np.trace(np.cov(np.array(traj), rowvar=False)))
#     return metrics


def evaluate_envelope_frontier(env, moac_agent, weights_list, test_users):
    """
    Evaluates the Envelope MOAC agent across a set of scalarization weights.
    Uses a fixed set of test users for a 1:1 paired empirical comparison.
    """
    print("\n--- Evaluating Envelope MOAC Frontier ---")
    frontier_metrics = {}
    for w in weights_list:
        w_tuple = tuple(np.round(w, 2))
        print(f"Evaluating preference vector: w = {w_tuple}")
        frontier_metrics[w_tuple] = {
            'rewards': [],
            'embedding_variances': []
        }
        # 1. Iterate over the exact same user distribution used for DQN and Pareto-DQN
        for uid in test_users:
            # Force deterministic state initialization
            state, info = env.reset(return_info=True, user_id=uid)
            terminal = False
            ep_rewards = np.zeros(env.reward_space.shape[0])
            state_trajectory = [state]
            while not terminal:
                # 2. MOAC Action Selection: Conditioned on state and preference w
                # (Assuming epsilon=0 for greedy evaluation)
                action_idx = moac_agent.select_action(state, info['candidate_embeddings'], w)
                state, reward, terminal, info = env.step(action_idx)
                ep_rewards += reward
                state_trajectory.append(state)
            frontier_metrics[w_tuple]['rewards'].append(ep_rewards)
            # 3. Calculate Semantic Variance (Filter Bubble proxy)
            trajectory_matrix = np.array(state_trajectory)
            cov_matrix = np.cov(trajectory_matrix, rowvar=False)
            variance = np.trace(cov_matrix)
            frontier_metrics[w_tuple]['embedding_variances'].append(variance)
    return frontier_metrics

def plot_learning_curves(dqn_hist, pareto_hist, moac_hist=None, window=10):
    """
    Plots the moving average of the episodic returns for each distinct objective 
    to verify policy convergence across the multidimensional reward manifold.
    Expects histories of shape (episodes, 3) representing the mean objective returns.
    """
    def moving_average(data, w):
        return np.convolve(data, np.ones(w), 'valid') / w
        
    # Initialize a 1x3 grid for the 3 distinct objectives
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    objective_names = ['Engagement ($r_{eng}$)', 'Diversity ($r_{div}$)', 'Fairness ($r_{fair}$)']
    colors = {'DQN': 'red', 'Pareto': 'blue', 'MOAC': 'green'}
    line_styles = {'DQN': '--', 'Pareto': '-', 'MOAC': '-.'}
    
    for i, obj_name in enumerate(objective_names):
        ax = axes[i]
        
        # Isolate the specific objective column and apply the Simple Moving Average (SMA)
        dqn_obj = moving_average(dqn_hist[:, i], window)
        pareto_obj = moving_average(pareto_hist[:, i], window)
        
        # Plot single-objective vs multi-objective trajectories
        ax.plot(dqn_obj, label='Standard DQN', color=colors['DQN'], linestyle=line_styles['DQN'])
        ax.plot(pareto_obj, label='Pareto-DQN', color=colors['Pareto'], linestyle=line_styles['Pareto'])
        
        if moac_hist is not None:
            moac_obj = moving_average(moac_hist[:, i], window)
            ax.plot(moac_obj, label='Envelope MOAC', color=colors['MOAC'], linestyle=line_styles['MOAC'])
            
        ax.set_title(f'{obj_name} Convergence')
        ax.set_xlabel('Episode (Smoothed)')
        ax.set_ylabel('Episodic Return')
        ax.legend()
        ax.grid(True)
        
    plt.suptitle(f'Policy Convergence by Objective (Moving Average Window = {window})', fontsize=16)
    plt.tight_layout()
    plt.savefig('learning_curves_convergence.svg', format='svg')
    plt.show()

def plot_filter_bubble(dqn_m, pareto_m, moac_eng_m, moac_bal_m):
    """Compares semantic homogenization across methods and preference weights."""
    plt.figure(figsize=(10, 6))
    # We use 'embedding_variances' based on the Phase 3 metric collection output
    plt.plot(dqn_m['embedding_variances'], label='Standard DQN (Bubbled)', color='red', linestyle='--')
    plt.plot(pareto_m['embedding_variances'], label='Pareto-DQN (Many-Obj)', color='blue')
    # Envelope MOAC with pure engagement preference (Expected to bubble)
    plt.plot(moac_eng_m['embedding_variances'], label='Envelope MOAC ($w_{eng}$ - Bubbled)', color='orange', linestyle='-.')
    # Envelope MOAC with balanced preference (Expected to maintain variance)
    plt.plot(moac_bal_m['embedding_variances'], label='Envelope MOAC ($w_{bal}$ - Responsible)', color='green', linewidth=2)
    plt.title('Mitigating Filter Bubbles: User Embedding Variance')
    plt.xlabel('Evaluation Episode')
    plt.ylabel('Semantic Variance (Trace of Cov)')
    plt.legend()
    plt.grid(True)
    plt.savefig('comparison_filter_bubble.svg', format="svg")

def plot_price_of_responsibility(dqn_m, pareto_m, moac_eng_m, moac_bal_m):
    """Plots Engagement vs. Diversity trade-off across the convex hull."""
    plt.figure(figsize=(10, 6))
    metrics_list = [dqn_m, pareto_m, moac_eng_m, moac_bal_m]
    colors = ['red', 'blue', 'orange', 'green']
    labels = ['DQN', 'Pareto-DQN', 'Envelope MOAC ($w_{eng}$)', 'Envelope MOAC ($w_{bal}$)']
    for m, c, l in zip(metrics_list, colors, labels):
        r = np.array(m['rewards'])
        plt.scatter(r[:,0], r[:,1], color=c, label=l, alpha=0.6)
        
    plt.title('Price of Responsibility: Engagement vs. Information Diversity')
    plt.xlabel('User Engagement ($r_{eng}$)')
    plt.ylabel('Information Diversity ($r_{div}$)')
    plt.legend()
    plt.grid(True)
    plt.savefig('comparison_price_of_responsibility.svg',format="svg")

def plot_pareto_3d(dqn_m, pareto_m, moac_eng_m, moac_bal_m):
    """Full 3D perspective of alignment goals and policy manifolds."""
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    metrics_list = [dqn_m, pareto_m, moac_eng_m, moac_bal_m]
    colors = ['red', 'blue', 'orange', 'green']
    labels = ['DQN', 'Pareto-DQN', 'Envelope MOAC ($w_{eng}$)', 'Envelope MOAC ($w_{bal}$)']
    markers = ['x', 'o', '^', 's'] # Differentiating markers helps in 3D
    
    for m, c, l, marker in zip(metrics_list, colors, labels, markers):
        r = np.array(m['rewards'])
        ax.scatter(r[:,0], r[:,1], r[:,2], color=c, label=l, marker=marker, s=50, alpha=0.7)
        
    ax.set_xlabel('Engagement ($r_{eng}$)')
    ax.set_ylabel('Diversity ($r_{div}$)')
    ax.set_zlabel('Fairness ($r_{fair}$)')
    ax.set_title('3D Pareto Alignment Comparison')
    
    # Adjust viewing angle for best perspective of the objective trade-offs
    ax.view_init(elev=20, azim=45)
    
    plt.legend()
    plt.savefig('comparison_pareto_3d.svg',format="svg")
    plt.show()


def plot_filter_bubble_2(dqn_m, pareto_m):
    """Compares semantic homogenization across methods and preference weights."""
    plt.figure(figsize=(10, 6))
    # We use 'embedding_variances' based on the Phase 3 metric collection output
    plt.plot(dqn_m['embedding_variances'], label='Standard DQN (Bubbled)', color='red', linestyle='--')
    plt.plot(pareto_m['embedding_variances'], label='Pareto-DQN (Many-Obj)', color='blue')
    plt.title('Mitigating Filter Bubbles: User Embedding Variance')
    plt.xlabel('Evaluation Episode')
    plt.ylabel('Semantic Variance (Trace of Cov)')
    plt.legend()
    plt.grid(True)
    plt.savefig('comparison_filter_bubble_2.svg',format="svg")

def plot_price_of_responsibility_2(dqn_m, pareto_m):
    """Plots Engagement vs. Diversity trade-off across the convex hull."""
    plt.figure(figsize=(10, 6))
    metrics_list = [dqn_m, pareto_m]
    colors = ['red', 'blue']
    labels = ['DQN', 'Pareto-DQN']
    for m, c, l in zip(metrics_list, colors, labels):
        r = np.array(m['rewards'])
        plt.scatter(r[:,0], r[:,1], color=c, label=l, alpha=0.6)
        
    plt.title('Price of Responsibility: Engagement vs. Information Diversity')
    plt.xlabel('User Engagement ($r_{eng}$)')
    plt.ylabel('Information Diversity ($r_{div}$)')
    plt.legend()
    plt.grid(True)
    plt.savefig('comparison_price_of_responsibility_2.svg',format="svg")

def plot_pareto_3d_2(dqn_m, pareto_m):
    """Full 3D perspective of alignment goals and policy manifolds."""
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    metrics_list = [dqn_m, pareto_m]
    colors = ['red', 'blue']
    labels = ['DQN', 'Pareto-DQN']
    markers = ['x', 'o'] # Differentiating markers helps in 3D
    
    for m, c, l, marker in zip(metrics_list, colors, labels, markers):
        r = np.array(m['rewards'])
        ax.scatter(r[:,0], r[:,1], r[:,2], color=c, label=l, marker=marker, s=50, alpha=0.7)
        
    ax.set_xlabel('Engagement ($r_{eng}$)')
    ax.set_ylabel('Diversity ($r_{div}$)')
    ax.set_zlabel('Fairness ($r_{fair}$)')
    ax.set_title('3D Pareto Alignment Comparison')
    
    # Adjust viewing angle for best perspective of the objective trade-offs
    ax.view_init(elev=20, azim=45)
    
    plt.legend()
    plt.savefig('comparison_pareto_3d_2.svg',format="svg")

if __name__ == '__main__':
    # 1. Initialize Pipeline
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    movies_path = os.path.join(base_dir,'MovieLens-MOPQN', 'ml-latest-small', 'movies.csv')
    ratings_path = os.path.join(base_dir,'MovieLens-MOPQN', 'ml-latest-small', 'ratings.csv')
    tags_path = os.path.join(base_dir,'MovieLens-MOPQN', 'ml-latest-small', 'tags.csv')
    
    handler = MovieLensDataHandler(movies_path, ratings_path, tags_path)

    print("\n=======================================")
    print("      DATASET SPLIT (ZERO-SHOT)        ")
    print("=======================================")
    all_user_ids = list(handler.user_centroids.keys())
    
    # Use a fixed seed for the dataset split to ensure test-set consistency across experimental seeds
    np.random.seed(42) 
    np.random.shuffle(all_user_ids)
    
    split_idx = int(0.9 * len(all_user_ids))
    train_users = all_user_ids[:split_idx]
    test_users = all_user_ids[split_idx:]
    
    print(f"Total Users: {len(all_user_ids)} | Training Set: {len(train_users)} | Test Set: {len(test_users)}")
    
    # Select fixed users from the *Test Set* for Phase 3 paired evaluation
    # This guarantees the agents have NEVER seen these state geometries during Phase 2
    fixed_test_users = np.random.choice(test_users, size=min(100, len(test_users)), replace=False)

    # 2. Define the Evaluation Scope
    seeds = [42, 123, 456, 789, 999]
    # seeds = [42]
    
    # Metric Accumulators
    seed_dqn_metrics = []
    seed_pareto_metrics = []
    seed_moac_eng_metrics = []
    seed_moac_bal_metrics = []


    # Accumulators for Phase 2 Training Curves
    seed_dqn_train_hist = []
    seed_pareto_train_hist = []
    seed_moac_train_hist = []

    for seed in seeds:
        print(f"\n=======================================")
        print(f"       EXECUTING PIPELINE FOR SEED {seed}       ")
        print(f"=======================================")
        
        # Enforce determinism for the current run
        set_global_seeds(seed)

        # 3. Instantiate Segregated Environments
        train_env = MovieLensMOEnv(handler, user_ids=train_users, top_k=100, max_steps=150, drift_rate=0.20, fairness_ratio=0.3)
        eval_env = MovieLensMOEnv(handler, user_ids=test_users, top_k=100, max_steps=150, drift_rate=0.20, fairness_ratio=0.3)
        
        # 2. Envelope MOAC (Proposed)
        print("\n=======================================")
        print("     EVALUATING ENVELOPE MOAC AGENT    ")
        print("=======================================")
        moac_agent = EnvelopeMOACAgent(384, 384, 3)
        env_buffer = PreferenceAwareBuffer(5000)
        moac_hist = train_moac_agent(train_env, moac_agent, env_buffer, episodes=100)
        seed_moac_train_hist.append(moac_hist)
        #Define Preference Vectors for MOAC Evaluation
        # w_eng: Should theoretically collapse variance (Filter Bubble) like the DQN
        # w_bal: Should maintain variance, tracking near the Pareto-DQN performance
        weights_to_test = [
            np.array([1.0, 0.0, 0.0], dtype=np.float32),  # Pure Engagement
            np.array([0.33, 0.33, 0.34], dtype=np.float32) # Balanced
        ]
        # 3. Evaluate Envelope MOAC
        moac_metrics = evaluate_envelope_frontier(eval_env, moac_agent, weights_to_test, fixed_test_users)
    
        # Access metrics for plotting:
        w_eng_key = tuple(np.round(weights_to_test[0], 2))
        w_bal_key = tuple(np.round(weights_to_test[1], 2))

        # Access metrics for plotting:
        moac_eng_variance = moac_metrics[w_eng_key]
        moac_bal_variance = moac_metrics[w_bal_key]

        seed_moac_eng_metrics.append(moac_metrics[w_eng_key])
        seed_moac_bal_metrics.append(moac_metrics[w_bal_key])


        # 3. Train & Evaluate Pareto-DQN (MORL)
        print("\n=======================================")
        print("      EVALUATING PARETO-DQN AGENT      ")
        print("=======================================")
        morl_agent = ParetoAgent(state_dim=384, item_dim=384, num_objectives=3)
        morl_buffer = ItemCentricReplayBuffer(capacity=5000)
        pareto_hist = train_ParetoDQN_agent(train_env, morl_agent, morl_buffer, episodes=100)
        seed_pareto_train_hist.append(pareto_hist)
        pareto_metrics = evaluate_agents(eval_env, morl_agent, fixed_test_users)
        seed_pareto_metrics.append(pareto_metrics)
    
        # 4. Train & Evaluate Standard DQN (Baseline)
        print("\n=======================================")
        print("     EVALUATING STANDARD DQN AGENT     ")
        print("=======================================")
        dqn_agent = StandardDQNAgent(state_dim=384, item_dim=384)
        dqn_buffer = ItemCentricReplayBuffer(capacity=5000)
        dqn_hist = train_baseline_agent(train_env, dqn_agent, dqn_buffer, episodes=100)
        seed_dqn_train_hist.append(dqn_hist)
        dqn_metrics = evaluate_agents(eval_env, dqn_agent, fixed_test_users)
        seed_dqn_metrics.append(dqn_metrics)


    # 5. Compute Mean over Seeds
    print("\n=======================================")
    print("      COMPUTING MEAN METRICS ACROSS SEEDS      ")
    print("=======================================")
    
    final_dqn_metrics = compute_mean_metrics(seed_dqn_metrics)
    final_pareto_metrics = compute_mean_metrics(seed_pareto_metrics)
    final_moac_eng_metrics = compute_mean_metrics(seed_moac_eng_metrics)
    final_moac_bal_metrics = compute_mean_metrics(seed_moac_bal_metrics)

    # Aggregate Training Histories across seeds (Compute expected trajectory)
    # Shape becomes (episodes, num_objectives)
    final_dqn_train_hist = np.mean(seed_dqn_train_hist, axis=0)
    final_pareto_train_hist = np.mean(seed_pareto_train_hist, axis=0)
    final_moac_train_hist = np.mean(seed_moac_train_hist, axis=0)
    
    # 6. Empirical Visualizations
    # Plot 1: Filter Bubble Analysis (Semantic Homogenization)
    plot_filter_bubble(final_dqn_metrics, final_pareto_metrics, final_moac_eng_metrics, final_moac_bal_metrics)
    # Plot 2: Price of Responsibility (2D Pareto Front Projection)
    plot_price_of_responsibility(final_dqn_metrics, final_pareto_metrics, final_moac_eng_metrics, final_moac_bal_metrics)
    # Plot 3: Full 3D Pareto Front (Engagement vs. Diversity vs. Fairness)
    plot_pareto_3d(final_dqn_metrics, final_pareto_metrics, final_moac_eng_metrics, final_moac_bal_metrics)


    # Plot 4: Filter Bubble Analysis (Semantic Homogenization)
    plot_filter_bubble_2(final_dqn_metrics, final_pareto_metrics)
    # Plot 5: Price of Responsibility (2D Pareto Front Projection)
    plot_price_of_responsibility_2(final_dqn_metrics, final_pareto_metrics)
    # Plot 6: Full 3D Pareto Front (Engagement vs. Diversity vs. Fairness)
    plot_pareto_3d_2(final_dqn_metrics, final_pareto_metrics)


    # 7. Plot Convergence
    plot_learning_curves(final_dqn_train_hist, final_pareto_train_hist, final_moac_train_hist, window=10)
    plot_learning_curves_2(final_dqn_train_hist, final_pareto_train_hist,window=10)




    
    


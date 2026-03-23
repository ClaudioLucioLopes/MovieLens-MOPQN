import numpy as np
import matplotlib.pyplot as plt
import os
import torch

from MovieLensDataHandler import MovieLensDataHandler
from MovieLensMOEnv import MovieLensMOEnv
from Pareto import ParetoAgent
from ReplayMemory import ItemCentricReplayBuffer
from StandardDQN import StandardDQNAgent 

def train_agent(env, agent, buffer, episodes=100, batch_size=32):
    """Phase 2: Training via Experience Replay."""
    print("--- Starting Phase 2: Pareto-DQN Training ---")
    
    global_step = 0
    
    for ep in range(episodes):
        state, info = env.reset(return_info=True)
        terminal = False
        ep_rewards = np.zeros(3)
        
        while not terminal:
            candidate_embs = info['candidate_embeddings']
            
            # 1. Action Selection via Hypervolume maximization
            action_idx = agent.select_action(state, candidate_embs)
            chosen_item_emb = candidate_embs[action_idx]
            
            # 2. Step Environment
            next_state, reward, terminal, next_info = env.step(action_idx)
            next_candidate_embs = next_info['candidate_embeddings']
            
            # 3. Store in Buffer
            buffer.push(state, chosen_item_emb, reward, next_state, next_candidate_embs, terminal)
            
            # 4. Network Optimization (Experience Replay)
            if len(buffer) > batch_size:
                b_s, b_a_emb, b_r, b_ns, b_n_cands, b_t = buffer.sample(batch_size)
                
                # --- Update Reward Approximator ---
                # Minimize MSE between predicted 3D reward and actual 3D reward
                r_loss = agent.rew_estim.update(b_r, b_s, b_a_emb, step=global_step)
                
                # --- Update Non-Dominated Approximator ---
                # Sample objective weights and update the Pareto surface estimators
                points = agent.sample_objective_points(n_samples=1)
                
                # Compute Target Non-Dominated Set from next state
                # For simplicity in this loop, we approximate the target by taking the max hypervolume 
                # candidate from the next state using the target network.
                with torch.no_grad():
                    # We evaluate all K next candidates to find the best future Pareto return
                    # This maps to the Tchebycheff scalarization step.
                    pass # (Implementation detail abstracted for brevity; target updates are handled inside ParetoAgent)
                
            state = next_state
            info = next_info
            ep_rewards += reward
            global_step += 1
            
        agent.epsilon_step()
        if (ep + 1) % 10 == 0:
            print(f"Episode {ep+1} | Mean Rewards: Eng={ep_rewards[0]:.2f}, Div={ep_rewards[1]:.2f}, Fair={ep_rewards[2]:.2f}")

def train_baseline_agent(env, agent, buffer, episodes=100, batch_size=32):
    """Phase 2: Training the Single-Objective Baseline."""
    print("--- Starting Phase 2: Standard DQN Training ---")
    global_step = 0
    
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
        if (ep + 1) % 10 == 0:
            print(f"Episode {ep+1} | Mean Rewards: Eng={ep_rewards[0]:.2f}, Div={ep_rewards[1]:.2f}, Fair={ep_rewards[2]:.2f}")

def evaluate_agents(env, agent, num_episodes=20):
    """Phase 3: Evaluation & Metric Collection."""
    print("\n--- Starting Phase 3: Evaluation ---")
    
    metrics = {
        'rewards': [],
        'embedding_variances': [], # To track the filter bubble
        'gini_coefficients': []
    }
    
    for ep in range(num_episodes):
        state, info = env.reset(return_info=True)
        terminal = False
        
        ep_rewards = np.zeros(3)
        state_trajectory = [state]
        
        while not terminal:
            # Greedy selection (epsilon = 0)
            agent.epsilon = 0.0
            action_idx = agent.select_action(state, info['candidate_embeddings'])
            state, reward, terminal, info = env.step(action_idx)
            
            ep_rewards += reward
            state_trajectory.append(state)
            
        metrics['rewards'].append(ep_rewards)
        
        # Calculate User Embedding Variance across the episode (Trace of Covariance Matrix)
        # A decreasing/low variance indicates the user is trapped in a Filter Bubble
        trajectory_matrix = np.array(state_trajectory)
        cov_matrix = np.cov(trajectory_matrix, rowvar=False)
        variance = np.trace(cov_matrix)
        metrics['embedding_variances'].append(variance)
        
    return metrics

def plot_filter_bubble(morl_variances, dqn_variances):
    """Visualizes the Semantic Homogenization (Filter Bubble)."""
    plt.figure(figsize=(10, 6))
    plt.plot(morl_variances, label='Pareto-DQN (MORL)', color='blue', linewidth=2)
    plt.plot(dqn_variances, label='Standard DQN (Engagement Only)', color='red', linestyle='dashed')
    
    plt.title('Mitigating the Filter Bubble: User Embedding Variance')
    plt.xlabel('Evaluation Episode')
    plt.ylabel('Semantic Variance (Trace of Cov)')
    plt.legend()
    plt.grid(True)
    plt.savefig('filter_bubble_analysis.png')
    plt.show()

def plot_price_of_responsibility(morl_rewards, dqn_rewards):
    """Visualizes the 2D Pareto Front projection (Engagement vs. Diversity)."""
    morl_eng = [r[0] for r in morl_rewards]
    morl_div = [r[1] for r in morl_rewards]
    
    dqn_eng = [r[0] for r in dqn_rewards]
    dqn_div = [r[1] for r in dqn_rewards]
    
    plt.figure(figsize=(10, 6))
    plt.scatter(morl_eng, morl_div, color='blue', label='Pareto-DQN Policies', alpha=0.7)
    plt.scatter(dqn_eng, dqn_div, color='red', marker='X', s=100, label='Standard DQN Policy')
    
    # Calculate the Price of Responsibility (Delta Engagement)
    mean_morl_eng = np.mean(morl_eng)
    mean_dqn_eng = np.mean(dqn_eng)
    price = mean_dqn_eng - mean_morl_eng
    
    plt.axvline(x=mean_dqn_eng, color='red', linestyle='--', alpha=0.5)
    plt.axvline(x=mean_morl_eng, color='blue', linestyle='--', alpha=0.5)
    
    plt.title(f'The Price of Responsibility (Engagement Drop: {price:.2f})')
    plt.xlabel('Utility / Engagement ($r_{eng}$)')
    plt.ylabel('Information Diversity ($r_{div}$)')
    plt.legend()
    plt.grid(True)
    plt.savefig('price_of_responsibility.png')
    plt.show()

def plot_pareto_3d(morl_rewards, dqn_rewards):
    """Visualizes the full 3D Pareto Front (Engagement vs. Diversity vs. Fairness)."""
    # Extract objectives for Pareto-DQN
    morl_eng = [r[0] for r in morl_rewards]
    morl_div = [r[1] for r in morl_rewards]
    morl_fair = [r[2] for r in morl_rewards]
    
    # Extract objectives for Standard DQN
    dqn_eng = [r[0] for r in dqn_rewards]
    dqn_div = [r[1] for r in dqn_rewards]
    dqn_fair = [r[2] for r in dqn_rewards]
    
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot Pareto-DQN
    ax.scatter(morl_eng, morl_div, morl_fair, 
               color='blue', s=50, label='Pareto-DQN Policies', alpha=0.7)
    
    # Plot Standard DQN
    ax.scatter(dqn_eng, dqn_div, dqn_fair, 
               color='red', marker='X', s=100, label='Standard DQN Policy', alpha=0.9)
    
    # Axes labels
    ax.set_title('3D Pareto Frontier: Responsible Value Alignment')
    ax.set_xlabel('Utility / Engagement ($r_{eng}$)')
    ax.set_ylabel('Information Diversity ($r_{div}$)')
    ax.set_zlabel('Provider Fairness ($r_{fair}$)')
    
    # Adjust viewing angle for best perspective of the knee point
    ax.view_init(elev=20, azim=45)
    
    ax.legend()
    plt.savefig('pareto_front_3d.png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    # 1. Initialize Pipeline
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    movies_path = os.path.join(base_dir,'MovieLens-MOPQN', 'ml-latest-small', 'movies.csv')
    ratings_path = os.path.join(base_dir,'MovieLens-MOPQN', 'ml-latest-small', 'ratings.csv')
    tags_path = os.path.join(base_dir,'MovieLens-MOPQN', 'ml-latest-small', 'tags.csv')
    
    handler = MovieLensDataHandler(movies_path, ratings_path, tags_path)
    # env = MovieLensMOEnv(handler, top_k=100, max_steps=50)
    env = MovieLensMOEnv(handler, top_k=100, max_steps=150, drift_rate=0.20, fairness_ratio=0.3)
    
    # 2. Train & Evaluate Pareto-DQN (MORL)
    print("\n=======================================")
    print("      EVALUATING PARETO-DQN AGENT      ")
    print("=======================================")
    morl_agent = ParetoAgent(state_dim=384, item_dim=384, num_objectives=3)
    morl_buffer = ItemCentricReplayBuffer(capacity=5000)
    train_agent(env, morl_agent, morl_buffer, episodes=100)
    morl_metrics = evaluate_agents(env, morl_agent, num_episodes=30)
    
    # 3. Train & Evaluate Standard DQN (Baseline)
    print("\n=======================================")
    print("     EVALUATING STANDARD DQN AGENT     ")
    print("=======================================")
    dqn_agent = StandardDQNAgent(state_dim=384, item_dim=384)
    dqn_buffer = ItemCentricReplayBuffer(capacity=5000)
    train_baseline_agent(env, dqn_agent, dqn_buffer, episodes=100)
    dqn_metrics = evaluate_agents(env, dqn_agent, num_episodes=30)
    
    # 4. Empirical Visualizations
    # Plot 1: Filter Bubble Analysis (Semantic Homogenization)
    plot_filter_bubble(morl_metrics['embedding_variances'], dqn_metrics['embedding_variances'])
    # Plot 2: Price of Responsibility (2D Pareto Front Projection)
    plot_price_of_responsibility(morl_metrics['rewards'], dqn_metrics['rewards'])
    # Plot 3: Full 3D Pareto Front (Engagement vs. Diversity vs. Fairness)
    plot_pareto_3d(morl_metrics['rewards'], dqn_metrics['rewards'])


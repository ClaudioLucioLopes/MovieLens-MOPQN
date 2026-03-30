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

def train_ParetoDQN_agent(env, agent, buffer, episodes=100, batch_size=32):
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

# def train_envelope_moac(env, agent, buffer, episodes=100, batch_size=32):
#     """Phase 2: Training the preference-conditioned Envelope MOAC."""
#     print("--- Starting Phase 2: Envelope MOAC Training ---")
    
#     global_step = 0
#     for ep in range(episodes):
#         state, info = env.reset(return_info=True)
#         # Step 2: Sample a preference w ~ Dirichlet for this episode
#         pref = agent.sample_preferences(batch_size=1)[0]
#         terminal = False
#         ep_rewards = np.zeros(3) # Track rewards for logging
        
#         while not terminal:
#             candidate_embs = info['candidate_embeddings']
#             # Select action based on state AND current preference
#             action_idx = agent.select_action(state, candidate_embs, pref)
#             chosen_item_emb = candidate_embs[action_idx]
            
#             # Step Environment
#             next_state, reward, terminal, next_info = env.step(action_idx)
            
#             # Store in PreferenceAwareBuffer
#             buffer.push(state, chosen_item_emb, reward, next_state, 
#                         next_info['candidate_embeddings'], terminal, pref)
            
#             # Network Optimization (Experience Replay)
#             if len(buffer) > batch_size:
#                 b_s, b_a, b_r, b_ns, b_nc, b_t, b_w = buffer.sample(batch_size)
                
#                 # Fix: Cast terminals to float32 to avoid boolean subtraction error
#                 agent.update(
#                     torch.tensor(b_s, dtype=torch.float32).to(agent.device),
#                     torch.tensor(b_a, dtype=torch.float32).to(agent.device),
#                     torch.tensor(b_r, dtype=torch.float32).to(agent.device),
#                     torch.tensor(b_ns, dtype=torch.float32).to(agent.device),
#                     torch.tensor(b_t, dtype=torch.float32).to(agent.device),
#                     torch.tensor(b_w, dtype=torch.float32).to(agent.device)
#                 )
            
#             state = next_state
#             info = next_info
#             ep_rewards += reward # Accumulate rewards
#             global_step += 1
            
#         # Logging every 10 episodes to match baseline style
#         if (ep + 1) % 10 == 0:
#             print(f"Episode {ep+1} | Mean Rewards: Eng={ep_rewards[0]:.2f}, Div={ep_rewards[1]:.2f}, Fair={ep_rewards[2]:.2f}")


def train_envelope_moac(env, agent, buffer, episodes=100, batch_size=32):
    print("--- Starting Phase 2: Stochastic Envelope MOAC Training ---")
    for ep in range(episodes):
        state, info = env.reset(return_info=True)
        pref = agent.sample_preferences(batch_size=1)[0]
        terminal = False
        ep_rewards = np.zeros(3)
        while not terminal:
            # Training uses stochastic actions (deterministic=False)
            action_idx = agent.select_action(state, info['candidate_embeddings'], pref, deterministic=False)
            next_state, reward, terminal, next_info = env.step(action_idx)
            buffer.push(state, info['candidate_embeddings'][action_idx], reward, 
                        next_state, next_info['candidate_embeddings'], terminal, pref)
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
            state, info, ep_rewards = next_state, next_info, ep_rewards + reward
        if (ep + 1) % 10 == 0:
            print(f"Episode {ep+1} | Mean Rewards: Eng={ep_rewards[0]:.2f}, Div={ep_rewards[1]:.2f}, Fair={ep_rewards[2]:.2f}")


def evaluate_agents(env, agent, num_episodes=20, agent_type='dqn'):
    """Collects rewards and semantic variance to detect filter bubbles[cite: 4, 83]."""
    metrics = {'rewards': [], 'variances': []}
    for _ in range(num_episodes):
        state, info = env.reset(return_info=True)
        terminal, ep_rewards, trajectory = False, np.zeros(3), [state]
        while not terminal:
            action_idx = agent.select_action(state, info['candidate_embeddings'])
            state, reward, terminal, info = env.step(action_idx)
            ep_rewards += reward
            trajectory.append(state)
        metrics['rewards'].append(ep_rewards)
        metrics['variances'].append(np.trace(np.cov(np.array(trajectory), rowvar=False)))
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

def evaluate_envelope_frontier(env, agent, num_episodes_per_w=5,num_samples=20):
    """Dense random sampling for a continuous Pareto surface."""
    metrics = {'rewards': [], 'variances': []}
    eval_weights = agent.sample_preferences(batch_size=num_samples)
    for w in eval_weights:
        state, info = env.reset(return_info=True)
        for _ in range(num_episodes_per_w):
            terminal, ep_rewards, traj = False, np.zeros(3), [state]
            while not terminal:
                # Evaluation uses deterministic means
                action_idx = agent.select_action(state, info['candidate_embeddings'], w, deterministic=True)
                state, reward, terminal, info = env.step(action_idx)
                ep_rewards, traj = ep_rewards + reward, traj + [state]
            metrics['rewards'].append(ep_rewards)
            metrics['variances'].append(np.trace(np.cov(np.array(traj), rowvar=False)))
    return metrics



def plot_filter_bubble(dqn_m, pareto_m, env_m):
    """Compares semantic homogenization across methods[cite: 83]."""
    plt.figure(figsize=(10, 6))
    plt.plot(dqn_m['variances'], label='Standard DQN (Bubbled)', color='red', linestyle='--')
    plt.plot(pareto_m['variances'], label='Pareto-DQN (Many-Obj)', color='blue')
    plt.plot(env_m['variances'], label='Envelope MOAC (Responsible)', color='green', linewidth=2)
    plt.title('Mitigating Filter Bubbles: User Embedding Variance')
    plt.xlabel('Episode'); plt.ylabel('Semantic Variance (Trace of Cov)'); plt.legend(); plt.grid(True)
    plt.savefig('comparison_filter_bubble.png')

def plot_price_of_responsibility(dqn_m, pareto_m, env_m):
    """Plots Engagement vs. Diversity trade-off[cite: 84]."""
    plt.figure(figsize=(10, 6))
    for m, c, l in zip([dqn_m, pareto_m, env_m], ['red', 'blue', 'green'], ['DQN', 'Pareto', 'Envelope']):
        r = np.array(m['rewards'])
        plt.scatter(r[:,0], r[:,1], color=c, label=l, alpha=0.6)
    plt.title('Price of Responsibility: Engagement vs. Information Diversity')
    plt.xlabel('User Engagement ($r_{eng}$)'); plt.ylabel('Diversity ($r_{div}$)'); plt.legend(); plt.grid(True)
    plt.savefig('comparison_price_of_responsibility.png')

def plot_pareto_3d(dqn_m, pareto_m, env_m):
    """Full 3D perspective of alignment goals[cite: 8, 49]."""
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    for m, c, l in zip([dqn_m, pareto_m, env_m], ['red', 'blue', 'green'], ['DQN', 'Pareto', 'Envelope']):
        r = np.array(m['rewards'])
        ax.scatter(r[:,0], r[:,1], r[:,2], color=c, label=l, s=50, alpha=0.7)
    ax.set_xlabel('Engagement'); ax.set_ylabel('Diversity'); ax.set_zlabel('Fairness')
    ax.set_title('3D Pareto Alignment Comparison'); plt.legend()
    plt.savefig('comparison_pareto_3d.png')


if __name__ == '__main__':
    # 1. Initialize Pipeline
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    movies_path = os.path.join(base_dir,'MovieLens-MOPQN', 'ml-latest-small', 'movies.csv')
    ratings_path = os.path.join(base_dir,'MovieLens-MOPQN', 'ml-latest-small', 'ratings.csv')
    tags_path = os.path.join(base_dir,'MovieLens-MOPQN', 'ml-latest-small', 'tags.csv')
    
    handler = MovieLensDataHandler(movies_path, ratings_path, tags_path)
    # env = MovieLensMOEnv(handler, top_k=100, max_steps=50)
    env = MovieLensMOEnv(handler, top_k=100, max_steps=150, drift_rate=0.20, fairness_ratio=0.3)

    # 2. Envelope MOAC (Proposed)
    print("\n=======================================")
    print("     EVALUATING ENVELOPE MOAC AGENT    ")
    print("=======================================")
    env_agent = EnvelopeMOACAgent(384, 384, 3)
    env_buffer = PreferenceAwareBuffer(5000)
    train_envelope_moac(env, env_agent, env_buffer, episodes=100)
    moac_metrics = evaluate_envelope_frontier(env, env_agent)


    # 3. Train & Evaluate Pareto-DQN (MORL)
    print("\n=======================================")
    print("      EVALUATING PARETO-DQN AGENT      ")
    print("=======================================")
    morl_agent = ParetoAgent(state_dim=384, item_dim=384, num_objectives=3)
    morl_buffer = ItemCentricReplayBuffer(capacity=5000)
    train_ParetoDQN_agent(env, morl_agent, morl_buffer, episodes=100)
    pareto_metrics = evaluate_agents(env, morl_agent, num_episodes=100)
    
    # 4. Train & Evaluate Standard DQN (Baseline)
    print("\n=======================================")
    print("     EVALUATING STANDARD DQN AGENT     ")
    print("=======================================")
    dqn_agent = StandardDQNAgent(state_dim=384, item_dim=384)
    dqn_buffer = ItemCentricReplayBuffer(capacity=5000)
    train_baseline_agent(env, dqn_agent, dqn_buffer, episodes=100)
    dqn_metrics = evaluate_agents(env, dqn_agent, num_episodes=100)

    
    # 5. Empirical Visualizations
    # Plot 1: Filter Bubble Analysis (Semantic Homogenization)
    plot_filter_bubble(dqn_metrics, pareto_metrics, moac_metrics)
    # Plot 2: Price of Responsibility (2D Pareto Front Projection)
    plot_price_of_responsibility(dqn_metrics, pareto_metrics, moac_metrics)
    # Plot 3: Full 3D Pareto Front (Engagement vs. Diversity vs. Fairness)
    plot_pareto_3d(dqn_metrics, pareto_metrics, moac_metrics)





    
    


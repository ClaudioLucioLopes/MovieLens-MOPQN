import numpy as np
import torch
from pymoo.indicators.hv import HV

from RewardApproximator import RewardApproximator
from NonDominatedApproximator import NonDominatedApproximator
from Estimator import Estimator

class ParetoAgent:
    """
    Pareto-DQN Agent using an Item-Centric Q-Network architecture.
    Evaluates continuous candidate embeddings to maximize hypervolume across multiple objectives.
    """
    def __init__(self, state_dim=384, item_dim=384, num_objectives=3, 
                 lr_reward=1e-4, lr_nd=1e-3, gamma=0.98, epsilon=1.0, epsilon_decay=0.999, 
                 epsilon_min=0.01, device='cpu'):
        
        self.nO = num_objectives
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.device = device
        
        # Using [0,0,0] as the reference point for hypervolume since rewards are normalized to [0, 1]
        self.ref_point = np.zeros(self.nO) 
        
        # 1. Initialize Item-Centric Networks
        reward_net = RewardApproximator(state_dim, item_dim, self.nO, device=device)
        nd_net = NonDominatedApproximator(state_dim, item_dim, self.nO, device=device)
        
        # 2. Wrap in Optimization Estimators
        self.rew_estim = Estimator(reward_net, lr=lr_reward, copy_every=100, device=device)
        self.nd_estim = Estimator(nd_net, lr=lr_nd, copy_every=100, device=device)

    def sample_objective_points(self, n_samples=10):
        """Samples the (d-1) objective plane to construct the Pareto front."""
        # For 3 objectives, we sample 2 dimensions in the range [0, 1]
        points = np.random.uniform(0, 1, (n_samples, self.nO - 1))
        return points

    def evaluate_candidates(self, state, candidate_embeddings, n_samples=10, use_target=False):
        """
        Batched forward pass evaluating all Top-K candidates.
        Computes Q_set = R(s, v_a) + gamma * ND(s, point, v_a)
        """
        num_candidates = candidate_embeddings.shape[0]
        
        # Expand state to match candidate batch size: (K, state_dim)
        states_batch = np.tile(state, (num_candidates, 1))
        
        # 1. Estimate Immediate Vectorial Rewards: Shape (K, 3)
        r_pred = self.rew_estim(states_batch, candidate_embeddings, use_target_network=use_target)
        
        # 2. Estimate Non-Dominated Future Returns
        points = self.sample_objective_points(n_samples) # Shape (n_samples, 2)
        
        # To evaluate all samples for all candidates, we tile the tensors
        # Resulting shapes for ND pass: (K * n_samples, ...)
        s_tiled = np.repeat(states_batch, n_samples, axis=0)
        c_tiled = np.repeat(candidate_embeddings, n_samples, axis=0)
        p_tiled = np.tile(points, (num_candidates, 1))
        
        # Predict the 3rd objective: Shape (K * n_samples, 1)
        nd_pred_last_dim = self.nd_estim(s_tiled, p_tiled, c_tiled, use_target_network=use_target)
        
        # Reconstruct the full ND points: Shape (K * n_samples, 3)
        nd_full_points = np.concatenate((p_tiled, nd_pred_last_dim), axis=1)
        
        # Reshape back to (K, n_samples, 3)
        nd_sets = nd_full_points.reshape(num_candidates, n_samples, self.nO)
        
        # 3. Calculate Final Q_set
        # R is (K, 3). We expand to (K, 1, 3) so it broadcasts across the n_samples
        r_expanded = np.expand_dims(r_pred, axis=1)
        q_sets = r_expanded + self.gamma * nd_sets
        
        return q_sets

    def compute_hypervolumes(self, q_sets):
        """Computes the hypervolume for each candidate's Q_set using pymoo."""
        num_candidates = q_sets.shape[0]
        hvs = np.zeros(num_candidates)
        
        # Initialize the pymoo HV indicator with the negated reference point
        ind = HV(ref_point=self.ref_point * -1.0)
        
        for i in range(num_candidates):
            # pymoo minimizes, so we negate the rewards to compute HV
            points = np.array(q_sets[i]) * -1.0
            
            try:
                # Calculate HV using the indicator
                hvs[i] = ind(points)
            except Exception:
                # Fallback if points are completely dominated by the ref point or invalid
                hvs[i] = 0.0
                
        return hvs

    def select_action(self, state, candidate_embeddings):
        """Epsilon-greedy action selection over the dynamic candidate pool."""
        if np.random.rand() < self.epsilon:
            # Random exploration within the dynamically bounded Top-K candidates
            return np.random.randint(len(candidate_embeddings))
        
        # 1. Evaluate all candidates to generate their Q_sets
        q_sets = self.evaluate_candidates(state, candidate_embeddings, n_samples=10)
        
        # 2. Compute Hypervolume for each candidate
        hvs = self.compute_hypervolumes(q_sets)
        
        # 3. Select the candidate that maximizes Hypervolume
        # Use random choice among ties to prevent index bias
        best_indices = np.argwhere(hvs == np.amax(hvs)).flatten()
        return np.random.choice(best_indices)

    def epsilon_step(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
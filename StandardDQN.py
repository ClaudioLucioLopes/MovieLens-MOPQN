import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy

class ItemCentricQNetwork(nn.Module):
    """A standard Q-Network that predicts a scalar Q-value for a state-item pair."""
    def __init__(self, state_dim=384, item_dim=384):
        super(ItemCentricQNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim + item_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.out = nn.Linear(256, 1) # Scalar output (Engagement)

    def forward(self, state, item_embedding):
        x = torch.cat((state.float(), item_embedding.float()), dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)

class StandardDQNAgent:
    """
    Baseline Agent: Optimizes strictly for Engagement (Utility).
    Serves as the empirical control to demonstrate the Filter Bubble.
    """
    def __init__(self, state_dim=384, item_dim=384, lr=1e-3, gamma=0.98, epsilon=1.0, epsilon_decay=0.999, epsilon_min=0.01, device='cpu'):
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.device = device
        
        self.q_net = ItemCentricQNetwork(state_dim, item_dim).to(device)
        self.target_net = copy.deepcopy(self.q_net).to(device)
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

    def select_action(self, state, candidate_embeddings):
        if np.random.rand() < self.epsilon:
            return np.random.randint(len(candidate_embeddings))
        
        with torch.no_grad():
            states_batch = torch.tensor(np.tile(state, (len(candidate_embeddings), 1)), dtype=torch.float32).to(self.device)
            cands_batch = torch.tensor(candidate_embeddings, dtype=torch.float32).to(self.device)
            
            # Predict scalar Q-values for all candidates
            q_values = self.q_net(states_batch, cands_batch).squeeze()
            
            # Select the item with the highest predicted Engagement
            return torch.argmax(q_values).item()

    def update(self, b_s, b_a_emb, b_r, b_ns, b_n_cands, b_t):
        self.optimizer.zero_grad()
        
        # We only care about Objective 0: Engagement
        rewards_eng = torch.tensor(b_r[:, 0], dtype=torch.float32).to(self.device)
        terminals = torch.tensor(b_t, dtype=torch.float32).to(self.device)
        
        # Current Q-values
        b_s_tensor = torch.tensor(b_s, dtype=torch.float32).to(self.device)
        b_a_tensor = torch.tensor(b_a_emb, dtype=torch.float32).to(self.device)
        current_q = self.q_net(b_s_tensor, b_a_tensor).squeeze()
        
        # Compute Target Q-values using the Target Network
        with torch.no_grad():
            batch_size = len(b_ns)
            max_next_q = torch.zeros(batch_size).to(self.device)
            
            for i in range(batch_size):
                if not b_t[i]:
                    ns_t = torch.tensor(np.tile(b_ns[i], (len(b_n_cands[i]), 1)), dtype=torch.float32).to(self.device)
                    nc_t = torch.tensor(b_n_cands[i], dtype=torch.float32).to(self.device)
                    next_qs = self.target_net(ns_t, nc_t).squeeze()
                    max_next_q[i] = torch.max(next_qs)
                    
            target_q = rewards_eng + self.gamma * max_next_q * (1 - terminals)
            
        loss = self.loss_fn(current_q, target_q)
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

    def update_target_network(self):
        self.target_net.load_state_dict(self.q_net.state_dict())
        
    def epsilon_step(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
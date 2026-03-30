import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy

class EnvelopeCritic(nn.Module):
    """
    Vectorized Q-Network for Envelope MOAC.
    Predicts [Engagement, Diversity, Fairness] simultaneously.

    A. Envelope Critic (Vectorized Q-Network)
    The Critic predicts a vector of expected returns Q(s,a,w)∈R3, where the components are [reng​,rdiv​,rfair​].
        Input: State s (User history centroid) + Action a (Item embedding) + Preference w.
        Output: A 3-dimensional vector of Q-values.
    """
    def __init__(self, state_dim=384, item_dim=384, pref_dim=3, hidden_dim=512):
        super(EnvelopeCritic, self).__init__()
        # Input: State + Item + Preference Vector
        self.fc1 = nn.Linear(state_dim + item_dim + pref_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        # Output: 3 objectives (vectorized Q-value)
        self.out = nn.Linear(hidden_dim // 2, 3) 

    def forward(self, state, item_emb, pref):
        x = torch.cat([state, item_emb, pref], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)



class EnvelopeActor(nn.Module):
    """Stochastic Policy Network for Entropy-Regularized MOAC."""
    def __init__(self, state_dim=384, pref_dim=3, item_dim=384, hidden_dim=512):
        super(EnvelopeActor, self).__init__()
        self.fc1 = nn.Linear(state_dim + pref_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        
        # Output Mean and Log Standard Deviation
        self.mu = nn.Linear(hidden_dim // 2, item_dim)
        self.log_std = nn.Linear(hidden_dim // 2, item_dim)

    def forward(self, state, pref):
        x = torch.cat([state, pref], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu = self.mu(x)
        log_std = torch.clamp(self.log_std(x), min=-20, max=2) # Numerical stability
        return mu, log_std

    def sample(self, state, pref):
        """Reparameterization trick: a = mu + std * epsilon."""
        mu, log_std = self.forward(state, pref)
        std = log_std.exp()
        normal = torch.randn_like(mu)
        x_t = mu + std * normal 
        action = torch.tanh(x_t) # Squash to [-1, 1]
        
        # Tanh log-probability correction
        log_prob = -((x_t - mu).pow(2) / (2 * std.pow(2))) - 0.5 * np.log(2 * np.pi) - log_std
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        return action, log_prob.sum(dim=-1, keepdim=True)

class EnvelopeMOACAgent:
    def __init__(self, state_dim=384, item_dim=384, pref_dim=3, lr=1e-4, alpha=0.2, gamma=0.98, device='cpu'):
        self.device = device
        self.gamma = gamma
        self.alpha = alpha # Entropy coefficient
        
        self.actor = EnvelopeActor(state_dim, pref_dim, item_dim).to(device)
        self.actor_target = copy.deepcopy(self.actor).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        
        self.critic = EnvelopeCritic(state_dim, item_dim, pref_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic).to(device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)

    def sample_preferences(self, batch_size=1):
        return np.random.dirichlet([1.0, 1.0, 1.0], size=batch_size).astype(np.float32)

    def select_action(self, state, candidate_embeddings, pref, deterministic=True):
        state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        pref_t = torch.tensor(pref, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            if deterministic:
                mu, _ = self.actor(state_t, pref_t)
                ideal_emb = torch.tanh(mu)
            else:
                ideal_emb, _ = self.actor.sample(state_t, pref_t)
            
        candidate_ts = torch.tensor(candidate_embeddings, dtype=torch.float32).to(self.device)
        similarities = F.cosine_similarity(ideal_emb, candidate_ts)
        return torch.argmax(similarities).item()

    def update(self, b_s, b_a, b_r, b_ns, b_t, b_w):
        # 1. Critic Update (Envelope Logic)
        self.critic_optimizer.zero_grad()
        curr_q = self.critic(b_s, b_a, b_w)
        
        with torch.no_grad():
            m_prefs = torch.tensor(self.sample_preferences(10), device=self.device)
            max_next_q = torch.zeros(b_ns.size(0), 3).to(self.device)
            
            for i in range(b_ns.size(0)):
                if not b_t[i]:
                    ns_rep = b_ns[i].repeat(10, 1)
                    na, _ = self.actor_target.sample(ns_rep, m_prefs)
                    nq_vec = self.critic_target(ns_rep, na, m_prefs)
                    utilities = torch.mv(nq_vec, b_w[i])
                    max_next_q[i] = nq_vec[torch.argmax(utilities)]
            
            target_q = b_r + self.gamma * max_next_q * (1 - b_t.float().unsqueeze(1))
        
        critic_loss = F.mse_loss(curr_q, target_q)
        critic_loss.backward()
        self.critic_optimizer.step()

        # 2. Entropy-Regularized Actor Update
        self.actor_optimizer.zero_grad()
        new_a, log_prob = self.actor.sample(b_s, b_w)
        q_vec = self.critic(b_s, new_a, b_w)
        scalar_q = torch.sum(q_vec * b_w, dim=1, keepdim=True)
        
        # Loss = Alpha * Log_Prob - Q
        actor_loss = (self.alpha * log_prob - scalar_q).mean()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        return critic_loss.item(), actor_loss.item()


# class EnvelopeActor(nn.Module):
#     """
#     Preference-conditioned Policy Network.
#     Outputs a target embedding in the semantic movie space.
#     The Actor proposes an action a (in the continuous embedding space) that maximizes the scalarized utility w⋅Q.
#         Input: State s + Preference w.
#         Output: A 384-dimensional latent action vector (which we then match to the closest candidate item).
#     """
#     def __init__(self, state_dim=384, pref_dim=3, item_dim=384, hidden_dim=512):
#         super(EnvelopeActor, self).__init__()
#         self.fc1 = nn.Linear(state_dim + pref_dim, hidden_dim)
#         self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
#         # Output is a vector in the same space as MovieLens embeddings
#         self.out = nn.Linear(hidden_dim // 2, item_dim)

#     def forward(self, state, pref):
#         x = torch.cat([state, pref], dim=-1)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         # Tanh ensures the output is bounded, similar to normalized embeddings
#         return torch.tanh(self.out(x))


# class EnvelopeMOACAgent:
#     def __init__(self, state_dim=384, item_dim=384, pref_dim=3, lr=1e-4, gamma=0.98, device='cpu'):
#         self.device = device
#         self.gamma = gamma
#         self.pref_dim = pref_dim
        
#         self.actor = EnvelopeActor(state_dim, pref_dim, item_dim).to(device)
#         self.actor_target = copy.deepcopy(self.actor).to(device)
#         self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        
#         self.critic = EnvelopeCritic(state_dim, item_dim, pref_dim).to(device)
#         self.critic_target = copy.deepcopy(self.critic).to(device)
#         self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)

#     def sample_preferences(self, batch_size=1):
#         """Step 2: Dirichlet sampling to cover the Pareto Front."""
#         return np.random.dirichlet([1.0, 1.0, 1.0], size=batch_size).astype(np.float32)

#     def select_action(self, state, candidate_embeddings, pref):
#         """Actor-driven selection matched to current candidate pool."""
#         state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
#         pref_t = torch.tensor(pref, dtype=torch.float32).unsqueeze(0).to(self.device)
        
#         with torch.no_grad():
#             ideal_emb = self.actor(state_t, pref_t) # Continuous action
            
#         # Map continuous ideal_emb to the closest discrete candidate
#         candidate_ts = torch.tensor(candidate_embeddings, dtype=torch.float32).to(self.device)
#         similarities = F.cosine_similarity(ideal_emb, candidate_ts)
#         return torch.argmax(similarities).item()

#     def update(self, b_s, b_a, b_r, b_ns, b_t, b_w):
#         # Step 3: Vectorized Critic Update (Envelope Logic)
#         self.critic_optimizer.zero_grad()
#         curr_q = self.critic(b_s, b_a, b_w)
        
#         with torch.no_grad():
#             # Sample random preferences to find the Envelope (max utility frontier)
#             m_prefs = torch.tensor(self.sample_preferences(10), device=self.device)
#             batch_size = b_ns.size(0)
#             max_next_q = torch.zeros(batch_size, 3).to(self.device)
            
#             for i in range(batch_size):
#                 if not b_t[i]:
#                     # Find which preference in m_prefs maximizes scalar utility for next state
#                     ns_rep = b_ns[i].repeat(10, 1)
#                     na = self.actor_target(ns_rep, m_prefs)
#                     nq_vec = self.critic_target(ns_rep, na, m_prefs)
                    
#                     utilities = torch.mv(nq_vec, b_w[i]) # Scalarize by current goal
#                     max_next_q[i] = nq_vec[torch.argmax(utilities)]
            
#             target_q = b_r + self.gamma * max_next_q * (1 - b_t.unsqueeze(1))
        
#         critic_loss = F.mse_loss(curr_q, target_q)
#         critic_loss.backward()
#         self.critic_optimizer.step()

#         # Step 4: Actor Update (Scalarized Policy Gradient)
#         self.actor_optimizer.zero_grad()
#         pred_a = self.actor(b_s, b_w)
#         actor_loss = -torch.mean(torch.sum(self.critic(b_s, pred_a, b_w) * b_w, dim=1))
#         actor_loss.backward()
#         self.actor_optimizer.step()
        
#         return critic_loss.item(), actor_loss.item()
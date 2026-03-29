import random
import numpy as np
from collections import deque

class ItemCentricReplayBuffer:
    """
    Experience Replay Buffer for Continuous Semantic Action Spaces.
    Stores the exact item embedding chosen, and the next candidate pool for target evaluation.
    """
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action_emb, reward, next_state, next_candidates_embs, terminal):
        self.buffer.append((
            state, 
            action_emb, 
            reward, 
            next_state, 
            next_candidates_embs, 
            terminal
        ))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, action_embs, rewards, next_states, next_candidates_embs, terminals = zip(*batch)
        
        return (
            np.array(states, dtype=np.float32),
            np.array(action_embs, dtype=np.float32),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(next_candidates_embs, dtype=np.float32), # Shape: (Batch, K, 384)
            np.array(terminals, dtype=np.bool_)
        )

    def __len__(self):
        return len(self.buffer)


class PreferenceAwareBuffer(ItemCentricReplayBuffer):
    """Extends existing buffer to store the active preference vector."""
    def push(self, state, action_emb, reward, next_state, next_candidates_embs, terminal, pref=None):
        p = pref if pref is not None else np.zeros(3)
        self.buffer.append((state, action_emb, reward, next_state, next_candidates_embs, terminal, p))

    def sample(self, batch_size):
        import random
        batch = random.sample(self.buffer, batch_size)
        states, action_embs, rewards, next_states, next_candidates_embs, terminals, prefs = zip(*batch)
        return (
            np.array(states, dtype=np.float32), np.array(action_embs, dtype=np.float32),
            np.array(rewards, dtype=np.float32), np.array(next_states, dtype=np.float32),
            np.array(next_candidates_embs, dtype=np.float32), np.array(terminals, dtype=np.bool_),
            np.array(prefs, dtype=np.float32)
        )
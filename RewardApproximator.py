import torch
import torch.nn as nn
import torch.nn.functional as F

class RewardApproximator(nn.Module):
    """
    Item-Centric Reward Estimator.
    Predicts the 3D reward vector [Engagement, Diversity, Fairness] based on the 
    semantic relationship between the user's continuous state and the item's embedding.
    """
    def __init__(self, state_dim=384, item_dim=384, nO=3, device='cpu'):
        super(RewardApproximator, self).__init__()

        self.state_dim = state_dim
        self.item_dim = item_dim
        self.nO = nO
        self.device = device

        # Input is the concatenation of user state and item embedding
        input_dim = self.state_dim + self.item_dim
        
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.out = nn.Linear(128, self.nO)

    def forward(self, state, item_embedding):
        """
        Args:
            state: Tensor of shape (batch_size, state_dim)
            item_embedding: Tensor of shape (batch_size, item_dim)
        Returns:
            out: Tensor of shape (batch_size, 3), the predicted vectorial reward.
        """
        state = state.float().to(self.device)
        item_embedding = item_embedding.float().to(self.device)
        
        # Concatenate continuous user and item representations
        x = torch.cat((state, item_embedding), dim=1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        out = self.out(x)
        return out
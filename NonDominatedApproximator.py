import torch
import torch.nn as nn
import torch.nn.functional as F

class NonDominatedApproximator(nn.Module):
    """
    Item-Centric Non-Dominated Set Estimator.
    Approximates the Pareto frontier surface for a given state-item pair.
    """
    def __init__(self, state_dim=384, item_dim=384, nO=3, device='cpu'):
        super(NonDominatedApproximator, self).__init__()
        
        self.state_dim = state_dim
        self.item_dim = item_dim
        self.nO = nO
        self.device = device
        
        # Input: state + (d-1) objectives + item_embedding
        input_dim = self.state_dim + (self.nO - 1) + self.item_dim
        
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        
        # Outputs a single scalar: the d-th objective value
        self.out = nn.Linear(128, 1)

    def forward(self, state, point, item_embedding):
        """
        Args:
            state: Tensor of shape (batch_size, state_dim)
            point: Tensor of shape (batch_size, nO-1), representing the sample points
                   on the (d-1) dimensional hyper-plane.
            item_embedding: Tensor of shape (batch_size, item_dim)
        Returns:
            out: Tensor of shape (batch_size, 1), the predicted final objective value.
        """
        state = state.float().to(self.device)
        point = point.float().to(self.device)
        item_embedding = item_embedding.float().to(self.device)
        
        # Concatenate user state, objective points, and the item embedding
        x = torch.cat((state, point, item_embedding), dim=1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        out = self.out(x)
        return out
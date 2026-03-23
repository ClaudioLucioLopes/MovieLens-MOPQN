import copy
import numpy as np
import torch
import torch.nn as nn

class Estimator(object):
    """
    Optimization wrapper for MORL PyTorch models.
    Handles gradient descent, loss calculation, and target network updates.
    """
    def __init__(self, model, lr=1e-3, tau=1.0, copy_every=100, clamp=None, device='cpu'):
        self.device = device
        self.model = model.to(self.device)
        self.target_model = copy.deepcopy(model).to(self.device)

        self.copy_every = copy_every
        self.tau = tau
        self.clamp = clamp
        
        self.opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.loss = nn.MSELoss(reduction='none')

    def should_copy(self, step):
        return self.copy_every and step > 0 and step % self.copy_every == 0

    def update_target(self, tau):
        for target_param, param in zip(self.target_model.parameters(), self.model.parameters()):
            target_param.data.copy_(tau * param.data + (1. - tau) * target_param.data)

    def predict(self, *net_args, use_target_network=False):
        """Passes continuous tensors to the network."""
        net = self.target_model if use_target_network else self.model
        
        # Ensure all inputs are PyTorch tensors on the correct device
        tensors = []
        for a in net_args:
            if not isinstance(a, torch.Tensor):
                a = torch.tensor(a, dtype=torch.float32)
            tensors.append(a.to(self.device))
            
        return net(*tensors)

    def __call__(self, *net_args, use_target_network=False):
        """Helper to return numpy arrays for action selection."""
        preds = self.predict(*net_args, use_target_network=use_target_network)
        return preds.detach().cpu().numpy()

    def update(self, targets, *net_args, step=None):
        """Performs a single gradient descent step."""
        self.opt.zero_grad()

        # Forward pass (e.g., f(state, item_embedding))
        preds = self.predict(*net_args, use_target_network=False)
        
        # Compute MSE Loss
        t_targets = torch.tensor(targets, dtype=torch.float32).to(self.device)
        l = self.loss(preds, t_targets)
        
        if self.clamp is not None:
            l = torch.clamp(l, min=-self.clamp, max=self.clamp)
        l = l.mean()

        # Backward pass
        l.backward()
        self.opt.step()

        # Target network Polyak/Hard update
        if step is not None and self.should_copy(step):
            self.update_target(self.tau)

        return l.item()
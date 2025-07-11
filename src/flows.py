# Classes and functions necessary for
# implementing a Real NVP
import torch
from torch import nn
from torch.nn.parameter import Parameter

class MLP(nn.Module):
    '''
    Multilayer perceptron module.
    '''
    def __init__(self, in_dim, hidden_dim = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), 
            nn.LeakyReLU(), 
            nn.Linear(hidden_dim, hidden_dim), 
            nn.LeakyReLU(), 
            nn.Linear(hidden_dim, in_dim)
        )

    def forward(self, x):
        return self.net(x)


class RealNVPLayer(nn.Module):
    '''
    Real NVP layer module.
    '''
    def __init__(self, dim, mask):
        super().__init__()
        
        # Uses MLP module as scale and
        # translate neural networks
        self.scale_net = MLP(dim)
        self.translate_net = MLP(dim)

        # Defines mask to implement
        # coupling layers
        self.mask = nn.Parameter(mask, requires_grad=False)

    def forward(self, x):
        # Forward transformation
        x_masked = x * self.mask
        s = self.scale_net(x_masked) * (1 - self.mask)
        t = self.translate_net(x_masked) * (1 - self.mask)
        z = x_masked + (1 - self.mask) * (x * torch.exp(s) + t)
        log_det_J = torch.sum(s, dim = 1)
        return z, log_det_J

    def inverse(self, y):
        # Inverse transformation
        y_masked = y * self.mask
        s = self.scale_net(y_masked) * (1 - self.mask)
        t = self.translate_net(y_masked) * (1 - self.mask)
        x = y_masked + (1 - self.mask) * ((y - t) * torch.exp(-s))
        log_det_J = torch.sum(s, dim = 1)
        return x, log_det_J


class RealNVP(nn.Module):
    '''
    Real NVP module.
    Consists of multiple Real NVP layers.
    '''
    def __init__(self, dim, n_layers, base_dist):
        super().__init__()
        self.dim = dim
        self.base_dist = base_dist
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            # Create masks
            mask_list = []
            for j in range(dim):
                mask_list.append((i + j)%2)
            mask = torch.tensor(
                mask_list, 
                dtype=torch.float32
            )
            # Add Real NVP layer
            self.layers.append(RealNVPLayer(self.dim, mask))

    def forward(self, x):
        # Forward transformation
        log_det_J = torch.zeros(x.size(0))
        for layer in self.layers:
            x, log_det = layer.forward(x)
            log_det_J += log_det
        return x, log_det_J

    def inverse(self, z):
        # Inverse transformation
        log_det_J = torch.zeros(z.size(0))
        for layer in reversed(self.layers):
            z, log_det = layer.inverse(z)
            log_det_J -= log_det
        return z, log_det_J

    def log_prob(self, x):
        # Computes the log pdf of the final samples
        z, log_det = self.inverse(x)
        return self.base_dist.log_prob(z) + log_det
        
    def sample(self, n_samples): 
        # Sample from the final distribution
        z = self.base_dist.sample((n_samples, 1))
        x,_ = self.forward(z.resize(n_samples, self.dim))
        return x


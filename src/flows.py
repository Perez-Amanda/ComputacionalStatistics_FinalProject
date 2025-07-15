import torch
from torch import nn
from torch.distributions import Normal

# Classes and functions necessary for Real NVP

class MLP(nn.Module):
    '''
    Multilayer perceptron module.
    '''
    def __init__(self, in_dim, out_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        return self.net(x)


class RealNVPLayer(nn.Module):
    '''
    Real NVP layer module.
    '''
    def __init__(self, dim, mask, context_dim):
        super().__init__()
        
        # Uses MLP module as scale and
        # translate neural networks
        self.scale_net = MLP(dim + context_dim, dim)
        self.translate_net = MLP(dim + context_dim, dim)

        # Defines mask to implement
        # coupling layers
        self.mask = mask

    def forward(self, x, context):
        # Forward transformation
        x_masked = x * self.mask
        # Adding context
        x_masked_with_context = torch.cat([x_masked, context], dim=-1)
        s = self.scale_net(x_masked_with_context) * (1 - self.mask)
        t = self.translate_net(x_masked_with_context) * (1 - self.mask)
        z = x_masked + (1 - self.mask) * (x * torch.exp(s) + t)
        log_det_J = torch.sum(s, dim=1)
        return z, log_det_J

    def inverse(self, y, context):
        # Inverse transformation
        y_masked = y * self.mask
        # Adding context
        y_masked_with_context = torch.cat([y_masked, context], dim=-1)
        s = self.scale_net(y_masked_with_context) * (1 - self.mask)
        t = self.translate_net(y_masked_with_context) * (1 - self.mask)
        x = y_masked + (1 - self.mask) * ((y - t) * torch.exp(-s))
        log_det_J = torch.sum(s, dim=1)
        return x, log_det_J


class RealNVP(nn.Module):
    '''
    Real NVP module.
    Consists of multiple Real NVP layers.
    '''
    def __init__(self, dim, n_layers, base_dist, context_dim):
        super().__init__()
        self.dim = dim
        self.base_dist = base_dist
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            # Create masks (alternating 0s and 1s)
            mask_list = [(i + j) % 2 for j in range(dim)]
            mask = torch.tensor(mask_list, dtype=torch.float32)

            # Add Real NVP layer
            self.layers.append(RealNVPLayer(self.dim, mask, context_dim))

    def forward(self, x, context):
        # Forward transformation
        log_det_J = torch.zeros(x.size(0), device=x.device)
        for layer in self.layers:
            x, log_det = layer.forward(x, context)
            log_det_J += log_det
        return x, log_det_J

    def inverse(self, z, context):
        # Inverse transformation
        log_det_J = torch.zeros(z.size(0), device=z.device)
        for layer in reversed(self.layers):
            z, log_det = layer.inverse(z, context)
            log_det_J -= log_det
        return z, log_det_J

    def log_prob(self, x, context):
        # Computes the log pdf of the final samples
        z, log_det = self.inverse(x, context)
        return self.base_dist.log_prob(z) + log_det
        
    def sample(self, n_samples, context):
        # Sample from the final distribution
        z = self.base_dist.sample((n_samples,))
        x, _ = self.forward(z.view(n_samples, self.dim), context)
        return x

# # Classes and functions necessary for
# # implementing a Real NVP
# import torch
# from torch import nn
# from torch.nn.parameter import Parameter

# class MLP(nn.Module):
#     '''
#     Multilayer perceptron module.
#     '''
#     def __init__(self, in_dim, out_dim, hidden_dim = 128):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(in_dim, hidden_dim), 
#             nn.LeakyReLU(), 
#             nn.Linear(hidden_dim, hidden_dim), 
#             nn.LeakyReLU(), 
#             nn.Linear(hidden_dim, out_dim)
#         )

#     def forward(self, x):
#         return self.net(x)


# class RealNVPLayer(nn.Module):
#     '''
#     Real NVP layer module.
#     '''
#     def __init__(self, dim, mask, context_dim):
#         super().__init__()
        
#         # Uses MLP module as scale and
#         # translate neural networks
#         self.scale_net = MLP(dim + context_dim, dim)
#         self.translate_net = MLP(dim + context_dim, dim)

#         # Defines mask to implement
#         # coupling layers
#         self.mask = nn.Parameter(mask, requires_grad=False)

#     def forward(self, x, context):
#         # Forward transformation
#         x_masked = x * self.mask
#         # Adding context
#         x_masked_with_context = torch.cat([x_masked, context], dim=-1)
#         s = self.scale_net(x_masked_with_context) * (1 - self.mask)
#         t = self.translate_net(x_masked_with_context) * (1 - self.mask)
#         z = x_masked_with_context + (1 - self.mask) * (x * torch.exp(s) + t)
#         log_det_J = torch.sum(s, dim = 1)
#         return z, log_det_J

#     def inverse(self, y, context):
#         # Inverse transformation
#         y_masked = y * self.mask
#         # Adding context
#         y_masked_with_context = torch.cat([y_masked, context], dim=-1)
#         s = self.scale_net(y_masked_with_context) * (1 - self.mask)
#         t = self.translate_net(y_masked_with_context) * (1 - self.mask)
#         x = y_masked_with_context + (1 - self.mask) * ((y - t) * torch.exp(-s))
#         log_det_J = torch.sum(s, dim = 1)
#         return x, log_det_J


# class RealNVP(nn.Module):
#     '''
#     Real NVP module.
#     Consists of multiple Real NVP layers.
#     '''
#     def __init__(self, dim, n_layers, base_dist, context_dim):
#         super().__init__()
#         self.dim = dim
#         self.base_dist = base_dist
#         self.layers = nn.ModuleList()
#         for i in range(n_layers):
#             # Create masks
#             mask_list = []
#             for j in range(dim):
#                 mask_list.append((i + j)%2)
#             mask = torch.tensor(
#                 mask_list, 
#                 dtype=torch.float32
#             )
#             # Add Real NVP layer
#             self.layers.append(RealNVPLayer(self.dim, mask, context_dim))

#     def forward(self, x, context):
#         # Forward transformation
#         log_det_J = torch.zeros(x.size(0))
#         for layer in self.layers:
#             x, log_det = layer.forward(x, context)
#             log_det_J += log_det
#         return x, log_det_J

#     def inverse(self, z, context):
#         # Inverse transformation
#         log_det_J = torch.zeros(z.size(0))
#         for layer in reversed(self.layers):
#             z, log_det = layer.inverse(z, context)
#             log_det_J -= log_det
#         return z, log_det_J

#     def log_prob(self, x, context):
#         # Computes the log pdf of the final samples
#         z, log_det = self.inverse(x, context)
#         return self.base_dist.log_prob(z) + log_det
        
#     def sample(self, n_samples, context): 
#         # Sample from the final distribution
#         z = self.base_dist.sample((n_samples, 1))
#         x,_ = self.forward(z.resize(n_samples, self.dim), context)
#         return x


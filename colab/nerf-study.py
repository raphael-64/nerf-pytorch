import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import nn, optim
from tqdm.notebook import tqdm
from IPython.display import clear_output
import time
import os
import wandb  # Optional: for experiment tracking

class NeRFConfig:
    pos_encoding_dims: int = 10
    dir_encoding_dims: int = 4
    network_width: int = 256
    early_mlp_layers: int = 5
    late_mlp_layers: int = 3
    num_coarse_samples: int = 64
    num_fine_samples: int = 128
    near_bound: float = 1.0
    far_bound: float = 4.0
    chunk_size: int = 32768
    batch_size: int = 64
    learning_rate: float = 5e-4
    lr_decay_steps: int = 250000
    lr_decay_rate: float = 0.1
    num_iterations: int = 250000
    display_every: int = 2000
    save_every: int = 4000
    checkpoint_dir: str = "checkpoints"
    device: str = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    data_path: str = "66bdbc812bd0a196e194052f3f12cb2e.npz"
    test_view_idx: int = 150

config = NeRFConfig()
print(f"Using device {config.device}")

class PositionalEncoding:
    def __init__(self, num_frequencies):
        self.num_frequencies = num_frequencies

    def __call__(self, x):
        encoded = [x]
        for i in range(self.num_frequencies):
            encoded.append(torch.sin(2 ** i * torch.pi * x))
            encoded.append(torch.cos(2 ** i * torch.pi * x))
        return torch.cat(encoded, dim=-1)

class NeRFMLP(nn.Module):
    def __init__(self):
        super.__init__()
        self.pos_encoder = PositionalEncoding(config.pos_encoding_dims)
        self.dir_encoder = PositionalEncoding(config.dir_encoding_dims)

        pos_encoder_feats = 3 + 3 * 2 * config.pos_encoding_dims
        dir_encoder_feats = 3 + 3 * 2 * config.dir_encoding_dims

        early_mlp = []
        
        early_mlp.append(nn.Linear(pos_encoder_feats, config.network_width))

        for _ in range(config.early_mlp_layers-1):
            early_mlp.append(nn.Linear(config.network_width, config.network_width))
            early_mlp.append(nn.ReLU)

        self.early_mlp = nn.Sequential(*early_mlp)
        
        late_mlp = []

        for _ in range(config.late_mlp_layers):
            late_mlp.append(nn.Linear(config.network_width))
            late_mlp.append(nn.ReLU)
        late_mlp.append(nn.Linear(config.network_width, ))

        self.late_mlp = nn.Sequential(*late_mlp)

        self.sigma_layer = nn.Linear(config.network_width, config.network_width+1)
        self.pre_final_layer = nn.Sequential(
            nn.Linear(dir_encoder_feats + config.network_width, config.network_width // 2),
            nn.ReLU()
        )
        self.final_layer = nn.Sequential(
            nn.Linear(config.network_width // 2, 3),
            nn.Sigmoid()
        )
    
    def forward(self, positions, directions):
        directions = directions / directions.norm(p=2, dim=-1)

        pos_encoded = self.pos_encoder(positions)
        dir_encoded = self.dir_encoder(directions)

        features = self.early_mlp(pos_encoded)
        features = self.late_mlp(torch.cat([features, dir_encoded], dim=-1))

        outputs = self.sigma_layer(features)
        sigma = torch.relu(outputs[...,0])

        rgb_features = self.pre_final_layer(torch.cat([dir_encoded, outputs[...,1:]], dim=-1))
        rgb = self.final_layer(rgb_features)

        return {"rgb": rgb, "sigma": sigma}
    
def get_coarse_query_points(num_samples, ray_directions, ray_origins, near_bound, far_bound):
    bin_size = (far_bound-near_bound) / num_samples
    bin_edges = near_bound + torch.arange(num_samples, device=ray_directions.device) * bin_size
    # bin_edges = torch.linspace(near_bound, far_bound, num_samples)

    random_offsets = torch.rand(*ray_directions.shape[:-1], num_samples, device=ray_directions.device)
    t_vals = bin_edges + random_offsets * bin_size

    points = ray_origins[] + ray_directions[] * t_vals[]

    return points, t_vals

def get_fine_query_points(num_fine_samples, ray_origins, ray_directions, far_bound, weights, coarse_t_vals):
    weights = weights + 1e-4
    pdf = weights / weights.sum(dim=-1, keepdim=True)
    cdf = torch.cumsum(pdf, dim=-1)
    cdf = torch.cat(torch.zeros_like(cdf[...,:1]), cdf[...,:-1], dim=-1)

    uniform_samples = torch.rand(*weights.shape[:-1], num_fine_samples, device=weights.device)

    # this just takes all the sorted uniform samples and searches for the cdf?? 
    # alternaitvely it takes the cdf and finds the uniform_samples within it, returning the indices.
    indices - torch.searchsorted(cdf, uniform_samples, right=True)
    # 
    below = torch.max(torch.zeros_like(indices), indices-1)














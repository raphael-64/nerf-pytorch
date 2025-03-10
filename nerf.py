import matplotlib.pyplot as plt
import numpy as np
import torch

from torch import nn, optim

def get_coarse_query_points(ray_origins, ray_directions, num_samples, bin_edges, bin_gap):
    random_offsets = torch.rand(*ray_directions.shape[:2], num_samples).to(ray_directions)
    sampled_depths = bin_edges + random_offsets * bin_gap

    sampled_points = ray_origins[..., None, :] + sampled_depths[..., :, None] * ray_directions[...,None,:]

    return sampled_points, sampled_depths



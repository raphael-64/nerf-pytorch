import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch import optim

def positional_encoding(x, L):

    out = [x]
    for i in range(L):
        freq = 2 ** L
        out.append(torch.sin(freq * np.pi * x))
        out.append(torch.cos(freq * np.pi * x))
    return torch.cat(out, dim=-1)


class NeRF(nn.Module):
    def __init(self, D=8, W=256, L_pos=10, L_dir=4, skips=[4]):
        super(NeRF, self).__init__()

        # frequency parameters for positional encoding function
        self.L_pos = L_pos
        self.L_dir = L_dir

        self.skips = skips

        # input dimensions
        self.in_ch_pos = 3 + 3 * 2 * L_pos # 3 for each 3d coord but also each has 2 for each x,y,z for sin/cos on it
        self.in_ch_dir = 3 + 3 * 2 * L_dir 

        self.pts_linears = nn.ModuleList(
            [nn.Linear(self.in_ch, W)] 
        )

        for i in range(1,D):
            if i in skips:
                self.pts_linears.append(nn.Linear(W+self.in_ch_pos, W)) # if index is in skip list, we prep to add skip connection
            else:
                self.pts_linears.append(nn.Linear(W, W))
        
        self.sigma_linear = nn.Linear(W,1) # predict density (single value per point)
        self.feature_linear = nn.Linear(W,W) # help predicts RGB color when combined with encoded view direction

        self.views_linears = nn.ModuleList([nn.Linear(W + self.in_ch_dir, W // 2)]) # concatenates feature vector and encoded viewing direction, reducing dim and applying ReLU
        self.rgb_linear = nn.Linear(W // 2, 3)# outputs 3 rgb values for each query point, sigmoid activation in forward ensures these are in 0,1 range

    def forward(self, x, d):
        x_encoded = positional_encoding(x, self.L_pos)

        d_normalized = d / d.norm(dim=-1, keepdim=True)
        d_encoded = positional_encoding(d_normalized, self.L_dir)

        h = x_encoded
        for i, layer in enumerate(self.pts_linears):
            h = layer(h)
            h = F.relu(h)

            if i in self.skips: # tf do i put here tho
                h = torch.cat([x_encoded, h], dim=-1)
        
        sigma = F.relu(self.sigma_linear(h))

        for layer in self.view_linears:
            h_color = layer(h_color)
            h_color = F.relu(h_color)

        rgb = torch.sigmod(self.rgb_linear(h_color))
        output = torch.cat([rgb, sigma], dim=-1)
        return output
                 


def get_coarse_query_points(ray_o, ray_d, N_samples, t_n, t_f):
    N_rays = ray_o.shape[0]

    # sampling depths
    t_vals = torch.linspace(0., 1., steps=N_samples, device=ray_o.device) # evenly spaced tensor, but between 0 and 1
    z_vals = t_n * (1. - t_vals) + t_f * t_vals # now between t_n and t_f, shape is (N_samples)
    z_vals = z_vals.expand(N_rays, N_samples) 

    #stratified sampling
    mids = 0.5 * (z_vals[:, 1:] + z_vals[:, :-1])
    upper = torch.cat([mids, z_vals[:, -1:]], dim=-1)
    lower = torch.cat([z_vals[:, :1], mids], dim=-1)
    t_rand = torch.rand(z_vals.shape, device = ray_o.device) # random numbers so u can sample within the bins
    z_vals = lower + (upper-lower) * t_rand # sample within each bin using t_Rand

    r_ts = ray_o[:, None, :] + ray_d[:, None, :] * z_vals[..., None] # calc using r = o + td formula

    return r_ts, z_vals

def get_fine_query_points(weights, t_vals_coarse, N_samples, ray_o, ray_d):

    N_rays = ray_o.shape[0]
    pdf = weights / torch.sum(weights)

    cdf = torch.cumsum(pdf, dim=-1)

    u = torch.rand(N_samples) 
    x_samples = torch.interp(u, cdf_y, cdf_x)  # Step 4: Approximate inverse CDF
    return x_samples
    

    return

def checkCoarseQueryPointsGetter():


    #testing get_coarse_query_points
    N_rays = 5
    N_samples = 10
    t_n = 0.5
    t_f = 2.0

    # Create test input tensors
    ray_o = torch.zeros((N_rays, 3))  # Origin at (0,0,0) for each ray
    ray_d = torch.tensor([[1.0, 0.0, 0.0]] * N_rays)  # Rays pointing in x-direction

    # Run function
    r_ts, z_vals = get_coarse_query_points(ray_o, ray_d, N_samples, t_n, t_f)

    # Check output shapes
    assert r_ts.shape == (N_rays, N_samples, 3), f"Unexpected shape for r_ts: {r_ts.shape}"
    print(r_ts.shape)
    assert z_vals.shape == (N_rays, N_samples), f"Unexpected shape for z_vals: {z_vals.shape}"
    print(z_vals.shape)

    # Check depth values are within range
    assert torch.all(z_vals >= t_n) and torch.all(z_vals <= t_f), "z_vals out of range"

    # Check if points are correctly computed along the ray direction
    expected_r_ts = ray_o[:, None, :] + ray_d[:, None, :] * z_vals[..., None]

    assert torch.allclose(r_ts, expected_r_ts, atol=1e-6), "r_ts computation is incorrect"

    print("All tests passed!")


if __name__ == "__main__":
    checkCoarseQueryPointsGetter()

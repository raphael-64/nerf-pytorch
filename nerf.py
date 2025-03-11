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

        # frequency parameters for positional encoding
        self.L_pos = L_pos
        self.L_dir = L_dir

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

        self.views_linears = nn.ModuleList([nn.Linear(W+ self.in_ch_dir, W // 2)]) # concatenates feature vector and encoded viewing direction, reducing dim and applying ReLU
        self.rgb_linear = nn.Linear(W // 2, 3)# outputs 3 rgb values for each query point, sigmoid activation in forward ensures these are in 0,1 range

        def forward(self, x, d):
            x_encoded = positional_encoding(x, self.L_pos)

            d_normalized = d / d.norm(dim=-1, keepdim=True)
            d_encoded = positional_encoding(d_normalized, self.L_dir)

            h = x_encoded
            for i, layer in enumerate(self.pts_linears):
                h = layer(h)
                h = F.relu(h)

                if i in skips:
                    h = torch.cat([x_encoded, h], dim=-1)
            
            sigma = F.relu(self.sigma_linear(h))
            for layer in self.view_linears:
                h_color = layer(h_color)
                h_color = F.relu(h_color)

            rgb = torch.sigmoid(self.rgb_linear(h_color))
            output = torch.cat([rgb, sigma], dim=-1)
            return output
        
            




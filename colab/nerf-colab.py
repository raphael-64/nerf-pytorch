# Cell 1: Setup and Dependencies
!pip install torch numpy matplotlib tqdm wandb

# Cell 2: Imports and Config
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import nn, optim
from tqdm.notebook import tqdm
from IPython.display import clear_output
import time
import os
import wandb  # Optional: for experiment tracking

# Configuration
CONFIG = {
    # Model Architecture
    "pos_encoding_dims": 10,    # L_pos: Number of frequencies for position encoding
    "dir_encoding_dims": 4,     # L_dir: Number of frequencies for direction encoding
    "network_width": 256,       # Width of the neural network layers
    "early_mlp_layers": 5,      # Number of layers in early MLP
    "late_mlp_layers": 3,       # Number of layers in late MLP
    
    # Volume Rendering
    "num_coarse_samples": 64,   # N_c: Number of coarse samples per ray
    "num_fine_samples": 128,    # N_f: Number of fine samples per ray
    "near_bound": 1.0,         # t_n: Near bound for sampling
    "far_bound": 4.0,          # t_f: Far bound for sampling
    "chunk_size": 32768,       # Number of points to process at once
    "batch_size": 64,          # Size of image patch (batch_size x batch_size rays)
    
    # Training
    "learning_rate": 5e-4,
    "lr_decay_steps": 250000,   # Steps for learning rate decay
    "lr_decay_rate": 0.1,      # Learning rate decay multiplier
    "num_iterations": 200000,
    "display_every": 2000,
    "save_every": 10000,
    "checkpoint_dir": "checkpoints",
    
    # Device
    "device": "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu",
    
    # Dataset
    "data_path": "data.npz",
    "test_view_idx": 150,
}

print(f"Using device: {CONFIG['device']}")

# Cell 3: NeRF Implementation
class PositionalEncoding:
    """Positional encoding as described in the NeRF paper Section 5.1"""
    def __init__(self, num_frequencies):
        self.num_frequencies = num_frequencies
    
    def __call__(self, x):
        """
        Apply positional encoding to input.
        
        Args:
            x: [..., 3] tensor to encode
            
        Returns:
            encoded: [..., 3 + 3*2*num_frequencies] tensor
        """
        encodings = [x]
        for i in range(self.num_frequencies):
            for fn in [torch.sin, torch.cos]:
                encodings.append(fn(2.0 ** i * torch.pi * x))
        return torch.cat(encodings, dim=-1)

class NeRFMLP(nn.Module):
    """Full NeRF MLP architecture from the paper"""
    def __init__(self, config=CONFIG):
        super().__init__()
        self.pos_encoder = PositionalEncoding(config["pos_encoding_dims"])
        self.dir_encoder = PositionalEncoding(config["dir_encoding_dims"])
        
        # Calculate input feature dimensions
        pos_enc_feats = 3 + 3 * 2 * config["pos_encoding_dims"]
        dir_enc_feats = 3 + 3 * 2 * config["dir_encoding_dims"]
        
        # Build early MLP (processes positions only)
        early_mlp = []
        in_features = pos_enc_feats
        for _ in range(config["early_mlp_layers"]):
            early_mlp.extend([
                nn.Linear(in_features, config["network_width"]),
                nn.ReLU()
            ])
            in_features = config["network_width"]
        self.early_mlp = nn.Sequential(*early_mlp)
        
        # Build late MLP (processes positions + early features)
        late_mlp = []
        in_features = pos_enc_feats + config["network_width"]
        for _ in range(config["late_mlp_layers"]):
            late_mlp.extend([
                nn.Linear(in_features, config["network_width"]),
                nn.ReLU()
            ])
            in_features = config["network_width"]
        self.late_mlp = nn.Sequential(*late_mlp)
        
        # Final layers
        self.sigma_layer = nn.Linear(config["network_width"], config["network_width"] + 1)
        self.pre_final_layer = nn.Sequential(
            nn.Linear(dir_enc_feats + config["network_width"], config["network_width"] // 2),
            nn.ReLU()
        )
        self.final_layer = nn.Sequential(
            nn.Linear(config["network_width"] // 2, 3),
            nn.Sigmoid()
        )
        
    def forward(self, positions, directions):
        """
        Forward pass of the network.
        
        Args:
            positions: [..., 3] Points in 3D space
            directions: [..., 3] Viewing directions
            
        Returns:
            dict containing:
                c_is: [..., 3] RGB colors
                sigma_is: [...] Densities
        """
        # Normalize directions
        directions = directions / directions.norm(p=2, dim=-1, keepdim=True)
        
        # Apply positional encoding
        pos_encoded = self.pos_encoder(positions)
        dir_encoded = self.dir_encoder(directions)
        
        # Early MLP - position only
        features = self.early_mlp(pos_encoded)
        
        # Late MLP - position + features
        features = self.late_mlp(torch.cat([pos_encoded, features], dim=-1))
        
        # Predict density and color features
        outputs = self.sigma_layer(features)
        sigma = torch.relu(outputs[..., 0])
        
        # Final layers - incorporate viewing direction
        rgb_features = self.pre_final_layer(torch.cat([dir_encoded, outputs[..., 1:]], dim=-1))
        rgb = self.final_layer(rgb_features)
        
        return {"c_is": rgb, "sigma_is": sigma}

def get_coarse_query_points(ray_directions, num_samples, near_bound, far_bound, ray_origins):
    """
    Generate sample points along rays for coarse sampling.
    
    Args:
        ray_directions: [..., 3] tensor of ray directions
        num_samples: number of samples per ray
        near_bound: nearest sampling distance
        far_bound: farthest sampling distance
        ray_origins: [..., 3] tensor of ray origin points
        
    Returns:
        points: [..., num_samples, 3] tensor of 3D sample points
        t_vals: [..., num_samples] tensor of distance values
    """
    # Calculate bin size for stratified sampling
    bin_size = (far_bound - near_bound) / num_samples
    bin_edges = near_bound + torch.arange(num_samples, device=ray_directions.device) * bin_size
    
    # Random sampling within each bin
    random_offsets = torch.rand(*ray_directions.shape[:-1], num_samples, device=ray_directions.device)
    t_vals = bin_edges + random_offsets * bin_size
    
    # Calculate 3D points along rays
    points = ray_origins[..., None, :] + t_vals[..., :, None] * ray_directions[..., None, :]
    
    return points, t_vals

def get_fine_query_points(weights, num_fine_samples, coarse_t_vals, far_bound, ray_origins, ray_directions):
    """
    Generate sample points along rays for fine sampling using importance sampling.
    
    Args:
        weights: [..., num_coarse_samples] tensor of weights from coarse sampling
        num_fine_samples: number of fine samples per ray
        coarse_t_vals: [..., num_coarse_samples] tensor of coarse sample distances
        far_bound: farthest sampling distance
        ray_origins: [..., 3] tensor of ray origin points
        ray_directions: [..., 3] tensor of ray directions
        
    Returns:
        points: [..., num_coarse_samples + num_fine_samples, 3] tensor of 3D sample points
        t_vals: [..., num_coarse_samples + num_fine_samples] tensor of distance values
    """
    # Create PDFs and CDFs from weights
    weights = weights + 1e-5  # Prevent division by zero
    pdfs = weights / weights.sum(dim=-1, keepdim=True)
    cdfs = torch.cumsum(pdfs, dim=-1)
    cdfs = torch.cat([torch.zeros_like(cdfs[..., :1]), cdfs[..., :-1]], dim=-1)
    
    # Sample uniformly
    uniform_samples = torch.rand(*weights.shape[:-1], num_fine_samples, device=weights.device)
    
    # Inverse transform sampling
    indices = torch.searchsorted(cdfs, uniform_samples, right=True)
    below = torch.max(torch.zeros_like(indices), indices-1)
    above = torch.min((coarse_t_vals.shape[-1]-1) * torch.ones_like(indices), indices)
    t_below = torch.gather(coarse_t_vals, -1, below)
    t_above = torch.gather(coarse_t_vals, -1, above)
    t_above[indices == coarse_t_vals.shape[-1]] = far_bound
    
    # Sample between bins
    t_vals_fine = t_below + (t_above - t_below) * torch.rand_like(t_below)
    
    # Combine and sort all samples
    t_vals_all, _ = torch.sort(torch.cat([coarse_t_vals, t_vals_fine], dim=-1), dim=-1)
    points = ray_origins[..., None, :] + t_vals_all[..., :, None] * ray_directions[..., None, :]
    
    return points, t_vals_all

def render_radiance_volume(points, ray_directions, chunk_size, nerf_model, t_vals):
    """
    Render the radiance field using volume rendering equation.
    
    Args:
        points: [..., num_samples, 3] tensor of sample points
        ray_directions: [..., 3] tensor of ray directions
        chunk_size: number of points to process at once
        nerf_model: neural network model
        t_vals: [..., num_samples] tensor of distance values
        
    Returns:
        rgb: [..., 3] tensor of rendered RGB values
        weights: [..., num_samples] tensor of sample weights
    """
    # Flatten points for batch processing
    flat_points = points.reshape((-1, 3))
    repeated_dirs = ray_directions.unsqueeze(-2).repeat(*([1] * (ray_directions.dim() - 1)), points.shape[-2], 1)
    flat_dirs = repeated_dirs.reshape((-1, 3))
    
    # Process in chunks
    colors, densities = [], []
    for i in range(0, flat_points.shape[0], chunk_size):
        chunk_points = flat_points[i:i + chunk_size]
        chunk_dirs = flat_dirs[i:i + chunk_size]
        
        with torch.set_grad_enabled(nerf_model.training):
            predictions = nerf_model(chunk_points, chunk_dirs)
            colors.append(predictions["c_is"])
            densities.append(predictions["sigma_is"])
    
    colors = torch.cat(colors).reshape(*points.shape)
    densities = torch.cat(densities).reshape(*points.shape[:-1])
    
    # Calculate transmittance
    delta_dists = t_vals[..., 1:] - t_vals[..., :-1]
    delta_dists = torch.cat([
        delta_dists,
        torch.full_like(delta_dists[..., :1], 1e10)
    ], dim=-1)
    delta_dists = delta_dists * ray_directions.norm(dim=-1).unsqueeze(-1)
    
    # Alpha compositing
    alphas = 1.0 - torch.exp(-densities * delta_dists)
    transmittance = torch.cumprod(1.0 - alphas + 1e-10, dim=-1)
    transmittance = torch.roll(transmittance, 1, -1)
    transmittance[..., 0] = 1.0
    
    weights = transmittance * alphas
    rgb = (weights[..., None] * colors).sum(dim=-2)
    
    return rgb, weights

def run_one_iter_of_nerf(ray_directions, ray_origins, config, coarse_model, fine_model):
    """Run one iteration of the hierarchical sampling NeRF model."""
    # Coarse network
    coarse_points, coarse_t_vals = get_coarse_query_points(
        ray_directions,
        config["num_coarse_samples"],
        config["near_bound"],
        config["far_bound"],
        ray_origins
    )
    coarse_rgb, coarse_weights = render_radiance_volume(
        coarse_points,
        ray_directions,
        config["chunk_size"],
        coarse_model,
        coarse_t_vals
    )
    
    # Fine network
    fine_points, fine_t_vals = get_fine_query_points(
        coarse_weights,
        config["num_fine_samples"],
        coarse_t_vals,
        config["far_bound"],
        ray_origins,
        ray_directions
    )
    fine_rgb, _ = render_radiance_volume(
        fine_points,
        ray_directions,
        config["chunk_size"],
        fine_model,
        fine_t_vals
    )
    
    return coarse_rgb, fine_rgb

def save_checkpoint(coarse_model, fine_model, optimizer, iteration, psnrs, config, filename="latest.pt"):
    """Save training checkpoint."""
    os.makedirs(config["checkpoint_dir"], exist_ok=True)
    path = os.path.join(config["checkpoint_dir"], filename)
    torch.save({
        'iteration': iteration,
        'coarse_model_state_dict': coarse_model.state_dict(),
        'fine_model_state_dict': fine_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'psnrs': psnrs,
        'config': config
    }, path)

def load_checkpoint(coarse_model, fine_model, optimizer, config, filename="latest.pt"):
    """Load training checkpoint."""
    path = os.path.join(config["checkpoint_dir"], filename)
    if os.path.exists(path):
        checkpoint = torch.load(path)
        coarse_model.load_state_dict(checkpoint['coarse_model_state_dict'])
        fine_model.load_state_dict(checkpoint['fine_model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_iter = checkpoint['iteration']
        psnrs = checkpoint.get('psnrs', [])
        return start_iter, psnrs
    return 0, []

def train_nerf():
    """Main training function."""
    # Set random seeds
    torch.manual_seed(9458)
    np.random.seed(9458)
    
    device = CONFIG["device"]
    
    # Initialize models
    coarse_model = NeRFMLP(CONFIG).to(device)
    fine_model = NeRFMLP(CONFIG).to(device)
    
    if device == "cuda":
        # Enable TF32 for better performance
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None
    
    # Initialize optimizer
    optimizer = optim.Adam(
        list(coarse_model.parameters()) + list(fine_model.parameters()),
        lr=CONFIG["learning_rate"]
    )
    criterion = nn.MSELoss()
    
    # Load checkpoint if exists
    start_iter, psnrs = load_checkpoint(coarse_model, fine_model, optimizer, CONFIG)
    if start_iter > 0:
        print(f"Resuming from iteration {start_iter}")
    
    # Load dataset
    data = np.load(CONFIG["data_path"])
    images = torch.from_numpy(data["images"]).float() / 255.0
    poses = torch.from_numpy(data["poses"]).float()
    focal = float(data["focal"])
    
    # Setup camera parameters
    img_size = images.shape[1]
    xs = torch.arange(img_size) - (img_size / 2 - 0.5)
    ys = torch.arange(img_size) - (img_size / 2 - 0.5)
    xs, ys = torch.meshgrid(xs, -ys, indexing="xy")
    
    # Create camera rays
    pixel_coords = torch.stack([xs, ys, torch.full_like(xs, -focal)], dim=-1)
    camera_coords = pixel_coords / focal
    init_rays = camera_coords.to(device)
    init_origin = torch.tensor([0, 0, float(data["camera_distance"])], device=device)
    
    # Setup test view
    test_idx = CONFIG["test_view_idx"]
    plt.figure(figsize=(6, 6))
    plt.imshow(images[test_idx])
    plt.title("Test View Ground Truth")
    plt.show()
    
    test_img = images[test_idx].to(device)
    test_pose = poses[test_idx, :3, :3].to(device)
    test_rays = torch.einsum("ij,hwj->hwi", test_pose, init_rays)
    test_origins = (test_pose @ init_origin).expand(test_rays.shape)
    
    # Training loop
    train_indices = np.arange(len(images)) != test_idx
    train_images = images[train_indices].to(device)
    train_poses = poses[train_indices].to(device)
    
    pbar = tqdm(range(start_iter, CONFIG["num_iterations"]))
    training_start_time = time.time()
    
    for i in pbar:
        # Random training view
        target_idx = np.random.randint(train_images.shape[0])
        target_pose = train_poses[target_idx, :3, :3]
        
        # Random patch of rays
        batch_size = CONFIG["batch_size"]
        patch_x = np.random.randint(0, img_size - batch_size + 1)
        patch_y = np.random.randint(0, img_size - batch_size + 1)
        patch_rays = init_rays[patch_y:patch_y+batch_size, patch_x:patch_x+batch_size]
        patch_target = train_images[target_idx, patch_y:patch_y+batch_size, patch_x:patch_x+batch_size]
        
        # Generate rays for training view
        rays = torch.einsum("ij,hwj->hwi", target_pose, patch_rays)
        ray_origins = (target_pose @ init_origin).expand(rays.shape)
        
        # Render and compute loss
        if scaler is not None:
            with torch.cuda.amp.autocast():
                coarse_rgb, fine_rgb = run_one_iter_of_nerf(
                    rays, ray_origins, CONFIG, coarse_model, fine_model
                )
                loss = criterion(coarse_rgb, patch_target) + criterion(fine_rgb, patch_target)
            
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            coarse_rgb, fine_rgb = run_one_iter_of_nerf(
                rays, ray_origins, CONFIG, coarse_model, fine_model
            )
            loss = criterion(coarse_rgb, patch_target) + criterion(fine_rgb, patch_target)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Learning rate decay
        new_lr = CONFIG["learning_rate"] * (CONFIG["lr_decay_rate"] ** (i / CONFIG["lr_decay_steps"]))
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'lr': f'{new_lr:.6f}',
            'time/iter': f'{(time.time() - training_start_time)/(i+1-start_iter):.2f}s'
        })
        
        # Evaluation and visualization
        if i % CONFIG["display_every"] == 0:
            coarse_model.eval()
            fine_model.eval()
            with torch.no_grad():
                _, test_render = run_one_iter_of_nerf(
                    test_rays, test_origins, CONFIG, coarse_model, fine_model
                )
            
            test_loss = criterion(test_render, test_img)
            psnr = -10.0 * torch.log10(test_loss)
            psnrs.append(psnr.item())
            
            # Visualization
            clear_output(wait=True)
            plt.figure(figsize=(20, 5))
            
            plt.subplot(141)
            plt.imshow(test_img.cpu())
            plt.title("Ground Truth")
            
            plt.subplot(142)
            plt.imshow(test_render.cpu())
            plt.title(f"Rendered (Iter {i})")
            
            plt.subplot(143)
            plt.plot(psnrs)
            plt.title(f"PSNR (Current: {psnr.item():.2f})")
            plt.xlabel("Evaluation Step")
            plt.ylabel("PSNR")
            
            plt.subplot(144)
            plt.imshow(torch.abs(test_render - test_img).cpu())
            plt.title("Error Map")
            plt.colorbar()
            
            plt.tight_layout()
            plt.show()
            
            coarse_model.train()
            fine_model.train()
        
        # Save checkpoint
        if i % CONFIG["save_every"] == 0:
            save_checkpoint(coarse_model, fine_model, optimizer, i, psnrs, CONFIG)
    
    # Final save
    save_checkpoint(coarse_model, fine_model, optimizer, CONFIG["num_iterations"], psnrs, CONFIG, "final.pt")
    print("Training complete!")

if __name__ == "__main__":
    train_nerf() 
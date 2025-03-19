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

    # it takes the cdf and finds the uniform_samples within it, returning the indices.
    indices = torch.searchsorted(cdf, uniform_samples, right=True)
    # this will return a tensor that is either right below the indices u want, or 0 if shits negative 
    below = torch.max(torch.zeros_like(indices), indices-1)
    above = torch.min(indices, (coarse_t_vals.shape[-1]-1) * torch.ones_like(indices))
    t_below = torch.gather(coarse_t_vals, -1, below)
    t_above = torch.gather(coarse_t_vals, -1, above)

    t_above[indices == coarse_t_vals.shape[-1]] = far_bound

    t_vals_fine = t_below + (t_above - t_below) * torch.rand_like(t_below)

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























    indices = torch.searchsorted(cdfs, uniform_samples, right=True)
    below = torch.max(torch.zeros_like(indices), indices-1)
    above = torch.min((coarse_t_vals.shape[-1]-1) * torch.ones_like(indices), indices)
    t_below = torch.gather(coarse_t_vals, -1, below)
    t_above = torch.gather(coarse_t_vals, -1, above)
    t_above[indices == coarse_t_vals.shape[-1]] = far_bound

    t_vals_fine = t_below + (t_above - t_below) * torch.rand_like(t_below)

    # Combine and sort all samples
    t_vals_all, _ = torch.sort(torch.cat([coarse_t_vals, t_vals_fine], dim=-1), dim=-1)
    points = ray_origins[..., None, :] + t_vals_all[..., :, None] * ray_directions[..., None, :]

    return points, t_vals_all




 















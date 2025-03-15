### **Neural Radiance Fields** Implementation in PyTorch

Derived from methodology outlined in the NeRF paper (Mildenhall et al., 2020) [https://arxiv.org/abs/2003.08934]

## Quick Start (Google Colab)

1. Create a new Colab notebook
2. Copy the contents of `fullnerf_colab.py` into three separate cells:
   - Cell 1: Everything under "Cell 1: Setup and Dependencies"
   - Cell 2: Everything under "Cell 2: Imports and Config"
   - Cell 3: Everything under "Cell 3: NeRF Implementation"
3. Set runtime to GPU: Runtime -> Change runtime type -> GPU
4. Run all cells in order

## Training

- Training takes ~2-3 hours on Colab GPU
- Progress visualizations every 2000 iterations
- Checkpoints saved every 10000 iterations
- Default test view: 150th image

## Configuration

Key parameters in `CONFIG` dictionary:

- `num_iterations`: 200000 (reduce for testing)
- `batch_size`: 64 (patch size for training)
- `learning_rate`: 5e-4
- `device`: Automatically selects GPU/CPU

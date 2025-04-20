from torch.utils.checkpoint import checkpoint

from config import CONFIG
from unetv2 import UNet
from diffusion import GaussianDiffusion
import torch
from torchvision.utils import save_image
import os


if __name__ == "__main__":
    device = CONFIG.device
    if device.type == 'cuda':
        checkpoint = "unet_weights_cuda.pt"
    elif device.type == 'cpu':
        checkpoint = "unet_weights_cpu.pt"

    model = UNet(*CONFIG.MODEL.get_params()).to(device)
    model.load_from_pretrained(checkpoint, device=device)
    diffusion = GaussianDiffusion(model, CONFIG.DIFFUSION.timesteps, CONFIG.DIFFUSION.beta_schedule, device=device)

    os.makedirs("samples", exist_ok=True)
    grid = diffusion.sample()
    save_name = os.path.join("samples", f"samples_{CONFIG.DATASET.name}_cfg={CONFIG.DIFFUSION.guidance_strength}.png")
    save_image(grid, save_name, nrow=CONFIG.DATASET.num_classes, normalize=True)
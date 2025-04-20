from torch.utils.checkpoint import checkpoint

from config import CONFIG
from unetv2 import UNet
from diffusion import GaussianDiffusion
from torchvision.utils import save_image
import os
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sample 5 images per class using the diffusion model and save them in the samples folder.")
    parser.add_argument("-g", "--cfg_scale", type=float, default=CONFIG.DIFFUSION.guidance_strength,
                        help="CFG guidance strength (default: value from CONFIG).")
    args = parser.parse_args()

    device = CONFIG.device
    checkpoint = f"unet_weights_{CONFIG.DATASET.name}_{device.type}.pt"

    model = UNet(*CONFIG.MODEL.get_params()).to(device)
    model.load_from_pretrained(checkpoint, device=device)
    diffusion = GaussianDiffusion(model, CONFIG.DIFFUSION.timesteps, CONFIG.DIFFUSION.beta_schedule, device=device)

    os.makedirs("samples", exist_ok=True)
    grid = diffusion.sample()
    save_name = os.path.join("samples", f"samples_{CONFIG.DATASET.name}_cfg={CONFIG.DIFFUSION.guidance_strength}.png")
    save_image(grid, save_name, nrow=CONFIG.DATASET.num_classes, normalize=True)
    print(f"Sampled images saved to {save_name}")
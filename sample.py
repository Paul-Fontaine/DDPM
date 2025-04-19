from unet import UNet
from diffusion import GaussianDiffusion
import torch
from torchvision.utils import save_image
import os


if __name__ == "__main__":
    checkpoint = "unet_weights_cuda.pt"
    device = torch.device('cuda' if 'cuda' in checkpoint else 'cpu')
    num_classes = 10
    model = UNet(num_classes, in_ch=1, base_ch=64, depth=2, time_emb_dim=128).to(device)
    model.load_from_pretrained(checkpoint, device=device)
    diffusion = GaussianDiffusion(model, timesteps=1000, device=device)

    os.makedirs("samples", exist_ok=True)

    for cfg_scale in range(1, 11):
        cfg_scale = float(cfg_scale)
        print(f"Sampling images with CFG scale: {cfg_scale}")
        grid = diffusion.sample(num_classes, (1, 28, 28), cfg_scale, device)
        save_image(grid, f"samples/sampled_images_{cfg_scale}.png", nrow=num_classes)

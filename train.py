# train.py

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import os
import random

from dataset import dataset
from unet import UNet
from diffusion import GaussianDiffusion

# --- Config ---
epochs = 20
batch_size = 10
lr = 2e-4
images_shape = (1, 28, 28)  # MNIST images
num_classes = None
drop_label_prob = 0.1  # For classifier-free guidance
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Dataloader ---
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

# --- Model + Diffusion ---
model = UNet(num_classes=num_classes, in_ch=1, base_ch=64, depth=2, time_emb_dim=128).to(device)
diffusion = GaussianDiffusion(model, timesteps=1000, device=device)

optimizer = optim.Adam(model.parameters(), lr=lr)

# --- Logging ---
n_images_generated_per_class = 2
log_dir = "./runs/ddpm_mnist"
os.makedirs("samples", exist_ok=True)
writer = SummaryWriter(log_dir=log_dir)


def train_ddpm():
    global_step = 0

    # --- Training loop ---
    for epoch in range(epochs):
        model.train()
        total_loss = 0

        with tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}", unit="batch") as tqdm_loader:
            tqdm_loader.set_postfix(loss=0)
            for i, (images, labels) in enumerate(tqdm_loader):
                print(f"Batch {i+1}/{len(loader)}")
                images = images.to(device)
                labels = labels.to(device)
                B = labels.shape[0]

                # --- Drop labels randomly for CFG ---
                if num_classes is not None:
                    drop_mask = torch.rand(labels.shape[0], device=device) < drop_label_prob
                    y = labels.clone()
                    y[drop_mask] = -1  # -1 indicates no label
                else:
                    y = None

                # Icrease the maximum noise level linearly
                t_max = int(diffusion.timesteps * (epoch+1/10)) if epoch < 10 else diffusion.timesteps
                t = torch.randint(0, t_max, (B,), dtype=torch.int16, device=device)

                print("start training")
                loss = diffusion.p_losses(images, t, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                print("logging")
                tqdm_loader.set_postfix(loss=loss.item())
                writer.add_scalar("Loss", loss.item(), global_step)
                total_loss += loss.item()
                global_step += 1

        avg_loss = total_loss / len(loader)
        writer.add_scalar("Average Loss", avg_loss, epoch)

        # --- Sampling ---
        print("Sampling...")
        model.eval()
        with torch.no_grad():
            if num_classes is not None:
                class_labels = torch.arange(num_classes, device=device).repeat(n_images_generated_per_class, 1).flatten()
            else:
                class_labels = None
            sampled = diffusion.p_sample_loop(images_shape=images_shape, y=class_labels)

            grid = make_grid(sampled, nrow=n_images_generated_per_class)
            writer.add_image("Samples", grid, epoch)

        # --- Save checkpoint ---
        torch.save(model.state_dict(), f"unet_weights.pt")

    writer.close()


if __name__ == "__main__":
    train_ddpm()
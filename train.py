# train.py
from config import CONFIG

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import time

from dataset import dataset
from unetv2 import UNet
from diffusion import GaussianDiffusion


def train_ddpm():
    epochs = CONFIG.TRAIN.num_epochs
    device = CONFIG.device

    # --- Dataloader ---
    loader = DataLoader(dataset, batch_size=CONFIG.TRAIN.batch_size, shuffle=True, num_workers=4)

    # --- Model + Diffusion ---
    model = UNet(*CONFIG.MODEL.get_params()).to(device)
    if CONFIG.MODEL.checkpoint is not None:
        print(f"Loading checkpoint from {CONFIG.MODEL.checkpoint}")
        model.load_from_pretrained(CONFIG.MODEL.checkpoint, device=device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    # print(model)
    diffusion = GaussianDiffusion(model, CONFIG.DIFFUSION.timesteps, CONFIG.DIFFUSION.beta_schedule, device=device)

    optimizer = optim.Adam(model.parameters(), lr=CONFIG.TRAIN.lr)

    # --- Logging ---
    time_formatted = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    log_dir = f"logs/{time_formatted}_epochs={epochs}_batch_size={CONFIG.TRAIN.batch_size}_guidance={CONFIG.DIFFUSION.guidance_strength}"
    writer = SummaryWriter(log_dir=log_dir)

    global_step = 0

    # --- Training loop ---
    for epoch in range(epochs):
        model.train()
        total_loss = 0

        with tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}", unit="batch") as tqdm_loader:
            tqdm_loader.set_postfix(loss=0)
            for i, (images, labels) in enumerate(tqdm_loader):
                images = images.to(device)
                labels = labels.to(device)
                B = labels.shape[0]

                # --- Drop labels randomly for CFG ---
                if CONFIG.TRAIN.drop_label_prob > 0:
                    drop_mask = torch.rand(labels.shape[0], device=device) < CONFIG.TRAIN.drop_label_prob
                    y = labels.clone()
                    y[drop_mask] = -1  # -1 indicates no label
                else:
                    y = None

                # Icrease the maximum noise level linearly
                E = CONFIG.TRAIN.increase_tmax_progressively_until_epoch
                if E is not None:
                    t_max = int(diffusion.timesteps * (epoch+1) /E ) if epoch < E else diffusion.timesteps
                else:
                    t_max = diffusion.timesteps
                t = torch.randint(0, t_max, (B,), dtype=torch.int16, device=device)

                loss = diffusion.p_losses(images, t, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss = loss.item()
                tqdm_loader.set_postfix(loss=loss)
                writer.add_scalar("Loss", loss, global_step)
                total_loss += loss
                global_step += 1

        avg_loss = total_loss / len(loader)
        writer.add_scalar("Average Loss", avg_loss, epoch)

        # --- Sampling ---
        grid = diffusion.sample()
        writer.add_image("Samples", grid, epoch)

        # --- Save checkpoint ---
        torch.save(model.state_dict(), f"unet_weights_{CONFIG.DATASET.name}_{device.type}.pt")

    writer.close()


if __name__ == "__main__":
    print(f"Using device: {CONFIG.device}")
    print(CONFIG())

    train_ddpm()

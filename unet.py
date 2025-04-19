# unet.py
from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def timestep_embedding(timesteps: List[int], dim):
    """
    Create sinusoidal timestep embeddings
    :param timesteps: [B] - integer time steps
    :param dim: embedding vectors dimension
    :return: embedding matrix [B, dim]
    """
    half = dim // 2
    emb = math.log(10000) / (half - 1)
    emb = torch.exp(torch.arange(half, device=timesteps.device) * -emb)
    emb = timesteps[:, None] * emb[None, :]  # [B, dim/2]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    return emb  # [B, dim]


# ----------------------
# Simple Block: Conv + Norm + ReLU
# ----------------------
class Block(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        """
        Simple block with Conv2d, GroupNorm, and ReLU.
        :param in_ch: number of input channels
        :param out_ch: number of output channels
        """
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(1, out_ch),  # single group normalization
            nn.ReLU()
        )

    def forward(self, x):
        return self.block(x)


# ----------------------
# FiLM layer (for class conditioning)
# ----------------------
class FiLM(nn.Module):
    def __init__(self, in_ch: int, emb_ch: int):
        """
        FiLM layer: applies a linear transformation to the embedding and uses it to scale and shift the input.
        :param in_ch: number of input channels
        :param emb_ch: number of embedding channels
        """
        super().__init__()
        self.linear = nn.Linear(emb_ch, in_ch * 2)

    def forward(self, x, emb):
        # x: [B, C, H, W], emb: [B, emb_ch]
        scale, shift = self.linear(emb).chunk(2, dim=1)  # [B, C], [B, C]
        scale = scale[:, :, None, None]  # [B, C, 1, 1]
        shift = shift[:, :, None, None]
        return x * (1 + scale) + shift


# ----------------------
# U-Net for denoising
# ----------------------
class UNet(nn.Module):
    def __init__(self, num_classes: int | None = None, in_ch: int = 1, base_ch: int = 64, depth: int = 2, time_emb_dim: int = 128):
        """
        U-Net architecture for denoising diffusion models.
        :param num_classes: int or None. If None, the model is unconditional. If int you can choose the class of the generated image.
        :param in_ch: number of input channels (e.g., 1 for grayscale images, 3 for RGB).
        :param depth: number of downsampling layers.
        :param base_ch: number of channels in the first layer. It will be multiplied by 2, 4, etc. in the downsampling path.
        :param time_emb_dim: dimension of the time embedding vector.
        """
        super().__init__()
        self.num_classes = num_classes
        self.depth = depth

        # Time embedding MLP
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.ReLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim)
        )

        # Optional class embedding
        if num_classes is not None:
            self.class_emb = nn.Embedding(num_classes, time_emb_dim)

        # Initial input conv
        self.conv_in = Block(in_ch, base_ch)

        # Downsampling blocks
        self.downs = nn.ModuleList()
        self.down_film = nn.ModuleList()
        channels = [base_ch]
        for i in range(depth):
            in_c = base_ch * (2 ** i)
            out_c = base_ch * (2 ** (i + 1))
            self.downs.append(Block(in_c, out_c))
            self.down_film.append(FiLM(out_c, time_emb_dim))
            channels.append(out_c)

        # Upsampling blocks
        self.ups = nn.ModuleList()
        self.up_film = nn.ModuleList()
        for i in reversed(range(depth)):
            in_c = base_ch * (2 ** (i + 1)) + base_ch * (2 ** i)  # skip connection concat
            out_c = base_ch * (2 ** i)
            self.ups.append(Block(in_c, out_c))
            self.up_film.append(FiLM(out_c, time_emb_dim))

        self.conv_out = nn.Conv2d(base_ch, in_ch, 1)

        self.pool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x, t, y=None):
        """
        x: [B, in_ch, H, W]
        t: [B]
        y: [B] or None
        """
        B = x.size(0)

        # Compute embeddings
        t_emb = timestep_embedding(t, self.time_mlp[0].in_features)
        t_emb = self.time_mlp(t_emb)

        if self.num_classes is not None and y is not None:
            y_emb = self.class_emb(y)
            t_emb = t_emb + y_emb

        # Initial conv
        h = self.conv_in(x)
        skips = []

        # Downsampling
        for i in range(self.depth):
            h = self.pool(h)
            h = self.downs[i](h)
            h = self.down_film[i](h, t_emb)
            skips.append(h)

        # Bottleneck = last downsampled h
        h = skips.pop()

        # Upsampling
        for i in range(self.depth):
            h = self.upsample(h)
            skip = skips.pop()
            h = torch.cat([h, skip], dim=1)
            h = self.ups[i](h)
            h = self.up_film[i](h, t_emb)

        return self.conv_out(h)

    def load_from_pretrained(self, path, device='cpu'):
        """
        Load weights from a pretrained model.
        :param path: path to the pretrained model.
        """
        state_dict = torch.load(path, map_location=device, weight_only=True)
        self.load_state_dict(state_dict['model'], strict=False)

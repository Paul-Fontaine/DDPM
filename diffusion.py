# diffusion.py
from config import CONFIG
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import make_grid


# Cosine schedule from Nichol & Dhariwal
def cosine_beta_schedule(timesteps, s=0.008):
    """
    Cosine schedule: generates a sequence of betas that define how much noise to add at each time step.
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)  # avoid extreme values


def linear_beta_schedule(timesteps):
    """
    Linear schedule: generates a sequence of betas that define how much noise to add at each time step.
    """
    return torch.linspace(1e-4, 0.02, timesteps)


class GaussianDiffusion(nn.Module):
    def __init__(self, model, timesteps=1000, beta_schedule='cosine', device='cuda'):
        """
        model: neural network that predicts the noise
        timesteps: total diffusion steps (e.g., 1000)
        """
        super().__init__()
        self.model = model
        self.timesteps = timesteps
        self.device = device

        # Noise schedule
        if beta_schedule == 'cosine':
            betas = cosine_beta_schedule(timesteps)
        elif beta_schedule == 'linear':
            betas = linear_beta_schedule(timesteps)
        else:
            raise NotImplementedError(f"{beta_schedule} schedule not supported")
        betas = betas.to(device)

        # Register buffers (non-trainable constants)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", 1. - betas)
        self.register_buffer("alphas_cumprod", torch.cumprod(self.alphas, dim=0))  # [T]
        self.register_buffer("alphas_cumprod_prev", torch.cat([torch.tensor([1.]).to(device), self.alphas_cumprod[:-1]]))

        # Precomputed terms for efficient sampling
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(self.alphas_cumprod))                      # [T]
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1. - self.alphas_cumprod))      # [T]
        self.register_buffer("log_one_minus_alphas_cumprod", torch.log(1. - self.alphas_cumprod))
        self.register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1. / self.alphas_cumprod))          # [T]
        self.register_buffer("sqrt_recipm1_alphas_cumprod", torch.sqrt(1. / self.alphas_cumprod - 1))    # [T]

        # Compute posterior variance, mean coefficients for q(x_{t-1} | x_t, x_0)
        self.register_buffer("posterior_variance", self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod))
        self.register_buffer("posterior_log_variance_clipped", torch.log(torch.clamp(self.posterior_variance, min=1e-20)))
        self.register_buffer("posterior_mean_coef1", self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1. - self.alphas_cumprod))
        self.register_buffer("posterior_mean_coef2", (1. - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1. - self.alphas_cumprod))

    def q_sample(self, x_start, t, noise=None):
        """
        Forward diffusion process: sample x_t from x_0.
        x_start: [B, C, H, W] - original images
        t: [B] - time steps
        Returns: x_t: [B, C, H, W]
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_alpha_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alpha_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        return sqrt_alpha_cumprod_t * x_start + sqrt_one_minus_alpha_cumprod_t * noise

    def p_losses(self, x_start, t, y):
        """
        Training loss function.
        x_start: clean image [B, C, H, W]
        t: time step [B]
        y: class labels
        Returns: scalar loss
        """
        noise = torch.randn_like(x_start)  # [B, C, H, W]
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)  # [B, C, H, W]

        predicted_noise = self.model(x_noisy, t, y)  # [B, C, H, W]
        return F.mse_loss(predicted_noise, noise)

    @torch.no_grad()
    def q_posterior(self, x_start, x_t, t):
        """
        Compute the mean and variance of q(x_{t-1} | x_t, x_0)
        :param x_start: predicted x_0, [B, C, H, W]
        :param x_t: current image x_t, [B, C, H, W]
        :param t: current timestep t, [B]
        :return: mean and variance for posterior q(x_{t-1} | x_t, x_0)
        """
        posterior_mean = (
                self.posterior_mean_coef1[t].view(-1, 1, 1, 1) * x_start +
                self.posterior_mean_coef2[t].view(-1, 1, 1, 1) * x_t
        )
        posterior_variance = self.posterior_variance[t].view(-1, 1, 1, 1)

        return posterior_mean, posterior_variance

    @torch.no_grad()
    def p_sample(self, x, t, y: List[int] = None, cfg_scale: float = 5.0):
        """
        :param x: [B, C, H, W] current noisy image
        :param t: [B] urrent timestep
        :param y: class labels or None
        :param cfg_scale: CFG guidance strength
        :return: x_t-1: [B, C, H, W] denoised image
        """
        # Unconditional prediction (y = None)
        eps_uncond = self.model(x, t, y=None)

        if y is not None:
            eps_cond = self.model(x, t, y)
            eps = eps_uncond + cfg_scale * (eps_cond - eps_uncond)
        else:
            eps = eps_uncond

        # Denoise using predicted noise
        sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
        sqrt_recip_alphas_t = self._extract(self.sqrt_recip_alphas_cumprod, t, x.shape)

        pred_x0 = (x - sqrt_one_minus_alphas_cumprod_t * eps) / sqrt_recip_alphas_t

        posterior_mean, posterior_variance = self.q_posterior(pred_x0, x, t)

        noise = torch.randn_like(x) if t[0] > 0 else torch.zeros_like(x)
        return posterior_mean + torch.sqrt(posterior_variance) * noise

    @torch.no_grad()
    def p_sample_loop(self, images_shape, y: List[int] = None, cfg_scale: float = 5.0):
        """
        :param images_shape: (C, H, W) — output image image_shape
        :param y: class labels of the images to generate. If y = None, sample 10 images without conditions.
        :param cfg_scale: CFG guidance strength
        :return: [B, C, H, W] generated images
        """
        B = len(y) if y is not None else 10
        imgs = torch.randn((B, *images_shape), device=self.device)  # [B, C, H, W]

        for i in reversed(range(0, self.timesteps)):
            t = torch.full((B,), i, device=self.device, dtype=torch.long)
            imgs = self.p_sample(imgs, t, y=y, cfg_scale=cfg_scale)

        return imgs

    def sample(self,
               num_classes = CONFIG.DATASET.num_classes,
               images_shape = CONFIG.DATASET.images_shape,
               cfg_scale = CONFIG.DIFFUSION.guidance_strength,
               device = CONFIG.device):
        """
        :return: grid of images with shape [5, num_classes] because it samples 5 images for each class.
        """
        self.eval()
        self.model.eval()
        with torch.no_grad():
            labels = torch.arange(num_classes, device=device).repeat(5, 1).flatten()
            images = self.p_sample_loop(images_shape, labels, cfg_scale)
            grid = make_grid(images, nrow=num_classes, normalize=True)
        return grid


    def _extract(self, a, t, x_shape):
        """
        Extract coefficients at given time steps and reshape for broadcasting.
        a: precomputed tensor [T]
        t: time steps [B]
        x_shape: shape of input (usually [B, C, H, W])
        Returns: tensor of shape [B, 1, 1, 1] for broadcasting
        """
        batch_size = t.shape[0]
        out = a.gather(-1, t.to(a.device).long()).to(self.device)  # [B]
        return out.view(batch_size, *((1,) * (len(x_shape) - 1)))  # [B, 1, 1, 1]

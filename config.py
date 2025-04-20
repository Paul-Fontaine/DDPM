import torch

class TRAIN_CONFIG:
    num_epochs = 30
    batch_size = 64
    lr = 2e-4
    drop_label_prob = 0.1  # For classifier-free guidance

class MODEL_CONFIG:
    down_chs = (32, 64, 128, 256)
    down_sample = (True, True, False)
    mid_chs = (256, 256, 128)
    up_chs = (256, 128, 64, 16)
    t_emb_dim = 128
    num_downc_layers: int = 2
    num_midc_layers: int = 2
    num_upc_layers: int = 2
    checkpoint = None # will resume training from this checkpoint if not None
    def __init__(self, num_classes, img_ch):
        self.num_classes = num_classes
        self.img_ch = img_ch

    def get_params(self):
        """
        return the model parameters as a tuple
        """
        return (self.num_classes, self.img_ch, self.down_chs,
                self.mid_chs, self.up_chs, self.down_sample, self.t_emb_dim, self.num_downc_layers,
                self.num_midc_layers, self.num_upc_layers)

class DIFFUSION_CONFIG:
    timesteps = 1000
    beta_schedule = "cosine"
    guidance_strength = 5.0

class DATASET_CONFIG:
    name = "CIFAR10" # or "CIFAR10" or "animals"
    num_classes = 10
    images_shape = (3, 32, 32)  # (1, 28, 28) for MNIST ¦ (3, 32, 32) for CIFAR10

class CONFIG:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    DATASET = DATASET_CONFIG()
    TRAIN = TRAIN_CONFIG()
    MODEL = MODEL_CONFIG(DATASET.num_classes, DATASET.images_shape[0])
    DIFFUSION = DIFFUSION_CONFIG()
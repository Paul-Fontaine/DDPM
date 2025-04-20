# dataset.py
from config import CONFIG
from torchvision import datasets, transforms

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]
])

if CONFIG.DATASET.name == "MNIST":
    dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
elif CONFIG.DATASET.name == "CIFAR10":
    dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
elif CONFIG.DATASET.name == "animals":
    dataset = datasets.ImageFolder(root="./data/animals/", transform=transform)

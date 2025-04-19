# dataset.py

from torchvision import datasets, transforms

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]
])

# You can later change this to CIFAR10, ImageNet, etc.
dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)

# dataset.py

from torchvision import datasets, transforms

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]
])

# You can later change this to CIFAR10, ImageNet, etc.
dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)

# reduce dataset size for faster training, keep 10 images for each class
indices = []
for i in range(10):
    class_indices = [j for j, (_, label) in enumerate(dataset) if label == i]
    indices.extend(class_indices[:10])  # Keep only 10 images per class
dataset.data = dataset.data[indices]
dataset.targets = dataset.targets[indices]


# Print dataset size
print(f"Dataset size: {len(dataset)}")
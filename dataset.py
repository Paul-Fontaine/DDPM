# dataset.py
from config import CONFIG
import random
from torchvision import datasets, transforms
from torchvision.transforms import functional as F

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]
])

class RandomSquareCrop:
    def __call__(self, img):
        width, height = img.size
        crop_size = min(width, height)

        if width == height:
            return img
        elif width > height:
            left = random.randint(0, width - crop_size)
            top = 0
        else:
            left = 0
            top = random.randint(0, height - crop_size)

        return F.crop(img, top, left, crop_size, crop_size)

transform_animals = transforms.Compose([
    RandomSquareCrop(),
    transforms.Resize((CONFIG.DATASET.images_shape[1], CONFIG.DATASET.images_shape[2])),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to [-1, 1]
])

if CONFIG.DATASET.name == "MNIST":
    dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
elif CONFIG.DATASET.name == "CIFAR10":
    dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
elif CONFIG.DATASET.name == "animals":
    import kagglehub
    import os
    import shutil

    if not os.path.exists("./data/animals/"):
        def fusion_animals(path):

            # Dossiers source
            root_dir = os.path.join(path, "animals/")
            output_dir = "data/animals"

            # Créer le dossier de sortie
            os.makedirs(output_dir, exist_ok=True)

            # Fusionner train et validation
            for split in ["train", "val"]:
                split_path = os.path.join(root_dir, split)
                for class_name in os.listdir(split_path):
                    class_src = os.path.join(split_path, class_name)
                    class_dst = os.path.join(output_dir, class_name)

                    os.makedirs(class_dst, exist_ok=True)

                    for filename in os.listdir(class_src):
                        src_file = os.path.join(class_src, filename)
                        dst_file = os.path.join(class_dst, filename)
                        # Pour éviter les conflits de noms
                        if os.path.exists(dst_file):
                            base, ext = os.path.splitext(filename)
                            count = 1
                            while os.path.exists(dst_file):
                                dst_file = os.path.join(class_dst, f"{base}_{count}{ext}")
                                count += 1
                        shutil.copy2(src_file, dst_file)
            print("Merged train and test folders into ./data/animals")

        path = kagglehub.dataset_download("antobenedetti/animals")
        print("Animals dataset downloaded in : ", path)
        fusion_animals(path)
        path_debut = path.split("datasets")[0]
        shutil.rmtree(path_debut)

    dataset = datasets.ImageFolder(root="./data/animals/", transform=transform)

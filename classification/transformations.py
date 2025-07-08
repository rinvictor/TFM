from torchvision import transforms
import random
import numpy as np
from PIL import Image
import os

def load_mean_std(file_path):
    with open(file_path, "r") as f:
        lines = f.readlines()
    mean = eval(lines[0].strip())
    std = eval(lines[1].strip())
    return mean, std

def get_train_transform(image_size=(224, 224), dataset_path=None):
    # return transforms.Compose([
    #     transforms.RandomAffine(
    #         degrees=30,
    #         translate=(0.1, 0.1),
    #         scale=(0.9, 1.2),
    #         shear=10
    #     ),
    #     transforms.RandomHorizontalFlip(p=0.5),
    #     transforms.RandomVerticalFlip(p=0.5),
    #     transforms.ColorJitter(
    #         brightness=0.1,
    #         contrast=0.1,
    #         saturation=0.1,
    #         hue=0
    #     ),
    #     transforms.CenterCrop(image_size),  # Ensure the image center
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                          std=[0.229, 0.224, 0.225]),
    # ])
    # return transforms.Compose([
    #     transforms.Resize(256),
    #     transforms.CenterCrop(image_size),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                          std=[0.229, 0.224, 0.225]),
    # ])
    # return A.Compose([
    #     A.Resize(256, 256),
    #     A.CenterCrop(image_size),  # Asegura el tamaño final
    #     A.HorizontalFlip(p=0.5),
    #     A.VerticalFlip(p=0.5),
    #     A.RandomBrightnessContrast(p=0.5),
    #     A.HueSaturationValue(p=0.5),
    #     A.ElasticTransform(alpha=1, sigma=50, p=0.3),  # sin alpha_affine
    #     A.GaussNoise(p=0.3),  # sin var_limit
    #     A.CoarseDropout(num_holes=2, max_height=32, max_width=32, p=0.3),  # usa estos parámetros
    #     A.Normalize(mean=[0.485, 0.456, 0.406],
    #                           std=[0.229, 0.224, 0.225]),
    #     A.ToTensorV2(),  # Convierte a tensor de PyTorch
    # ])
    if dataset_path is not None:
        mean, std = load_mean_std(os.path.join(dataset_path, "mean_std.txt"))
    else: # imagenet default values
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    return transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ColorJitter(
            brightness=0.5,
            contrast=0.5,
            saturation=0.5,
            hue=0.1
        ),  # combina RandomBrightnessContrast y HueSaturationValue
        # ElasticTransform(alpha=1, sigma=50, p=0.3),  # ElasticTransform personalizado
        # GaussianNoise(std=0.1, p=0.3),  # GaussNoise personalizado (std=0.1 es un ejemplo)
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.3), #extra
        transforms.RandomRotation(30), #extra
        transforms.Resize((256)),  # Resize fijo 256x256
        transforms.CenterCrop(image_size),  # Crop central 224x224 (ajusta según image_size)
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                      std=[0.229, 0.224, 0.225]),
        transforms.Normalize(mean=mean, std=std),

        transforms.RandomErasing(p=0.3),  # Equivalente aproximado a CoarseDropout
    ])


def get_val_transform(image_size=(224, 224), dataset_path=None): #Si uso 512x512, el resize deberia ser de 576, recvisar si son cuadradas
    if dataset_path is not None:
        mean, std = load_mean_std(os.path.join(dataset_path, "mean_std.txt"))
    else:  # imagenet default values
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                      std=[0.229, 0.224, 0.225]),
        transforms.Normalize(mean=mean, std=std),
    ])





class ElasticTransform:
    def __init__(self, alpha=1, sigma=50, p=0.3):
        self.alpha = alpha
        self.sigma = sigma
        self.p = p

    def __call__(self, img):
        if random.random() > self.p:
            return img

        # Convierte a numpy array
        image_np = np.array(img)

        # Generar campos de desplazamiento aleatorios
        random_state = np.random.RandomState(None)
        shape = image_np.shape[:2]

        dx = (random_state.rand(*shape) * 2 - 1)
        dy = (random_state.rand(*shape) * 2 - 1)

        from scipy.ndimage import gaussian_filter, map_coordinates
        dx = gaussian_filter(dx, self.sigma, mode="reflect") * self.alpha
        dy = gaussian_filter(dy, self.sigma, mode="reflect") * self.alpha

        x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        indices = (y + dy).reshape(-1), (x + dx).reshape(-1)

        for i in range(image_np.shape[2]):  # Para cada canal
            image_np[..., i] = map_coordinates(image_np[..., i], indices, order=1, mode='reflect').reshape(shape)

        return Image.fromarray(image_np.astype(np.uint8))

# Transformación personalizada: Gaussian Noise
class GaussianNoise:
    def __init__(self, mean=0., std=0.1, p=0.3):
        self.mean = mean
        self.std = std
        self.p = p

    def __call__(self, img):
        if random.random() > self.p:
            return img
        img_np = np.array(img).astype(np.float32) / 255.0
        noise = np.random.normal(self.mean, self.std, img_np.shape)
        img_np = img_np + noise
        img_np = np.clip(img_np, 0, 1)
        img_np = (img_np * 255).astype(np.uint8)
        return Image.fromarray(img_np)
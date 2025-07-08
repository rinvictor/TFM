import pandas as pd
from PIL import Image
import os

from torchvision.transforms import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import random

def get_isic_transform():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.HueSaturationValue(p=0.5),
        A.ElasticTransform(alpha=1, sigma=50, p=0.3),  # sin alpha_affine
        A.GaussNoise(p=0.3),  # sin var_limit
        A.CoarseDropout(num_holes=2, max_height=32, max_width=32, p=0.3),  # usa estos parámetros
    ])

def get_isic_transform_randomized():
    num_transforms = random.randint(2, 5)
    return A.Compose([
        A.OneOf([
            A.HorizontalFlip(p=1.0),
            A.VerticalFlip(p=1.0),
            A.RandomRotate90(p=1.0),
        ], p=0.7),  # One of the three flips or rotations with 70% probability

        A.SomeOf([
            A.RandomBrightnessContrast(p=0.5),
            A.HueSaturationValue(p=0.5),
            A.ElasticTransform(alpha=1, sigma=50, p=0.3),
            A.GaussNoise(p=0.3),
            A.CoarseDropout(num_holes=2, max_height=32, max_width=32, p=0.3),
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.5),
            A.Blur(blur_limit=3, p=0.2),
            A.Sharpen(p=0.2),
        ], n=num_transforms, replace=False, p=1.0),  # Between 2 and 4 transformations from the list, without replacement
    ])


def get_isic_transform_randomized_v2():
    num_transforms = random.randint(3, 5)
    return A.Compose([
        A.OneOf([
            A.HorizontalFlip(p=1.0),
            A.VerticalFlip(p=1.0),
            A.RandomRotate90(p=1.0),
        ], p=0.7),

        A.OneOf([
            A.MotionBlur(blur_limit=3, p=1.0),
            A.MedianBlur(blur_limit=3, p=1.0),
            A.GaussNoise(p=1.0),
        ], p=0.4),

        A.SomeOf([
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.6),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=10, val_shift_limit=10, p=0.5),
            A.RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=10, p=0.3),
            A.RandomGamma(gamma_limit=(80, 120), p=0.3),
            A.Equalize(p=0.3),

            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=10, p=0.5),  # OK según el warning
            A.ElasticTransform(alpha=0.5, sigma=30, p=0.2),  # Quitamos alpha_affine

            # Versión válida de CoarseDropout
            A.CoarseDropout(
                p=0.3
            ),

            A.Blur(blur_limit=3, p=0.2),
            A.Sharpen(alpha=(0.1, 0.3), lightness=(0.7, 1.0), p=0.2),
        ], n=num_transforms, replace=False, p=1.0),
    ])



def generate_synthetic_images(output_path, images_to_transform,num_augmentations_per_image):
    os.makedirs(output_path, exist_ok=True)
    synthetic_records = []
    for image_path in images_to_transform:
        original_image = Image.open(image_path).convert('RGB')
        for i in range(num_augmentations_per_image):
            transform = get_isic_transform_randomized_v2()
            image_np = np.array(original_image)

            base_name = os.path.splitext(os.path.basename(image_path))[0]
            output_filename = f"{base_name}_aug_randomized_{i}.jpg"
            full_output_path = os.path.join(output_path, output_filename)

            augmented = transform(image=image_np)
            augmented_image_np = augmented['image']  # (H, W, C), uint8
            augmented_image = Image.fromarray(augmented_image_np)
            augmented_image.save(full_output_path)

            print(f"Saved augmented image: {output_filename}")
            synthetic_records.append({
                'image_path': full_output_path,
                'source_image': image_path
            })
    df_output = pd.DataFrame(synthetic_records)
    csv_path = os.path.join(output_path, "augmented_images.csv")
    df_output.to_csv(csv_path, index=False)
    print(f"Saved CSV with augmented image paths at: {csv_path}")


if __name__ == "__main__":
    """
    Extrayendo tamaños: 100%|████████████████| 26161/26161 [00:40<00:00, 646.06it/s]
             width                             height                          
              mean      std  min   max  mode     mean      std  min   max  mode
label                                                                          
benign     4059.19  2113.53  640  6000  6000  2689.69  1441.00  480  6000  4000
malignant  2791.26  1529.50  640  6000  1872  1943.69  1060.01  480  4288  1053
    """
    images_path = '/nas/data/isic/raw_images'
    output_path = '/nas/data/isic/synthetic_images_randomized_v3'
    train_original_csv_path = '/nas/data/isic/original/train.csv'
    objective_class = 'malignant'
    num_augmentations_per_image = 20
    df = pd.read_csv(train_original_csv_path)
    images_to_transform = df[df['label'] == objective_class]['image_path'].tolist()

    generate_synthetic_images(output_path, images_to_transform, num_augmentations_per_image)
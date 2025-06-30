from torchvision import transforms

def get_train_transform(image_size=(224, 224)):
    return transforms.Compose([
        transforms.RandomAffine(
            degrees=30,
            translate=(0.1, 0.1),
            scale=(0.9, 1.2),
            shear=10
        ),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ColorJitter(
            brightness=0.1,
            contrast=0.1,
            saturation=0.1,
            hue=0
        ),
        transforms.CenterCrop(image_size),  # Ensure the image center
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def get_val_transform(image_size=(224, 224)):
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
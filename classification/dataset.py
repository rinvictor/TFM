from torch.utils.data import Dataset
import cv2
from PIL import Image
from typing import Optional, Callable
from torchvision import transforms
from collections import Counter
from torch.utils.data import DataLoader, WeightedRandomSampler

class ClassificationDataset(Dataset):
    def __init__(self, images_with_labels, transform: Optional[Callable] = None, label_encoding=None):
        super().__init__()
        self.samples = images_with_labels
        self.transform = transform
        self.label_encoding = label_encoding

    def __getitem__(self, index):
        img_path, label = self.samples[index]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        if self.transform:
            image = self.transform(image)

        if self.label_encoding:
            label = self.label_encoding[label]
        return image, label


    def __len__(self):
        return len(self.samples) # todo habra que multiplicar por el numero de augmentaciones


def get_contrastive_loader(dataset, batch_size=64, num_workers=4, augment=True):
    """
    Devuelve un DataLoader para entrenamiento contrastivo, con oversampling opcional y augmentaciones fuertes.
    """
    # Recomendado: augmentaciones fuertes solo para contrastive
    import torchvision.transforms as T
    if augment:
        dataset.transform = T.Compose([
    T.RandomResizedCrop(224),
    T.RandomHorizontalFlip(),
    T.AutoAugment(policy=T.AutoAugmentPolicy.IMAGENET),
    T.ToTensor()
])

    # Estimamos pesos para sampler en función de clases
    labels = [label for _, label in dataset.samples]
    class_counts = Counter(labels)
    num_samples = len(labels)
    weights = [1.0 / class_counts[label] for label in labels]

    sampler = WeightedRandomSampler(weights, num_samples=num_samples, replacement=True)

    loader = DataLoader(dataset,
                        batch_size=batch_size,
                        sampler=sampler,
                        num_workers=num_workers,
                        pin_memory=True,
                        drop_last=True)
    return loader



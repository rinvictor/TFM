from torch.utils.data import Dataset
import cv2
from PIL import Image
from typing import Optional, Callable

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



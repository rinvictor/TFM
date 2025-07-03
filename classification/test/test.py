import os
import pandas as pd
from PIL import Image

import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
from sklearn.metrics import accuracy_score, f1_score

# Dataset
class BinaryImageDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image = Image.open(row['image_path']).convert('RGB')
        label = int(row['label'])

        if self.transform:
            image = self.transform(image)

        return image, label

# Modelo
def get_model():
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 1)  # salida binaria
    return model

# Entrenamiento + validación
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = BinaryImageDataset("train.csv")
    val_dataset = BinaryImageDataset("val.csv")

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    model = get_model().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    num_epochs = 20
    for epoch in range(num_epochs):
        # Entrenamiento
        model.train()
        train_loss = 0.0
        train_preds = []
        train_targets = []

        for imgs, labels in train_loader:
            imgs = imgs.to(device)
            labels = labels.float().unsqueeze(1).to(device)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            preds = torch.sigmoid(outputs).detach().cpu().numpy() > 0.5
            train_preds.extend(preds.astype(int).flatten())
            train_targets.extend(labels.cpu().numpy().flatten().astype(int))

        avg_train_loss = train_loss / len(train_loader)
        train_f1 = f1_score(train_targets, train_preds)

        # Validación
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_targets = []

        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(device)
                labels = labels.float().unsqueeze(1).to(device)

                outputs = model(imgs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                preds = torch.sigmoid(outputs).cpu().numpy() > 0.5
                val_preds.extend(preds.astype(int).flatten())
                val_targets.extend(labels.cpu().numpy().flatten().astype(int))

        avg_val_loss = val_loss / len(val_loader)
        val_f1 = f1_score(val_targets, val_preds)

        print(f"Epoch {epoch+1}/{num_epochs} "
              f"- Train Loss: {avg_train_loss:.4f} | F1: {train_f1:.4f} "
              f"- Val Loss: {avg_val_loss:.4f} | F1: {val_f1:.4f}")

    torch.save(model.state_dict(), "model.pth")
    print("✅ Modelo guardado como model.pth")

if __name__ == "__main__":
    main()

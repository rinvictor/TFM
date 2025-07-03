import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import csv

class SupConLoss(nn.Module):
    """
    Supervised Contrastive Loss.
    """
    def __init__(self, temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        device = features.device
        batch_size = features.shape[0]

        # Normalize features
        features = F.normalize(features, dim=1)

        # Compute similarity matrix
        similarity_matrix = torch.matmul(features, features.T) / self.temperature
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)

        logits_mask = torch.ones_like(mask) - torch.eye(batch_size).to(device)
        mask = mask * logits_mask

        xp_logits = torch.exp(similarity_matrix) * logits_mask
        log_prob = similarity_matrix - torch.log(xp_logits.sum(1, keepdim=True) + 1e-12)

        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1).clamp(min=1.0)
        loss = -mean_log_prob_pos.mean()
        return loss

class ProjectionHead(nn.Module):
    def __init__(self, in_dim, out_dim=128):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, out_dim)
        )

    def forward(self, x):
        return self.proj(x)


class ContrastiveTrainer:
    def __init__(self, encoder, dataloader, projection_head, optimizer, device):
        self.encoder = encoder
        self.dataloader = dataloader
        self.projection_head = projection_head
        self.optimizer = optimizer
        self.device = device
        self.criterion = SupConLoss(temperature=0.15)

    def train(self, epochs=20, log_path="contrastive_train_log.csv"):
        self.encoder.train()
        self.projection_head.train()
        # Abrir el archivo CSV y escribir el encabezado
        with open(log_path, mode='w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['epoch', 'avg_loss'])  # Encabezado

            for epoch in range(epochs):
                total_loss = 0
                progress_bar = tqdm(self.dataloader, desc=f"Contrastive Epoch {epoch + 1}/{epochs}")
                for batch_idx, (x, y) in enumerate(progress_bar):
                    x = x.to(self.device)
                    y = y.to(self.device)

                    features = self.encoder(x)
                    projections = self.projection_head(features)
                    loss = self.criterion(projections, y)

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    total_loss += loss.item()
                    progress_bar.set_postfix({'batch_loss': loss.item(), 'avg_loss': total_loss / (batch_idx + 1)})

                avg_loss = total_loss / len(self.dataloader)
                print(f"Epoch {epoch + 1}/{epochs}, Average loss: {avg_loss}")
                writer.writerow([epoch + 1, avg_loss])
                print(f"Contrastive Epoch {epoch + 1}, Loss: {avg_loss:.4f}")

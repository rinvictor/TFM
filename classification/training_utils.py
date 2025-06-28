from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from torch.optim import Adam, AdamW, SGD
from torch.nn import CrossEntropyLoss
from torchvision import models
import torch.nn as nn
import torch
from tqdm import tqdm
import numpy as np
import random

class OptimizerFactory:
    def get_optimizer(self, optimizer_name: str, model_params, initial_lr, **config):
        if optimizer_name == "adam":
            return _get_adam(model_params, initial_lr, **config)
        elif optimizer_name == "adamw":
            return _get_adamw(model_params, initial_lr, **config)
        elif optimizer_name == "sgd":
            return _get_sgd(model_params, initial_lr, **config)
        else:
            raise NotImplementedError(f"{optimizer_name} optimizer is not implemented")

def _get_adam(model_params, initial_lr, **config):
    return Adam(params=model_params, lr=initial_lr, **config)

def _get_adamw(model_params, initial_lr, **config):
    return AdamW(params=model_params, lr=initial_lr, **config)

def _get_sgd(model_params, initial_lr, **config):
    return SGD(params=model_params, lr=initial_lr, momentum=0.9, **config)


class LossFunctionFactory:
    def get_loss_function(self, loss_name, **kwargs):
        if loss_name == "ce":
            return _get_ce(**kwargs)
        else:
            raise NotImplementedError(f"{loss_name} loss is not implemented")

def _get_ce(**kwargs):
    return CrossEntropyLoss(**kwargs)

# class CustomClassifier(nn.Module):
#     def __init__(self, encoder, num_classes, head=None):
#         super(CustomClassifier, self).__init__()
#         self.encoder = encoder
#         self.head = head if head else nn.Identity()  #Only if head is present
#         self.classifier = nn.Linear(self._get_output_features(),num_classes)
#     def _get_output_features(self):
#         # Pasa un dummy input por el encoder + head para obtener el tamaño de salida
#         dummy_input = torch.randn(1, 3, 224, 224)
#         with torch.no_grad():
#             x = self.encoder(dummy_input)
#             x = self.head(x)
#             # If the encoder returns a 4D tensor (e.g., from a CNN), we need to flatten it
#             # (batch_size, channels * height * width)
#             if x.dim() > 2:
#                 x = torch.flatten(x, 1) # Flatten the output except batch dimension
#         return x.shape[1]
#
#     def forward(self, x):
#         x = self.encoder(x)
#         x = self.head(x)
#         if x.dim() > 2:
#             x = torch.flatten(x, 1)
#         return self.classifier(x)

class CustomClassifier(nn.Module):
    def __init__(self, encoder, num_classes, dropout_rate=0):
        """
        A generic classifier that can adapt to different encoder architectures

        Args:
            encoder: An encoder (ResNet, EfficientNet, MobileNetV2).
            num_classes (int): Number of classes.
            dropout_rate (float): The dropout probability.
        """
        super(CustomClassifier, self).__init__()
        self.encoder = encoder

        num_features = self._get_encoder_output_features(self.encoder)

        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(num_features, num_classes)
        )

    def _get_encoder_output_features(self, encoder):
        if hasattr(encoder, 'fc'):  # Resnet
            return encoder.fc.in_features
        elif hasattr(encoder, 'classifier'):  # EfficientNet, MobileNetV2
            final_layer = encoder.classifier[-1]
            return final_layer.in_features
        else:
            raise NotImplementedError("Not implemented for this encoder type.")

    def forward(self, x):
        if isinstance(self.encoder, models.ResNet):
            # Image → Conv Layers → Pooling → nn.Identity() → 2048 characterístics vector
            original_classifier = self.encoder.fc
            self.encoder.fc = nn.Identity()
            features = self.encoder(x)
            self.encoder.fc = original_classifier

        elif isinstance(self.encoder, (models.EfficientNet, models.MobileNetV2)):
            features = self.encoder.features(x)
            features = self.encoder.avgpool(features)
            features = torch.flatten(features, 1)

        else:
            raise NotImplementedError("Not implemented for this encoder type.")

        # Characteristics to our classifier
        return self.classifier(features)

class EncoderFactory:
    def get_encoder(self, encoder_name, pretrained):
        if encoder_name == "resnet18":
            return _get_resnet_18(pretrained=pretrained)
        elif encoder_name == "resnet50":
            return _get_resnet_50(pretrained=pretrained)
        elif encoder_name == "resnet101":
            return _get_resnet_101(pretrained=pretrained)
        elif encoder_name == "efficientnet-b0":
            return _get_efficientnet_b0(pretrained=pretrained)
        elif encoder_name == "efficientnet-b1":
            return _get_efficientnet_b1(pretrained=pretrained)
        elif encoder_name == "mobilenet-v2":
            return _get_mobilenet_v2(pretrained=pretrained)
        else:
            raise NotImplementedError(f"{encoder_name} encoder is not implemented")


def _get_resnet_18(pretrained):
    encoder = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained == 'imagenet' else None)
    # encoder.fc = nn.Identity()
    return encoder

def _get_resnet_50(pretrained):
    encoder = models.resnet50(pretrained=models.ResNet50_Weights.IMAGENET1K_V1 if pretrained == 'imagenet' else None)
    # encoder.fc = nn.Identity()
    return encoder

def _get_resnet_101(pretrained):
    encoder = models.resnet101(pretrained=models.ResNet101_Weights.IMAGENET1K_V1 if pretrained == 'imagenet' else None)
    # encoder.fc = nn.Identity()
    return encoder

def _get_efficientnet_b0(pretrained):
    encoder = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained == 'imagenet' else None)
    return encoder

def _get_efficientnet_b1(pretrained):
    encoder = models.efficientnet_b1(weights=models.EfficientNet_B1_Weights.IMAGENET1K_V1 if pretrained == 'imagenet' else None)
    # encoder.classifier = nn.Identity()
    return encoder

def _get_mobilenet_v2(pretrained):
    encoder = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained == 'imagenet' else None)
    # encoder.classifier = nn.Identity()
    return encoder


class BaseEpoch:
    def __init__(self, model, loss_fn, device, optimizer=None):
        self.model = model
        self.loss_fn = loss_fn
        self.device = device
        self.optimizer = optimizer

    def run(self, data_loader, training=False):
        self.model.train() if training else self.model.eval()
        epoch_loss = 0
        total_samples = 0
        all_preds = []
        all_labels = []

        with torch.set_grad_enabled(training):
            with tqdm(total=len(data_loader), desc="Training" if training else "Validating", dynamic_ncols=True) as pbar:
                for inputs, labels in data_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    batch_size = inputs.size(0)
                    if training:
                        self.optimizer.zero_grad()

                    outputs = self.model(inputs)
                    loss = self.loss_fn(outputs, labels)
                    epoch_loss += loss.item() * batch_size
                    total_samples += batch_size
                    if training:
                        loss.backward()
                        self.optimizer.step()

                    preds = torch.argmax(outputs, dim=1)
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())

                    pbar.update(1)
                    pbar.set_postfix({"Loss": f"{loss.item():.4f}"})

        mean_loss = epoch_loss / total_samples if total_samples > 0 else 0

        return mean_loss, all_preds, all_labels



def calculate_standard_metrics(preds, labels, average='macro', idx_to_class=None):
    acc = accuracy_score(labels, preds)
    precision, recall, f1, support = precision_recall_fscore_support(y_true=labels, y_pred=preds, average=average)
    if average is None:
        metrics = {'accuracy': acc}
        for i in range(len(precision)):
            class_key = idx_to_class.get(i, i) if idx_to_class else i
            metrics[f'class_{class_key}_precision'] = precision[i]
            metrics[f'class_{class_key}_recall'] = recall[i]
            metrics[f'class_{class_key}_f1'] = f1[i]
    else:
        metrics = {
            'accuracy': acc,
            'precision': precision,
            'recall': recall,
            'f1': f1,
        }
    return metrics

def calculate_confusion_matrix(labels, preds):
    return confusion_matrix(labels, preds)

def build_label_encoding(labels):
    unique_classes = sorted(set(labels))
    return {cls_name: idx for idx, cls_name in enumerate(unique_classes)}

def set_seed(seed):
    """Set the random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
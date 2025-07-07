from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from torch.optim import Adam, AdamW, SGD
from torch.nn import CrossEntropyLoss
import torch
from tqdm import tqdm
from segmentation_models_pytorch.losses import FocalLoss
import numpy as np

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
        elif loss_name == 'focal':
            if 'weight' in kwargs:
                del kwargs['weight']  # Focal loss does not support weight
            return _get_focal(**kwargs)
        elif loss_name == 'balanced_focal':
            if 'weight' in kwargs:
                del kwargs['weight']  # Focal loss does not support weight
            return _get_focal(**kwargs)
        else:
            raise NotImplementedError(f"{loss_name} loss is not implemented")

def _get_ce(**kwargs):
    return CrossEntropyLoss(**kwargs)

def _get_focal(**kwargs):
    return FocalLoss(mode='multiclass', **kwargs)


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
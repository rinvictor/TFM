from torchvision import models
import torch.nn as nn
import timm
import torch

def get_model(model_name, num_classes, pretrained='imagenet', dropout_rate=0.0):
    cnn_models_families = [
        "resnet", "efficientnet", "mobilenet"
    ]
    transformer_models_families = [
        "vit", "swin"
    ]
    if any(model_name.startswith(prefix) for prefix in cnn_models_families):
        return CNNModelFactory.get_model(model_name=model_name,
                                         num_classes=num_classes,
                                         pretrained=pretrained,
                                         dropout_rate=dropout_rate)
    elif any(model_name.startswith(prefix) for prefix in transformer_models_families):
        if pretrained == 'imagenet':
            pretrained = True
        elif pretrained is None:
            pretrained = False
        else:
            raise NotImplementedError(f"Pretrained '{model_name}' weights not implemented for Transformers")
        return TransformerModelFactory.get_model(model_name=model_name,
                                                 num_classes=num_classes,
                                                 pretrained=pretrained,
                                                 dropout_rate=dropout_rate)
    else:
        raise NotImplementedError(f"Model '{model_name}' not supported")


class CNNModelFactory:
    @staticmethod
    def get_model(model_name, num_classes, pretrained='imagenet', dropout_rate=0.0):
        encoder = EncoderFactory().get_encoder(encoder_name=model_name,
                                               pretrained=pretrained)
        return CustomClassifier(encoder=encoder, num_classes=num_classes, dropout_rate=dropout_rate)

class TransformerModelFactory:
    @staticmethod
    def get_model(model_name, num_classes, pretrained=True, dropout_rate=0.0):
        return timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes, drop_rate=dropout_rate)


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
        elif encoder_name == "efficientnet-b2":
            return _get_efficientnet_b2(pretrained=pretrained)
        elif encoder_name == "efficientnet-b3":
            return _get_efficientnet_b3(pretrained=pretrained)
        elif encoder_name == "efficientnet-b4":
            return _get_efficientnet_b4(pretrained=pretrained)
        elif encoder_name == "mobilenet-v2":
            return _get_mobilenet_v2(pretrained=pretrained)
        else:
            raise NotImplementedError(f"{encoder_name} encoder is not implemented")


def _get_resnet_18(pretrained):
    encoder = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained == 'imagenet' else None)
    return encoder

def _get_resnet_50(pretrained):
    encoder = models.resnet50(pretrained=models.ResNet50_Weights.IMAGENET1K_V1 if pretrained == 'imagenet' else None)
    return encoder

def _get_resnet_101(pretrained):
    encoder = models.resnet101(pretrained=models.ResNet101_Weights.IMAGENET1K_V1 if pretrained == 'imagenet' else None)
    return encoder

def _get_efficientnet_b0(pretrained):
    encoder = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained == 'imagenet' else None)
    return encoder

def _get_efficientnet_b1(pretrained):
    encoder = models.efficientnet_b1(weights=models.EfficientNet_B1_Weights.IMAGENET1K_V1 if pretrained == 'imagenet' else None)
    return encoder

def _get_efficientnet_b2(pretrained):
    encoder = models.efficientnet_b2(weights=models.EfficientNet_B2_Weights.IMAGENET1K_V1 if pretrained == 'imagenet' else None)
    return encoder

def _get_efficientnet_b3(pretrained):
    encoder = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1 if pretrained == 'imagenet' else None)
    return encoder

def _get_efficientnet_b4(pretrained):
    encoder = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.IMAGENET1K_V1 if pretrained == 'imagenet' else None)
    return encoder

def _get_mobilenet_v2(pretrained):
    encoder = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained == 'imagenet' else None)
    return encoder

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
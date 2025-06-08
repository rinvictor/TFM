import logging
import time
from datetime import timedelta
import argparse
import os

import pandas as pd
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchvision import transforms

from classification.dataset import ClassificationDataset
from classification.training_utils import CustomClassifier, EncoderFactory, OptimizerFactory, LossFunctionFactory, \
    BaseEpoch, calculate_standard_metrics, calculate_confusion_matrix, build_label_encoding
from logger_utils import get_logger

OPTIMIZERS = ['adam', 'adamw', 'sgd']
LOSSES = ['ce', 'bce']
METRICS = ['f1', 'accuracy', 'precision', 'recall']

class TrainClassificationModel:
    def __init__(self):
        self.parser = argparse.ArgumentParser(prog='TRAIN_CLASSIFIER',
                                              description='Trains a classification model',
                                              formatter_class=argparse.RawDescriptionHelpFormatter)

        self.add_arguments()
        self.args = self.parser.parse_args()
        self.best_model = None
        self.best_model_metric = float('-inf')
        self.best_model_path = f"checkpoints/checkpoints_{self.args.logger_experiment_name}_{self.args.run_name}_{self.args.encoder_name}"

    def add_argument(self, *args, **kw):
        if isinstance(args, tuple):
            for value in args:
                if '_' in value:
                    raise ValueError('Arguments must not contain _ use - instead.')

        return self.parser.add_argument(*args, **kw)

    def add_arguments(self):
        self.add_argument(
            "--num-classes",
            type=int,
            required=True,
            help="Number of target classes for classification. For binary segmentation, set to 1. "
            "For n classes, set to n+1.",
        )
        self.add_argument(
            "--initial-lr",
            type=float,
            required=False,
            default=1e-3,
            help="Initial learning rate for training",
        )
        self.add_argument(
            "--loss-function",
            type=str,
            required=True,
            choices=LOSSES,
            help=f"Loss function name. Allowed values: {' ,'.join(LOSSES)}",
        )
        self.add_argument(
            "--batch-size",
            type=int,
            required=True,
            help="Batch size for model training",
        )
        self.add_argument(
            "--max-epochs",
            type=int,
            required=False,
            default=100,
            help="Max number of training steps",
        )
        self.add_argument(
            "--encoder-name",
            type=str,
            required=True,
            help="Name of the encoder (feature extractor) of the model to train",
        )

        self.add_argument(
            "--optimizer",
            type=str,
            required=True,
            choices=OPTIMIZERS,
            help=f"Optimizer to use. Allowed values: {' ,'.join(OPTIMIZERS)}",
        )
        self.add_argument(
            "--dataset-path",
            type=str,
            required=True,
            help="Path to the dataset directory",
        )
        self.add_argument(
            "--scheduler-patience",
            type=int,
            required=False,
            default=10,
            help="Max number of epochs without metric improvement before modifying learning rate",
        )

        self.add_argument(
            "--reduction-factor",
            type=float,
            required=False,
            default=0.1,
            help="Reduction factor for learning rate",
        )

        self.add_argument(
            '--pretrained',
            type=str,
            required=False,
            help="Weights to use for the encoder. If 'imagenet', it will use the pretrained weights from ImageNet.",
            default='imagenet',
        )

        self.add_argument(
            "--logger-experiment-name",
            type=str,
            required=True,
            help="Name of the MLflow/Wandb experiment to log the training run",
        )

        self.add_argument(
            "--run-name",
            type=str,
            required=True,
            help="Name of the MLflow run to log the training run",
        )

        self.add_argument(
            "--metric-for-best-model",
            type=str,
            required=False,
            default="f1",
            choices=METRICS,
            help="Metric to select the best model during training",
        )

        self.add_argument(
            "--logger",
            type=str,
            required=False,
            default="mlflow",
            choices=["mlflow", "wandb"],
            help="Logger to use: mlflow or wandb",
        )

    def set_up_experiment(self):
        #todo ver como gestiono lo de la softmax
        try:
            encoder = EncoderFactory().get_encoder(self.args.encoder_name, pretrained=self.args.pretrained) #todo lo de pretained tiene que ser opcional
            model = CustomClassifier(encoder=encoder, num_classes=self.args.num_classes)
        except Exception as error:
            print(f"Something failed creating the model: {error}")
            return False
        try:
            optimizer = OptimizerFactory().get_optimizer(optimizer_name=self.args.optimizer, model_params=model.parameters(),
                                                         initial_lr=self.args.initial_lr)
        except Exception as error:
            print(f"Something failed setting the optimizer up: {error}")
            return False

        try:
            loss_function = LossFunctionFactory().get_loss_function(self.args.loss_function)
        except Exception as error:
            print(f"Something failed getting the loss function: {error}")
            return False

        try:
            scheduler = ReduceLROnPlateau(
                optimizer=optimizer,
                patience=self.args.scheduler_patience,
                factor=self.args.reduction_factor,
            )
        except Exception as error:
            print(f"Something failed setting the scheduler up: {error}")
            return False

        return model, optimizer, loss_function, scheduler

    def set_up_dataset_loader(self):
        df_train = pd.read_csv(os.path.join(self.args.dataset_path, 'train.csv'))
        train_images_with_labels = list(zip(df_train['image_path'], df_train['label']))
        class_to_idx_train = build_label_encoding(df_train['label'])

        df_val = pd.read_csv(os.path.join(self.args.dataset_path, 'val.csv'))
        val_images_with_labels = list(zip(df_val['image_path'], df_val['label']))
        class_to_idx_val = build_label_encoding(df_val['label'])

        df_test = pd.read_csv(os.path.join(self.args.dataset_path, 'test.csv'))
        test_images_with_labels = list(zip(df_test['image_path'], df_test['label']))
        class_to_idx_test = build_label_encoding(df_test['label'])

        if not (class_to_idx_train == class_to_idx_val == class_to_idx_test):
            raise ValueError("Label encoding mismatch between train, val, and test splits!")

        def get_train_transform(image_size=(224, 224)):
            return transforms.Compose([
                transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0), ratio=(0.75, 1.33)),
                transforms.RandomHorizontalFlip(),
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

        train_dataset = ClassificationDataset(
            images_with_labels = train_images_with_labels,
            transform = get_train_transform(),
            label_encoding = class_to_idx_train,
        )
        val_dataset = ClassificationDataset(
            images_with_labels = val_images_with_labels,
            transform = get_val_transform(),
            label_encoding = class_to_idx_val,
        )
        test_dataset = ClassificationDataset(
            images_with_labels = test_images_with_labels,
            transform = get_val_transform(),
            label_encoding = class_to_idx_test,
        )

        train_loader = DataLoader(dataset=train_dataset,
                                  batch_size=self.args.batch_size,
                                  shuffle=True,
                                  num_workers=4,
                                  drop_last=True)
        val_loader = DataLoader(dataset=val_dataset,
                                batch_size=self.args.batch_size,
                                shuffle=False,
                                num_workers=4,
                                drop_last=True)
        test_loader = DataLoader(dataset=test_dataset,
                                 batch_size=self.args.batch_size,
                                 shuffle=False,
                                 num_workers=4,
                                 drop_last=True)

        return train_loader, val_loader, test_loader

    def train_classification_model(self):
        def _display_metrics(metrics):
            print(f"\nEpoch {epoch + 1}/{self.args.max_epochs} | "
                  f"F1 Score: {metrics['f1']:.4f} | "
                  f"Accuracy: {metrics['accuracy']:.4f} | "
                  f"Precision: {metrics['precision']:.4f} | "
                  f"Recall: {metrics['recall']:.4f}")

        def _transform_to_str(transform):
            # Try to get a readable string for the transform pipeline
            if hasattr(transform, '__repr__'):
                return repr(transform)
            return str(transform)

        model, optimizer, loss_function, scheduler = self.set_up_experiment()
        device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        print(f"The model will be trained on {device}.")

        train_loader, val_loader, test_loader = self.set_up_dataset_loader()


        train_epoch = BaseEpoch(model, loss_function, device, optimizer)
        val_epoch = BaseEpoch(model, loss_function, device)
        metric_for_best_model = self.args.metric_for_best_model

        logger_config_params = {
                "num_classes": self.args.num_classes,
                "initial_lr": self.args.initial_lr,
                "loss_function": self.args.loss_function,
                "batch_size": self.args.batch_size,
                "max_epochs": self.args.max_epochs,
                "encoder_name": self.args.encoder_name,
                "optimizer": self.args.optimizer,
                "scheduler_patience": self.args.scheduler_patience,
                "reduction_factor": self.args.reduction_factor,
                "pretrained": self.args.pretrained,
                "metric_for_best_model": metric_for_best_model,
                "train_device": str(device),
                "dataset_path": self.args.dataset_path,
                "train_transforms": _transform_to_str(train_loader.dataset.transform),
                "val_transforms": _transform_to_str(val_loader.dataset.transform),
                "test_transforms": _transform_to_str(test_loader.dataset.transform),
            }
        logger = get_logger(
            logger_name=self.args.logger,
            run_name=self.args.run_name,
            logger_experiment_name=self.args.logger_experiment_name,
            config=logger_config_params  # For WandbLogger, this will be used as config
        )
        if not logger:
            logging.error("Logger initialization failed. Exiting training.")
            raise ValueError("Logger initialization failed.")

        with logger as run_logger:
            run_logger.log_params(logger_config_params)
            assert isinstance(train_loader.dataset, ClassificationDataset)
            unique_labels = set(train_loader.dataset.label_encoding.values())
            average = 'binary' if len(unique_labels) == 2 else 'macro'
            for epoch in range(self.args.max_epochs):
                train_loss, train_preds, train_labels = train_epoch.run(train_loader, training=True)
                train_metrics = calculate_standard_metrics(train_preds, train_labels, average=average)
                _display_metrics(train_metrics)

                #Validation step
                val_loss, val_preds, val_labels = val_epoch.run(val_loader, training=False)
                val_metrics = calculate_standard_metrics(val_preds, val_labels, average=average)
                _display_metrics(val_metrics)

                scheduler.step(val_loss)
                metrics_to_log = {
                    "train_f1": train_metrics["f1"],
                    "val_f1": val_metrics["f1"],
                    "train_accuracy": train_metrics["accuracy"],
                    "val_accuracy": val_metrics["accuracy"],
                    "train_precision": train_metrics["precision"],
                    "val_precision": val_metrics["precision"],
                    "train_recall": train_metrics["recall"],
                    "val_recall": val_metrics["recall"],
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                }
                run_logger.log_metrics(metrics_to_log, step=epoch)

                if val_metrics[metric_for_best_model] > self.best_model_metric:
                    self.best_model_metric = val_metrics[metric_for_best_model]
                    self.best_model = model
                    run_logger.log_model(
                        model=self.best_model,
                        model_name="best_model",
                        model_path=self.best_model_path,
                        epoch=epoch,
                    )

            # Testing best model
            test_epoch = BaseEpoch(self.best_model, loss_function, device)
            test_loss, test_preds, test_labels = test_epoch.run(test_loader, training=False)
            unique_test_labels = set(test_labels)
            test_average = 'binary' if len(unique_test_labels) == 2 else 'macro'
            test_metrics = calculate_standard_metrics(test_preds, test_labels, average=test_average)
            test_metrics_to_log = {
                "test_f1": test_metrics["f1"],
                "test_accuracy": test_metrics["accuracy"],
                "test_precision": test_metrics["precision"],
                "test_recall": test_metrics["recall"],
                "test_loss": test_loss,
            }
            run_logger.log_metrics(test_metrics_to_log)

            print("Test set results:")
            _display_metrics(test_metrics)
            cm = calculate_confusion_matrix(test_labels, test_preds)
            print('Confusion Matrix:')
            print(cm)
            return True


    def run(self) -> bool:
        try:
            init_time = time.time()
            exit_ok = self.train_classification_model()
            finish_time = time.time()
            print(f"Total elapsed time {timedelta(seconds=finish_time - init_time)}")
        except Exception as e:
            print(f"Uncaught exception: {str(e)}")
            exit_ok = False
        finally:
            print(f"Training finished!")
        return exit_ok


def main():
    app = TrainClassificationModel()
    app.run()

if __name__ == '__main__':
    main()
import logging
import time
from datetime import timedelta
import argparse
import os

import pandas as pd
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, WeightedRandomSampler

from transformations import get_train_transform, get_val_transform
from dataset import ClassificationDataset, get_contrastive_loader
from training_utils import CustomClassifier, EncoderFactory, OptimizerFactory, LossFunctionFactory, \
    BaseEpoch, calculate_standard_metrics, calculate_confusion_matrix, build_label_encoding, set_seed
from logger_utils import get_logger
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from general_utils import parse_bracketed_arg
from contrastive import ProjectionHead, ContrastiveTrainer

OPTIMIZERS = ['adam', 'adamw', 'sgd']
LOSSES = ['ce', 'focal']
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
        self.class_weights = None
        self.early_stopping_patience = self.args.early_stopping_patience
        self.epochs_without_improvement = 0

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

        self.add_argument(
            "--max-workers",
            type=int,
            required=False,
            default=12,
            help="Maximum number of workers for data loading. Default is 16.",
        )

        self.add_argument(
            '--use-weighted-sampling',
            action='store_true',
            help='If set, uses weighted sampling to handle class imbalance during training. '
        )

        self.add_argument(
            '--use-class-weights',
            action='store_true',
            help='If set, uses class weights in the loss function to handle class imbalance.'
        )

        self.add_argument(
            '--seed',
            type=int,
            required=False,
            default=42,
            help="Random seed for reproducibility.",
        )

        self.add_argument(
            '--dropout-rate',
            type=float,
            required=False,
            default=0,
            help="Dropout rate for the model. Default is 0 (no dropout).",
        )

        self.add_argument(
            '--early-stopping-patience',
            type=int,
            required=False,
            default=None,
            help='Number of epochs to wait for improvement before stopping.'
        )

        self.add_argument(
            '--dataset-downsampling-fraction',
            type=float,
            required=False,
            default=None,
            help='If set, the train dataset will be downsampled to this fraction. '
                 'Useful for large training datasets, to speed up training.'
        )

        self.add_argument(
            '--contrastive-pretrain',
            action='store_true',
            help='Pretains the model using contrastive learning before fine-tuning for classification.')

    def contrastive_pretrain(self, encoder, dataloader, device, epochs=30):
        import torch.optim as optim
        encoder.to(device)
        in_dim=encoder.fc.in_features
        encoder.fc = torch.nn.Identity()
        projection_head = ProjectionHead(in_dim=in_dim).to(device) # todo harcodeado para resnet
        optimizer = optim.Adam(list(encoder.parameters()) + list(projection_head.parameters()), lr=1e-4) #todo pueden ser otros valores?? otro optimizador?

        print("Starting contrastive pretraining...")
        trainer = ContrastiveTrainer(encoder, dataloader, projection_head, optimizer, device)
        trainer.train(epochs=epochs)
        torch.save(encoder.state_dict(), "encoder_contrastive.pth")

    def set_up_experiment(self, train_dataset):
        try:
            encoder = EncoderFactory().get_encoder(self.args.encoder_name, pretrained=self.args.pretrained) #todo lo de pretained tiene que ser opcional
        except Exception as error:
            print(f"Something failed creating the encoder: {error}")
            return False

        if self.args.contrastive_pretrain:
            device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
            contrastive_loader = get_contrastive_loader(train_dataset, batch_size=64, num_workers=self.args.max_workers, augment=True)
            self.contrastive_pretrain(encoder, contrastive_loader, device, epochs=60) #todo cosas harcodeadas
        #Carga pesos preentrenados si existen
        if os.path.exists("encoder_contrastive.pth"):
            encoder.load_state_dict(torch.load("encoder_contrastive.pth"))

        try:
            model = CustomClassifier(encoder=encoder, num_classes=self.args.num_classes,
                                     dropout_rate=self.args.dropout_rate)
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
            loss_name, loss_config = parse_bracketed_arg(self.args.loss_function)
            print(f"Using loss function: {loss_name} with config: {loss_config}")
            loss_function = LossFunctionFactory().get_loss_function(loss_name, weight=self.class_weights, **loss_config)
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
        if self.args.dataset_downsampling_fraction is not None:
            original_size = len(df_train)
            df_train_downsampled = df_train.groupby('label', group_keys=False).apply(
                lambda x: x.sample(frac=self.args.dataset_downsampling_fraction,
                                   random_state=self.args.seed),
                include_groups=True
            )
            df_train = df_train_downsampled
            new_size = len(df_train)
            print(f"Downsampling the training dataset to "
                  f"{self.args.dataset_downsampling_fraction * 100}% of its original size. "
                  f"New size: {new_size} samples (from {original_size} samples).")

        if self.args.use_class_weights:
            print("Calculating class weights for the loss function.")
            train_labels = df_train['label'].to_numpy()
            classes = np.unique(train_labels)
            weights = compute_class_weight(
                class_weight='balanced',
                classes=classes,
                y=train_labels
            )
            self.class_weights = torch.tensor(weights, dtype=torch.float32)
            print(f"Calculated class weights: {self.class_weights}")
        if self.args.use_weighted_sampling:
            print("Using WeightedRandomSampler for class imbalance handling.")
            # Calculate class weights
            class_counts = df_train['label'].value_counts()
            class_weights_dict = {
                label: 1.0 / count if count > 0 else 0.0
                for label, count in class_counts.items()
            }
            print("Inverse class weights:", class_weights_dict)
            labels_for_weight_lookup = df_train['label'].tolist()
            sample_weights = torch.tensor([class_weights_dict[label] for label in labels_for_weight_lookup], dtype=torch.double)
            sampler = WeightedRandomSampler(
                weights= sample_weights,
                num_samples=len(sample_weights),
                replacement=True
            )
        else:
            sampler = None

        train_images_with_labels = list(zip(df_train['image_path'], df_train['label']))
        class_to_idx_train = build_label_encoding(df_train['label'])

        df_val = pd.read_csv(os.path.join(self.args.dataset_path, 'val.csv'))
        val_images_with_labels = list(zip(df_val['image_path'], df_val['label']))
        class_to_idx_val = build_label_encoding(df_val['label'])

        df_test = pd.read_csv(os.path.join(self.args.dataset_path, 'test.csv'))
        test_images_with_labels = list(zip(df_test['image_path'], df_test['label']))
        class_to_idx_test = build_label_encoding(df_test['label'])

        print(f"\nTrain set - Total: {len(df_train)}")
        print(df_train['label'].value_counts())

        print(f"\nVal set - Total: {len(df_val)}")
        print(df_val['label'].value_counts())

        print(f"\nTest set - Total: {len(df_test)}")
        print(df_test['label'].value_counts())

        if not (class_to_idx_train == class_to_idx_val == class_to_idx_test):
            raise ValueError("Label encoding mismatch between train, val, and test splits!")

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
                                  shuffle=True if not self.args.use_weighted_sampling else False,
                                  num_workers=self.args.max_workers,
                                  drop_last=True,
                                  sampler=sampler)
        val_loader = DataLoader(dataset=val_dataset,
                                batch_size=self.args.batch_size,
                                shuffle=False,
                                num_workers=self.args.max_workers,
                                drop_last=True)
        test_loader = DataLoader(dataset=test_dataset,
                                 batch_size=self.args.batch_size,
                                 shuffle=False,
                                 num_workers=self.args.max_workers,
                                 drop_last=True)

        return train_loader, val_loader, test_loader, train_dataset

    def train_classification_model(self):
        def _display_metrics(metrics, split: str = "train"):
            print(f"\n📊 {split.upper()} GLOBAL METRICS\n"
                  f"Epoch {epoch + 1}/{self.args.max_epochs} | "
                  f"F1 Score: {metrics['f1']:.4f} | "
                  f"Accuracy: {metrics['accuracy']:.4f} | "
                  f"Precision: {metrics['precision']:.4f} | "
                  f"Recall: {metrics['recall']:.4f}")

        def _display_metrics_per_class(metrics: dict, split: str = "train"):
            print(f"\n📊 {split.upper()} METRICS PER CLASS")
            class_metrics = {}

            for k, v in metrics.items():
                if k == "accuracy":
                    continue
                if k.startswith("class_"):
                    parts = k.split("_", 2)
                    class_name = parts[1]
                    metric_name = parts[2]
                    class_metrics.setdefault(class_name, {})[metric_name] = v

            for class_name in sorted(class_metrics.keys()):
                m = class_metrics[class_name]
                print(f"  🏷️  Class '{class_name}': "
                      f"Precision: {m.get('precision', 0):.4f} | "
                      f"Recall: {m.get('recall', 0):.4f} | "
                      f"F1: {m.get('f1', 0):.4f}")

        def _transform_to_str(transform):
            # Try to get a readable string for the transform pipeline
            if hasattr(transform, '__repr__'):
                return repr(transform)
            return str(transform)

        # For reproducibility
        set_seed(self.args.seed)
        train_loader, val_loader, test_loader, train_dataset = self.set_up_dataset_loader()
        model, optimizer, loss_function, scheduler = self.set_up_experiment(train_dataset)
        device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        print(f"The model will be trained on {device}.")




        train_epoch = BaseEpoch(model, loss_function, device, optimizer)
        val_epoch = BaseEpoch(model, loss_function, device)
        metric_for_best_model = self.args.metric_for_best_model

        logger_config_params = {
                "num_classes": self.args.num_classes,
                "initial_lr": self.args.initial_lr,
                "loss_function": self.args.loss_function,
                "loss_config": str(parse_bracketed_arg(self.args.loss_function)[1]),
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
                "use_weighted_sampling": self.args.use_weighted_sampling,
                "dataset_downsampling_fraction": self.args.dataset_downsampling_fraction,
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
            average = 'macro' # The objetive is to calculate the metrics globally, so we use macro
            idx_to_class = {v: k for k, v in train_loader.dataset.label_encoding.items()}
            for epoch in range(self.args.max_epochs):
                self.best_model = model
                continue
                train_loss, train_preds, train_labels = train_epoch.run(train_loader, training=True)
                # Global metrics calculation
                train_metrics = calculate_standard_metrics(train_preds, train_labels, average=average)
                train_metrics_per_class = calculate_standard_metrics(train_preds, train_labels,
                                                                     average=None, idx_to_class=idx_to_class)
                _display_metrics(train_metrics, split="train")
                _display_metrics_per_class(train_metrics_per_class, split="train")

                #Validation step
                val_loss, val_preds, val_labels = val_epoch.run(val_loader, training=False)
                val_metrics = calculate_standard_metrics(val_preds, val_labels, average=average)
                val_metrics_per_class = calculate_standard_metrics(val_preds, val_labels,
                                                                   average=None, idx_to_class=idx_to_class)
                _display_metrics(val_metrics, split="val")
                _display_metrics_per_class(val_metrics_per_class, split="val")

                scheduler.step(val_loss)
                current_learning_rate = optimizer.param_groups[0]['lr']
                print("Current learning rate:", current_learning_rate)

                metrics_to_log = {
                    "train_f1_global": train_metrics["f1"],
                    "val_f1_global": val_metrics["f1"],
                    "train_accuracy_global": train_metrics["accuracy"],
                    "val_accuracy_global": val_metrics["accuracy"],
                    "train_precision_global": train_metrics["precision"],
                    "val_precision_global": val_metrics["precision"],
                    "train_recall_global": train_metrics["recall"],
                    "val_recall_global": val_metrics["recall"],
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "learning_rate": current_learning_rate,
                }
                metrics_to_log.update({f"train_{k}": v for k, v in train_metrics_per_class.items() if k != "accuracy"})
                metrics_to_log.update({f"val_{k}": v for k, v in val_metrics_per_class.items() if k != "accuracy"})
                run_logger.log_metrics(metrics_to_log, step=epoch)

                if val_metrics[metric_for_best_model] > self.best_model_metric:
                    self.best_model_metric = val_metrics[metric_for_best_model]
                    self.epochs_without_improvement = 0
                    self.best_model = model
                    run_logger.log_model(
                        model=self.best_model,
                        model_name="best_model",
                        model_path=self.best_model_path,
                        epoch=epoch,
                    )
                else:
                    self.epochs_without_improvement += 1

                if self.early_stopping_patience is not None and self.epochs_without_improvement >= self.early_stopping_patience:
                    print(f"Early stopping triggered after {self.epochs_without_improvement} epochs without improvement.")
                    run_logger.log_metrics({"early_stopping_triggered": True}, step=epoch)
                    break

            # Testing best model
            test_epoch = BaseEpoch(self.best_model, loss_function, device)
            test_loss, test_preds, test_labels = test_epoch.run(test_loader, training=False)

            test_metrics = calculate_standard_metrics(test_preds, test_labels, average='macro')
            test_metrics_per_class = calculate_standard_metrics(test_preds, test_labels,
                                                                average=None, idx_to_class=idx_to_class)
            test_metrics_to_log = {
                "test_f1_global": test_metrics["f1"],
                "test_accuracy_global": test_metrics["accuracy"],
                "test_precision_global": test_metrics["precision"],
                "test_recall_global": test_metrics["recall"],
                "test_loss_global": test_loss,
            }
            test_metrics_to_log.update({f"test_{k}": v for k, v in test_metrics_per_class.items() if k != "accuracy"})
            run_logger.log_metrics(test_metrics_to_log)

            print("Test set results:")
            _display_metrics(test_metrics, split="test")
            _display_metrics_per_class(test_metrics_per_class, split="test")

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
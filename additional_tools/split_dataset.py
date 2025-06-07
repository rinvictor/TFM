import argparse
import csv
import os
import shutil

import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit


class ClassificationDatasetSplitter:
    def __init__(self):
        self.parser = argparse.ArgumentParser(prog='SPLIT_DATASET',
                                              description='Splits the dataset into training, validation and test sets',
                                              formatter_class=argparse.RawDescriptionHelpFormatter)

        self.add_argument(
            "--dataset-to-split-path",
            type=str,
            required=True,
            help="Path to the dataset we want to split",
        )

        self.add_argument(
            "--classes-csv-path",
            type=str,
            required=True,
            help="Path to the classes csv file",
        )

        self.add_argument(
            "--output-dataset-path",
            type=str,
            required=True,
            help="Output path for the split dataset",
        )

        self.parser.add_argument(
            "--val-split",
            type=float,
            required=False,
            default=0.2,
            help="Fraction of the dataset to use for validation (e.g., 0.1 = 10%)"
        )
        self.parser.add_argument(
            "--test-split",
            type=float,
            required=False,
            default=0.1,
            help="Fraction of the dataset to use for testing (e.g., 0.1 = 10%)"
        )
        self.parser.add_argument(
            "--include-class-subdir",
            action="store_true",
            help="If set, images will be saved in subdirectories named after their class inside each split folder, eg: outpout-path/train/class"
        )
        self.args = self.parser.parse_args()

    def add_argument(self, *args, **kw):
        if isinstance(args, tuple):
            for value in args:
                if '_' in value:
                    raise ValueError('Arguments must not contain _ use - instead.')

        return self.parser.add_argument(*args, **kw)

    @staticmethod
    def copy_files(images, labels, output_dataset_path, include_class_subdir=False):
        for img_path, label in zip(images, labels):
            if include_class_subdir:
                dst_dir = os.path.join(output_dataset_path, str(label).lower().replace(" ", "_") )
            else:
                dst_dir = os.path.join(output_dataset_path)
            os.makedirs(dst_dir, exist_ok=True)
            shutil.copy(str(img_path), str(dst_dir))

    @staticmethod
    def save_split_csv(images, labels,output_dataset_path, split_name):
        csv_out_path = os.path.join(output_dataset_path, f"{split_name}.csv")
        with open(csv_out_path, mode="w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["image_path", "label"])
            for img_path, label in zip(images, labels):
                writer.writerow([img_path, str(label).lower().replace(" ", "_")])

    def stratified_split(self):
        input_dataset_path = self.args.dataset_to_split_path
        output_dataset_path = self.args.output_dataset_path
        val_size = self.args.val_split
        test_size = self.args.test_split
        csv_path = self.args.classes_csv_path
        include_class_subdir = self.args.include_class_subdir

        if os.path.exists(input_dataset_path) and os.path.exists(csv_path):
            image_classes_df = pd.read_csv(csv_path)
            disk_images = set(os.listdir(input_dataset_path))
            csv_images = set(image_classes_df['image_name'])
            extra_images = disk_images - csv_images
            if extra_images:
                raise Exception(f'{len(extra_images)} images in the folder are not listed in the CSV.')

            images = image_classes_df['image_name'].apply(lambda x: os.path.join(input_dataset_path, x)).tolist()
            labels = image_classes_df['class'].tolist()

            train_images, val_images, test_images = [], [], []
            train_labels, val_labels, test_labels = [], [], []
            train_val_images, train_val_labels = [], []
            # train + val vs test split
            split_1 = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
            for train_val_idx, test_idx in split_1.split(images, labels):
                train_val_images = [images[i] for i in train_val_idx]
                train_val_labels = [labels[i] for i in train_val_idx]
                test_images = [images[i] for i in test_idx]
                test_labels = [labels[i] for i in test_idx]

            # train vs val split
            val_ratio = val_size / (1 - test_size)
            split_2 = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio, random_state=42)
            for train_idx, val_idx in split_2.split(train_val_images, train_val_labels):
                train_images = [train_val_images[i] for i in train_idx]
                train_labels = [train_val_labels[i] for i in train_idx]
                val_images = [train_val_images[i] for i in val_idx]
                val_labels = [train_val_labels[i] for i in val_idx]

            if not train_images or not train_labels or not val_images or not val_labels or not test_images or not test_labels:
                raise Exception('Something went wrong creating splitting the dataset.')
            try:
                self.copy_files(train_images, train_labels, os.path.join(output_dataset_path, 'train'), include_class_subdir)
                self.save_split_csv(train_images, train_labels, output_dataset_path,"train")

                self.copy_files(val_images, val_labels, os.path.join(output_dataset_path, 'val'), include_class_subdir)
                self.save_split_csv(val_images, val_labels, output_dataset_path,"val")

                self.copy_files(test_images, test_labels, os.path.join(output_dataset_path, 'test'), include_class_subdir)
                self.save_split_csv(test_images, test_labels, output_dataset_path, "test")
            except Exception as e:
                raise e


        else:
            raise FileNotFoundError("Input dataset and classes csv file must exist")



def main():
    splitter = ClassificationDatasetSplitter()
    splitter.stratified_split()


if __name__ == "__main__":
    main()
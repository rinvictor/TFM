import pandas as pd
from datasets import load_dataset
import os
import csv
import kagglehub
from kagglehub import KaggleDatasetAdapter
import shutil
from PIL import Image

def save_raw_images_hugging_face(output_dir, dataset_name):
    os.makedirs(output_dir, exist_ok=True)
    dataset = load_dataset(dataset_name)
    csv_filename = os.path.join(output_dir, 'raw_data.csv')
    with open(csv_filename, mode='w', newline='') as csv_file:
        fieldnames = ['image_name', 'text']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for i, data in enumerate(dataset['train']):
            img = data['image']
            text = data['text']
            path_to_save = os.path.join(output_dir, 'raw_images')
            os.makedirs(path_to_save, exist_ok=True)
            image_file_name = f"image_{i + 1}.jpg"
            img.save(os.path.join(path_to_save, image_file_name))
            writer.writerow({'image_name': image_file_name, 'text': text})


def save_raw_images_kaggle(output_dir, dataset_name):
    os.makedirs(output_dir, exist_ok=True)
    dataset_path = kagglehub.dataset_download(dataset_name)
    shutil.copytree(dataset_path, output_dir, dirs_exist_ok=True)

def build_classification_dataset_from_segmentation(input_directory, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    subdir_classes = [d for d in os.listdir(input_directory)
                   if os.path.isdir(os.path.join(input_directory, d))]

    csv_filename = os.path.join(output_dir, 'raw_data.csv')
    with open(csv_filename, mode='w', newline='') as csv_file:
        fieldnames = ['image_name', 'text']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for class_name in subdir_classes:
            img_dir = os.path.join(os.path.join(input_directory, class_name), 'images')
            os.makedirs(output_dir, exist_ok=True)
            for image_file in os.listdir(img_dir):
                img_class = class_name
                path_to_save = os.path.join(output_dir, 'raw_images')
                os.makedirs(path_to_save, exist_ok=True)
                img = Image.open(os.path.join(img_dir, image_file))
                img.save(os.path.join(path_to_save, image_file.replace(' ', '_')))
                writer.writerow({'image_name': image_file.replace(' ', '_'), 'text': img_class})


def build_brain_tumor(input_directory, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    subdir_classes = [d for d in os.listdir(input_directory)
                   if os.path.isdir(os.path.join(input_directory, d))]

    csv_filename = os.path.join(output_dir, 'raw_data.csv')
    with open(csv_filename, mode='w', newline='') as csv_file:
        fieldnames = ['image_name', 'text']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for class_name in subdir_classes:
            img_dir = os.path.join(os.path.join(input_directory, class_name))
            for image_file in os.listdir(img_dir):
                if '_flipped_horizontal' in image_file:
                    img_class = class_name

                    path_to_save = os.path.join(output_dir, 'raw_images')
                    os.makedirs(path_to_save, exist_ok=True)
                    img = Image.open(os.path.join(img_dir, image_file))
                    img.save(os.path.join(path_to_save, image_file))
                    writer.writerow({'image_name': image_file, 'text': img_class})

def build_isic(input_directory, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    csv_filename = os.path.join(output_dir, 'raw_data.csv')
    metadata_df = pd.read_csv(os.path.join(input_directory, 'metadata.csv'))
    with open(csv_filename, mode='w', newline='') as csv_file:
        fieldnames = ['image_name', 'text']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for image_file in os.listdir(input_directory):
            if image_file.endswith('.jpg'):
                img_class = metadata_df[metadata_df['isic_id']
                                        == image_file.replace('.jpg', '')
                ]['benign_malignant'].iloc[0]
                path_to_save = os.path.join(output_dir, 'raw_images')
                os.makedirs(path_to_save, exist_ok=True)
                img = Image.open(os.path.join(input_directory, image_file))
                img.save(os.path.join(path_to_save, image_file))
                writer.writerow({'image_name': image_file, 'text': img_class})





if __name__ == '__main__':
    # save_raw_images_hugging_face('../data/oct', 'aditya11997/retinal_oct_analysis2')
    # save_raw_images_kaggle('../data/brain-tumor-mri-image-dataset',
    #                       'Hghdhygf/brain-tumor-mri-image-dataset')
    # build_classification_dataset_from_segmentation('../data/covid-19-radiography-raw/COVID-19_Radiography_Dataset',
    #                                                '../data/covid-19-radiography-raw-classification')
    # build_brain_tumor('../data/brain-tumor-mri-image-dataset/80 of training image with DAUG (5k Each Total 20k)/80% of training image with DAUG (5k Each, Total 20k)',
    #                   '../data/brain-tumor-mri-image-dataset-prepared')
    build_isic('../data/ISIC-images(1)', '../data/isic/prepared')
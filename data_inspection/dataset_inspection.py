import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import os

from PIL import Image


def text_inspection(data_csv):
    df = pd.read_csv(data_csv)
    text_distribution = df['text'].value_counts()
    return text_distribution

def classes_inspection(df_text_distribution, data_csv):
    # todo habria que investigar si las clases tienen relacion entre si, pero a priori son diferentes
    def assign_manual_classes_oct(description):
        description = description.lower()
        if any(word in description for word in ['healthy', 'normal']):
            return 'Healthy Retina'
        elif 'choroidal neovascularization' in description:
            return 'Choroidal Neovascularization'
        elif 'vitreous detachment' in description:
            return 'Vitreous Detachment'
        elif 'cystoid macular edema' in description:
            return 'Cystoid Macular Edema'
        elif 'epiretinal membrane' in description:
            return 'Epiretinal Membrane'
        elif any(word in description for word in ['diabetic', 'diabetes']):
            return 'Diabetic Retinopathy'
        elif any(word in description for word in ['macular degeneration', 'degeneration of the macula']):
            return 'Macular Degeneration'
        elif 'glaucoma' in description:
            return 'Glaucoma'
        elif 'retinal detachment' in description:
            return 'Retinal Detachment'
        else:
            return 'Other'

    def assign_manual_classes_covid(description):
        description = description.lower()
        if 'normal' in description:
            return 'Normal'
        elif 'covid' in description:
            return 'Covid'
        elif 'lung' in description:
            return 'Lung Opacity'
        elif 'pneumonia' in description:
            return 'Viral Pneumonia'
        else:
            return 'Other'

    def assign_manual_classes_brain_tumors(description):
        description = description.lower()
        if 'glioma' in description:
            return 'Glioma'
        elif 'meningioma' in description:
            return 'Meningioma'
        elif 'notumor' in description:
            return 'Notumor'
        elif 'pituitary' in description:
            return 'Pituitary'
        else:
            return 'Other'

    def assign_manual_classes_isic(description):
        description = description.lower()
        if 'benign' in description:
            return 'Benign'
        elif 'malignant' in description:
            return 'Malignant'
        else:
            return 'Other'

    df_classes_assignation = pd.DataFrame({
        'description': df_text_distribution.index,
        'class': [assign_manual_classes_isic(desc) for desc in df_text_distribution.index],
    })
    df_data = pd.read_csv(data_csv)
    df_merged = df_data.merge(df_classes_assignation, left_on='text', right_on='description', how='left')
    class_distribution = df_merged['class'].value_counts().reset_index()
    class_distribution.columns = ['Class', 'Count']

    df_image_class = df_merged[['image_name', 'description', 'class']]

    return df_classes_assignation, class_distribution, df_image_class


def clean_and_save_data(data_csv, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    text_distribution = text_inspection(data_csv)
    text_distribution.to_csv(os.path.join(output_dir, 'text_distribution.csv'))

    df_classes_assignation, class_distribution, df_image_class = classes_inspection(text_distribution, data_csv)
    df_classes_assignation.to_csv(os.path.join(output_dir, 'classes_text_assignation.csv'))
    class_distribution.to_csv(os.path.join(output_dir, 'general_classes_distribution.csv'))
    df_image_class.to_csv(os.path.join(output_dir, 'image_class.csv'))

    plt.figure(figsize=(10, 6))
    ax = sns.barplot(data=class_distribution, x='Count', y='Class', palette='viridis', hue='Class', legend=False)
    for i in ax.containers:
        ax.bar_label(i, fmt='%d', label_type='edge', fontsize=10, padding=3)

    plt.title('Classes distribution', fontsize=14)
    plt.xlabel('Number of images')
    plt.ylabel('Classes')
    plt.tight_layout()

    plot_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plot_dir, exist_ok=True)
    plt.savefig(os.path.join(plot_dir, 'class_distribution.png'))

def image_size_inspection(dataset_path, output_dir):
    image_sizes = []
    for image_file in os.listdir(dataset_path):
        image_path = os.path.join(dataset_path, image_file)
        img = Image.open(image_path)
        image_sizes.append(img.size)
    widths, heights = zip(*image_sizes)
    size_data = {
        'Width': widths,
        'Height': heights
    }
    df = pd.DataFrame(size_data)
    plt.figure(figsize=(10, 6))

    viridis_colors = sns.color_palette("viridis", n_colors=2)

    ax = sns.histplot(df['Width'], bins=20, color=viridis_colors[0], kde=False, label='Width', alpha=0.7)
    sns.histplot(df['Height'], bins=20, color=viridis_colors[-1], kde=False, label='Height', alpha=0.7)
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(f'{height:.0f}', (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='center', fontsize=10, color='black', xytext=(0, 5),
                        textcoords='offset points')

    plt.title('Histogram of Image Sizes', fontsize=14)
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.legend(title='Dimension', loc='upper right')
    plt.tight_layout()

    plot_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plot_dir, exist_ok=True)
    plt.savefig(os.path.join(plot_dir, 'size_histogram.png'))

def main():
    input_dir = '../data/isic/prepared'
    clean_and_save_data(data_csv = os.path.join(input_dir, 'raw_data.csv'), output_dir = os.path.join(input_dir, 'metadata'))
    image_size_inspection(dataset_path = os.path.join(input_dir, 'raw_images'), output_dir = os.path.join(input_dir, 'metadata'))


if __name__ == '__main__':
    main()
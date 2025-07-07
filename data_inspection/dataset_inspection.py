import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import os

from PIL import Image
from tqdm import tqdm


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


def extract_image_sizes(df, path_column='image_path'):
    """
    Añade columnas 'width' y 'height' al DataFrame con las dimensiones de cada imagen.
    """
    widths, heights = [], []

    for path in tqdm(df[path_column], desc='Extrayendo tamaños'):
        try:
            with Image.open(path) as img:
                width, height = img.size
                widths.append(width)
                heights.append(height)
        except Exception as e:
            print(f"Error al procesar {path}: {e}")
            widths.append(None)
            heights.append(None)

    df['width'] = widths
    df['height'] = heights
    return df

def summarize_sizes_by_class(df, label_column='ylabel'):
    """
    Agrupa por clase y muestra estadísticas descriptivas de ancho y alto,
    incluyendo el valor más común (moda) de cada dimensión.
    """
    import scipy.stats as stats

    # Agrupamos las estadísticas básicas
    summary = df.groupby(label_column)[['width', 'height']].agg(['mean', 'std', 'min', 'max']).round(2)

    # Calculamos la moda (el valor más común) por clase
    width_mode = df.groupby(label_column)['width'].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else None)
    height_mode = df.groupby(label_column)['height'].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else None)

    # Añadimos las modas al DataFrame de resumen
    summary[('width', 'mode')] = width_mode
    summary[('height', 'mode')] = height_mode

    # Reordenamos columnas para mayor claridad
    summary = summary.reindex(columns=[
        ('width', 'mean'), ('width', 'std'), ('width', 'min'), ('width', 'max'), ('width', 'mode'),
        ('height', 'mean'), ('height', 'std'), ('height', 'min'), ('height', 'max'), ('height', 'mode'),
    ])

    # Ordenar columnas jerárquicas
    summary.columns = pd.MultiIndex.from_tuples(summary.columns)

    return summary.round(2)


def plot_image_sizes_by_class(df, label_column='label'):
    """
    Muestra boxplots del ancho y alto de las imágenes por clase.
    """
    plt.figure(figsize=(14, 6))

    # Ancho
    plt.subplot(1, 2, 1)
    sns.boxplot(x=label_column, y='width', data=df)
    plt.title('Distribución del ancho por clase')
    plt.xticks(rotation=45)
    plt.grid(True)

    # Alto
    plt.subplot(1, 2, 2)
    sns.boxplot(x=label_column, y='height', data=df)
    plt.title('Distribución del alto por clase')
    plt.xticks(rotation=45)
    plt.grid(True)

    plt.tight_layout()
    plt.show()

def plot_histograms_by_class(df, label_column='label'):
    """
    Muestra histogramas facetados de width y height por clase.
    """
    g = sns.FacetGrid(df, col=label_column, col_wrap=3, sharex=False, sharey=False, height=4)
    g.map(sns.histplot, 'width', kde=False, bins=20, color='skyblue')
    g.fig.suptitle('Histograma de ancho por clase', y=1.05)
    plt.show()

    g = sns.FacetGrid(df, col=label_column, col_wrap=3, sharex=False, sharey=False, height=4)
    g.map(sns.histplot, 'height', kde=False, bins=20, color='salmon')
    g.fig.suptitle('Histograma de alto por clase', y=1.05)
    plt.show()

def plot_width_histogram_with_mode(df, label_column='label'):
    modes = df.groupby(label_column)['width'].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else None)

    g = sns.FacetGrid(df, col=label_column, col_wrap=3, sharex=False, height=4)
    g.map(sns.histplot, 'width', bins=20, color='skyblue')

    for ax, (label, mode_val) in zip(g.axes.flat, modes.items()):
        ax.axvline(mode_val, color='red', linestyle='--', label=f'Moda: {mode_val}')
        ax.legend()

    g.fig.suptitle('Histograma de ancho por clase (con moda)', y=1.05)
    plt.show()

def main():
    # input_dir = '../data/isic/prepared'
    # clean_and_save_data(data_csv = os.path.join(input_dir, 'raw_data.csv'), output_dir = os.path.join(input_dir, 'metadata'))
    # image_size_inspection(dataset_path = os.path.join(input_dir, 'raw_images'), output_dir = os.path.join(input_dir, 'metadata'))
    train_original_csv_path = '/nas/data/isic/original/train.csv'
    df = pd.read_csv(train_original_csv_path)
    df = extract_image_sizes(df, path_column='image_path')
    stats_by_class = summarize_sizes_by_class(df, label_column='label')
    print(stats_by_class)
    plot_image_sizes_by_class(df)
    plot_histograms_by_class(df)
    plot_width_histogram_with_mode(df)

if __name__ == '__main__':
    main()
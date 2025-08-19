import os
from PIL import Image
import numpy as np
from tqdm import tqdm
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

def process_image(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image not found: {path}")
    img = Image.open(path).convert("RGB")
    img = np.array(img).astype(np.float32) / 255.0
    n_pixels = img.shape[0] * img.shape[1]
    channel_sum = img.sum(axis=(0, 1))
    channel_sum_squared = (img ** 2).sum(axis=(0, 1))
    return n_pixels, channel_sum, channel_sum_squared

def compute_mean_std_from_csv_parallel(csv_path, image_column="image_path"):
    df = pd.read_csv(csv_path)
    if image_column not in df.columns:
        raise ValueError(f"Column '{image_column}' not found in the CSV.")
    image_paths = df[image_column].tolist()
    if not image_paths:
        raise ValueError("No image paths found in CSV.")

    n_pixels_total = 0
    channel_sum_total = np.zeros(3)
    channel_sum_squared_total = np.zeros(3)
    num_workers = min(32, os.cpu_count() * 5)
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        results = list(tqdm(executor.map(process_image, image_paths), total=len(image_paths),
                            desc=f"Calculating mean and standard deviation in parallel using {num_workers} threads..."))

    for n_pixels, channel_sum, channel_sum_squared in results:
        n_pixels_total += n_pixels
        channel_sum_total += channel_sum
        channel_sum_squared_total += channel_sum_squared

    mean = channel_sum_total / n_pixels_total
    std = np.sqrt(channel_sum_squared_total / n_pixels_total - mean ** 2)
    return mean, std

def save_mean_std_to_file(csv_path, mean, std):
    dir_path = os.path.dirname(csv_path)
    output_path = os.path.join(dir_path, "mean_std.txt")
    with open(output_path, "w") as f:
        f.write(f"{mean.tolist()}\n")
        f.write(f"{std.tolist()}\n")
    print(f"Results saved in: {output_path}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-csv", type=str,
        required=True,
        help="Directory containing the csv where images are stored."
    )
    args = parser.parse_args()

    mean, std = compute_mean_std_from_csv_parallel(args.train_csv)
    print(f"Mean per channel: {mean}")
    print(f"Standard deviation per channel: {std}")
    save_mean_std_to_file(args.train_csv, mean, std)
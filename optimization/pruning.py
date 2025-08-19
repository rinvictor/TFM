import argparse
import os
import time
import random
import copy
import psutil
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm
import wandb

def get_model(model_name: str, num_classes: int, pretrained: bool = True, dropout_rate: float = 0.0):
    return timm.create_model(
        model_name,
        pretrained=pretrained,
        num_classes=num_classes,
        drop_rate=dropout_rate
    )


class ImageDataset(Dataset):
    def __init__(self, csv_path, transform=None):
        self.df = pd.read_csv(csv_path)
        self.transform = transform
        self.label_map = {'benign': 0, 'malignant': 1}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.iloc[idx]['image_path']
        label_str = self.df.iloc[idx]['label']
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = self.label_map[label_str]
        return image, label


def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def evaluate_detailed(model, loader, device="cpu"):
    model.eval()
    y_true, y_pred = [], []
    with torch.inference_mode():
        for x, y in loader:
            if isinstance(y, tuple):
                y = y[0]
            x = x.to(device, non_blocking=True)
            out = model(x)
            preds = out.argmax(dim=1)
            y_true.extend(y.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    acc = accuracy_score(y_true, y_pred)
    f1_weighted = f1_score(y_true, y_pred, average="weighted")
    f1_macro = f1_score(y_true, y_pred, average="macro")
    f1_per_class = f1_score(y_true, y_pred, average=None)

    f1_benign = float(f1_per_class[0]) if len(f1_per_class) > 0 else 0.0
    f1_malignant = float(f1_per_class[1]) if len(f1_per_class) > 1 else 0.0

    class_counts = Counter(y_true)
    majority_class = max(class_counts, key=class_counts.get) if len(class_counts) > 0 else 0
    f1_majority = f1_benign if majority_class == 0 else f1_malignant

    print(f"F1 B: {f1_benign:.4f}")
    print(f"F1 M: {f1_malignant:.4f} ⚠️  (CRITICAL)")
    print(f"F1 M: {f1_macro:.4f}")
    print(f"F1 B: {f1_majority:.4f} (Class {majority_class}: {class_counts.get(majority_class, 0)} samples)")

    return acc, f1_weighted, f1_macro, f1_benign, f1_malignant, f1_majority


def calculate_real_compression(original_model, pruned_model):
    orig_params = sum(p.numel() for p in original_model.parameters())

    active_params = 0
    for module in pruned_model.modules():
        if hasattr(module, 'weight_mask'):
            active_params += int(module.weight_mask.sum().item())
        elif hasattr(module, 'weight'):
            active_params += module.weight.numel()

    compression_ratio = (1 - active_params / orig_params) * 100
    return compression_ratio


def get_sparsity_info(model):
    total_params = 0
    pruned_params = 0

    for _, module in model.named_modules():
        if hasattr(module, 'weight_mask'):
            mask = module.weight_mask
            total_params += mask.numel()
            pruned_params += int((mask == 0).sum().item())

    if total_params > 0:
        sparsity = pruned_params / total_params * 100
        return sparsity, total_params, pruned_params
    return 0.0, 0, 0


def count_alive_params(model):
    total_alive = 0
    for module in model.modules():
        if hasattr(module, "weight_mask"):
            total_alive += int(module.weight_mask.sum().item())
        elif hasattr(module, "weight"):  # En capas sin pruning
            total_alive += module.weight.numel()
    return int(total_alive)


def save_minimal_model_and_size(model, path):
    for module in model.modules():
        if hasattr(module, "weight_mask"):
            try:
                prune.remove(module, "weight")
            except ValueError:
                pass
    torch.save(model.state_dict(), path)
    size_disk_mb = os.path.getsize(path) / (1024 * 1024)
    total_params = count_alive_params(model)
    return size_disk_mb, total_params


def real_benchmark(model, loader, save_name="model", original_model=None, device="cpu", output_dir="."):
    os.makedirs(output_dir, exist_ok=True)
    process = psutil.Process(os.getpid())

    def get_model_size_mb(m):
        total_bytes = 0
        for param in m.parameters():
            total_bytes += param.numel() * param.element_size()
        for buffer in m.buffers():
            total_bytes += buffer.numel() * buffer.element_size()
        return total_bytes / (1024 * 1024)

    model = model.to(device)
    size_memory_mb = get_model_size_mb(model)

    save_path = os.path.join(output_dir, f"{save_name}.pth")
    size_disk_mb, total_params = save_minimal_model_and_size(copy.deepcopy(model).cpu(), save_path)

    compression_real = 0.0
    sparsity_info = ""
    if original_model is not None:
        compression_real = calculate_real_compression(original_model, model)
        sparsity, total, pruned = get_sparsity_info(model)
        if total > 0:
            sparsity_info = f"Sparsity: {sparsity:.2f}%, Pruned: {pruned}/{total}"

    cpu_usages = []
    mem_usages = []

    model.eval()
    start = time.time()
    with torch.inference_mode():
        for batch in loader:
            x = batch[0].to(device, non_blocking=True)
            _ = model(x)
            cpu_usages.append(process.cpu_percent(interval=None))
            mem_usages.append(process.memory_info().rss / 1024 / 1024)
    end = time.time()
    avg_time_ms = (end - start) / max(len(loader.dataset), 1) * 1000

    print(f"Tamaño en memoria: {size_memory_mb:.2f} MB")
    print(f"Tamaño en disco: {size_disk_mb:.2f} MB")
    print(f"Parámetros activos: {total_params:,d}")
    print(f"Compresión real: {compression_real:.2f}% 🎯")
    if sparsity_info:
        print(sparsity_info)
    if len(cpu_usages) > 0:
        print(f"CPU promedio (%): {sum(cpu_usages) / len(cpu_usages):.2f}")
        print(f"CPU máximo (%): {max(cpu_usages):.2f}")
    if len(mem_usages) > 0:
        print(f"Memoria máxima (MB): {max(mem_usages):.2f}")

    return size_memory_mb, size_disk_mb, avg_time_ms, compression_real, total_params


def comprehensive_pruning_study(
    base_model,
    test_loader,
    pruning_levels,
    device="cpu",
    output_dir=".",
    run_structured=True,
    run_unstructured=True,
    ln_structured_n=2,
    ln_structured_dim=0,
    save_prefix="run"
):
    results = {'structured': [], 'unstructured': []}
    base_model_cpu_copy = copy.deepcopy(base_model).cpu()

    if run_structured:
        print("🏗️  PROBANDO PRUNING ESTRUCTURADO...")
        for level in pruning_levels:
            print(f"\n🔧 Estructurado - Probando level: {level * 100:.0f}%")
            pruned_model = copy.deepcopy(base_model)

            if level > 0:
                for _, module in pruned_model.named_modules():
                    if isinstance(module, nn.Conv2d):
                        prune.ln_structured(module, name="weight", amount=level, n=ln_structured_n, dim=ln_structured_dim)

            active_params = count_alive_params(pruned_model)
            print(f"Parámetros activos tras pruning en nivel {level * 100:.1f}%: {active_params}")

            size_mem, size_disk, time_ms, compression_real, total_params = real_benchmark(
                pruned_model,
                test_loader,
                save_name=f"{save_prefix}_struct_{int(level * 100)}pct",
                original_model=base_model_cpu_copy,
                device=device,
                output_dir=output_dir
            )
            acc, f1_weighted, f1_macro, f1_benign, f1_malignant, f1_majority = evaluate_detailed(pruned_model.to(device), test_loader, device)

            results['structured'].append({
                'level_pct': level * 100,
                'size_mem_mb': size_mem,
                'size_disk_mb': size_disk,
                'active_params': active_params,
                'total_params_alive': total_params,
                'latency_ms_per_sample': time_ms,
                'compression_real_pct': compression_real,
                'accuracy': acc,
                'f1_weighted': f1_weighted,
                'f1_macro': f1_macro,
                'f1_benign': f1_benign,
                'f1_malignant': f1_malignant,
                'f1_majority': f1_majority
            })

            print(f"✅ Estructurado {level * 100:.0f}%: Acc={acc:.4f}, F1-Maligno={f1_malignant:.4f}, "
                  f"F1-Macro={f1_macro:.4f}, Compresión={compression_real:.2f}%, Tamaño={size_disk:.2f}MB")

    if run_unstructured:
        print("\n🧩 PROBANDO PRUNING NO ESTRUCTURADO...")
        for level in pruning_levels:
            print(f"\n🔧 No Estructurado - Probando level: {level * 100:.0f}%")
            pruned_model = copy.deepcopy(base_model)

            if level > 0:
                for _, module in pruned_model.named_modules():
                    if isinstance(module, nn.Conv2d):
                        prune.l1_unstructured(module, name="weight", amount=level)

            active_params = count_alive_params(pruned_model)
            print(f"Parámetros activos tras pruning en nivel {level * 100:.1f}%: {active_params}")

            size_mem, size_disk, time_ms, compression_real, total_params = real_benchmark(
                pruned_model,
                test_loader,
                save_name=f"{save_prefix}_unstruct_{int(level * 100)}pct",
                original_model=base_model_cpu_copy,
                device=device,
                output_dir=output_dir
            )
            acc, f1_weighted, f1_macro, f1_benign, f1_malignant, f1_majority = evaluate_detailed(pruned_model.to(device), test_loader, device)

            results['unstructured'].append({
                'level_pct': level * 100,
                'size_mem_mb': size_mem,
                'size_disk_mb': size_disk,
                'active_params': active_params,
                'total_params_alive': total_params,
                'latency_ms_per_sample': time_ms,
                'compression_real_pct': compression_real,
                'accuracy': acc,
                'f1_weighted': f1_weighted,
                'f1_macro': f1_macro,
                'f1_benign': f1_benign,
                'f1_malignant': f1_malignant,
                'f1_majority': f1_majority
            })

            print(f"✅ No Estructurado {level * 100:.0f}%: Acc={acc:.4f}, F1-Maligno={f1_malignant:.4f}, "
                  f"F1-Macro={f1_macro:.4f}, Compresión={compression_real:.2f}%, Tamaño={size_disk:.2f}MB")

    return results


def parse_levels(levels_str: str):

    parts = [p.strip() for p in levels_str.split(",") if p.strip() != ""]
    return [float(p) for p in parts]


def main():
    parser = argparse.ArgumentParser(description="Estudio comparativo de pruning (estructurado vs no estructurado) parametrizable.")

    parser.add_argument("--csv-path", type=str, required=True, help="Ruta al CSV con las imágenes (columnas: image_path,label).")
    parser.add_argument("--image-size", type=int, default=224, help="Tamaño de imagen (alto=ancho).")
    parser.add_argument("--batch-size", type=int, default=1, help="Tamaño de batch para inferencia.")
    parser.add_argument("--num-workers", type=int, default=4, help="Número de workers del DataLoader.")

    parser.add_argument("--model-name", type=str, default="swin_tiny_patch4_window7_224", help="Nombre del modelo TIMM.")
    parser.add_argument("--num-classes", type=int, default=2, help="Número de clases.")
    parser.add_argument("--pretrained", action="store_true", help="Usar pesos preentrenados.")
    parser.add_argument("--dropout-rate", type=float, default=0.0, help="Dropout rate del clasificador final.")

    parser.add_argument("--checkpoint-path", type=str, default="", help="Ruta local al checkpoint (.pth) con state_dict.")
    parser.add_argument("--artifact-uri", type=str, default="", help="URI del artifact de W&B (owner/project/artifact:version).")
    parser.add_argument("--artifact-file", type=str, default="", help="Nombre del archivo dentro del artifact a cargar (por ejemplo: best_model_epoch_4_state_dict.pth).")

    parser.add_argument("--pruning-levels", type=str,
                        default="0.0,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9",
                        help="Niveles de pruning separados por coma en fracción [0-1].")
    parser.add_argument("--run-structured", action="store_true", help="Ejecutar pruning estructurado.")
    parser.add_argument("--run-unstructured", action="store_true", help="Ejecutar pruning no estructurado.")
    parser.add_argument("--ln-structured-n", type=int, default=2, help="Parámetro n para ln_structured (por defecto L2).")
    parser.add_argument("--ln-structured-dim", type=int, default=0, help="Dimensión para ln_structured (por defecto 0).")

    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", choices=["cpu", "cuda"], help="Dispositivo de inferencia.")
    parser.add_argument("--output-dir", type=str, default="./outputs", help="Directorio para guardar modelos y CSVs.")
    parser.add_argument("--output-prefix", type=str, default="pruning", help="Prefijo para archivos de salida.")
    parser.add_argument("--seed", type=int, default=42, help="Semilla de reproducibilidad.")

    parser.add_argument("--use-wandb", action="store_true", help="Habilitar Weights & Biases.")
    parser.add_argument("--wandb-project", type=str, default="tu_proyecto", help="Nombre del proyecto en W&B.")
    parser.add_argument("--wandb-entity", type=str, default="", help="Entidad de W&B (opcional).")
    parser.add_argument("--wandb-run-name", type=str, default="", help="Nombre del run en W&B (opcional).")

    args = parser.parse_args()

    set_seed(args.seed)

    run = None
    if args.use_wandb:
        wandb_kwargs = dict(project=args.wandb_project)
        if args.wandb_entity:
            wandb_kwargs["entity"] = args.wandb_entity
        if args.wandb_run_name:
            wandb_kwargs["name"] = args.wandb_run_name
        run = wandb.init(**wandb_kwargs, job_type="inference", config=vars(args))

    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor()
    ])
    test_dataset = ImageDataset(args.csv_path, transform=transform)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(args.device == "cuda"),
    )

    model = get_model(args.model_name, num_classes=args.num_classes, pretrained=args.pretrained, dropout_rate=args.dropout_rate)


    checkpoint_loaded = False
    if args.checkpoint_path:
        if not os.path.isfile(args.checkpoint_path):
            raise FileNotFoundError(f"No se encontró el checkpoint: {args.checkpoint_path}")
        state = torch.load(args.checkpoint_path, map_location="cpu")
        model.load_state_dict(state)
        checkpoint_loaded = True
    elif args.artifact_uri:
        if not args.use_wandb:
            raise ValueError("Para cargar desde un artifact de W&B, habilita --use-wandb.")
        run_local = run or wandb.init(project=args.wandb_project)
        artifact = run_local.use_artifact(args.artifact_uri, type='model')
        artifact_dir = artifact.download()
        artifact_file = args.artifact_file
        if not artifact_file:

            all_files = [f for f in os.listdir(artifact_dir) if f.endswith(".pth")]
            if not all_files:
                raise FileNotFoundError("No se encontró archivo .pth en el artifact. Especifica --artifact-file.")
            artifact_file = all_files[0]
        model_path = os.path.join(artifact_dir, artifact_file)
        state = torch.load(model_path, map_location="cpu")
        model.load_state_dict(state)
        checkpoint_loaded = True

    if not checkpoint_loaded:
        print("Advertencia: no se proporcionó checkpoint ni artifact; el modelo se usará con pesos iniciales.")

    model = model.to(args.device)

    pruning_levels = parse_levels(args.pruning_levels)

    run_structured = args.run_structured or (not args.run_unstructured)
    run_unstructured = args.run_unstructured or (not args.run_structured)

    os.makedirs(args.output_dir, exist_ok=True)

    print("Pruning study...")
    results = comprehensive_pruning_study(
        base_model=model,
        test_loader=test_loader,
        pruning_levels=pruning_levels,
        device=args.device,
        output_dir=args.output_dir,
        run_structured=run_structured,
        run_unstructured=run_unstructured,
        ln_structured_n=args.ln_structured_n,
        ln_structured_dim=args.ln_structured_dim,
        save_prefix=args.output_prefix
    )

    if results['structured']:
        df_struct = pd.DataFrame(results['structured'])
        df_struct.to_csv(os.path.join(args.output_dir, f"{args.output_prefix}_structured.csv"), index=False)
    if results['unstructured']:
        df_unstruct = pd.DataFrame(results['unstructured'])
        df_unstruct.to_csv(os.path.join(args.output_dir, f"{args.output_prefix}_unstructured.csv"), index=False)

    print("Results available in:", args.output_dir)

    if run is not None:
        wandb.finish()


if __name__ == "__main__":
    main()
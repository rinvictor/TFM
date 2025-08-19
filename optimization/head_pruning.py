import argparse
import os
import re
import copy
import random
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm

try:
    import wandb
    WANDB_AVAILABLE = True
except Exception:
    WANDB_AVAILABLE = False


# --------------- Data & Utilities ---------------
class ImageDataset(Dataset):
    """Simple image dataset reading 'image_path' and 'label' columns from a CSV."""
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
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_model(model_name, num_classes, pretrained=True, dropout_rate=0.0):
    """Create a timm model with optional dropout and num_classes."""
    return timm.create_model(
        model_name,
        pretrained=pretrained,
        num_classes=num_classes,
        drop_rate=dropout_rate
    )


def evaluate_detailed(model, loader, device="cpu"):
    """Evaluate accuracy and F1 metrics, returning a tuple: acc, f1_weighted, f1_macro, f1_benign, f1_malignant."""
    model.eval()
    y_true, y_pred = [], []
    with torch.inference_mode():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            out = model(x)
            preds = out.argmax(dim=1)
            y_true.extend(y.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
    f1_benign = float(f1_per_class[0]) if len(f1_per_class) > 0 else 0.0
    f1_malignant = float(f1_per_class[1]) if len(f1_per_class) > 1 else 0.0
    return acc, f1_weighted, f1_macro, f1_benign, f1_malignant


# --------------- Pruning/Masking for Attention Heads (Swin-like) ---------------
def prune_window_attention_heads(attn_module, heads_to_prune: List[int]):
    """
    Permanently remove heads from a WindowAttention-like module by slicing its QKV and projection.
    This changes module parameters and num_heads.
    """
    embed_dim = attn_module.qkv.in_features  # input dim to qkv Linear
    num_heads = attn_module.num_heads
    for h in heads_to_prune:
        if h >= num_heads or h < 0:
            raise ValueError(f"Head index {h} out of range (num_heads={num_heads})")

    head_dim = embed_dim // num_heads
    keep_heads = [h for h in range(num_heads) if h not in heads_to_prune]
    new_num_heads = len(keep_heads)

    def block_indices(start, heads, head_dim):
        # For Q/K/V blocks inside the qkv Linear output dimension (3*embed_dim)
        return [i for h in heads for i in range(start + h * head_dim, start + (h + 1) * head_dim)]

    # qkv weight has shape [3*embed_dim, embed_dim]; we slice row dimension
    q_indices = block_indices(0, keep_heads, head_dim)
    k_indices = block_indices(embed_dim, keep_heads, head_dim)
    v_indices = block_indices(2 * embed_dim, keep_heads, head_dim)
    keep_indices = q_indices + k_indices + v_indices

    with torch.no_grad():
        attn_module.qkv.weight = torch.nn.Parameter(attn_module.qkv.weight.data[keep_indices, :])
        attn_module.qkv.bias = torch.nn.Parameter(attn_module.qkv.bias.data[keep_indices])

        new_embed_dim = head_dim * new_num_heads
        # Slice projection to match reduced head dimension
        attn_module.proj.weight = torch.nn.Parameter(attn_module.proj.weight.data[:new_embed_dim, :new_embed_dim])
        attn_module.proj.bias = torch.nn.Parameter(attn_module.proj.bias.data[:new_embed_dim])

        attn_module.num_heads = new_num_heads
        attn_module.embed_dim = new_embed_dim

        # Relative position bias table shape: [num_relative_positions, num_heads]
        if hasattr(attn_module, "relative_position_bias_table"):
            rel_pos_bias = attn_module.relative_position_bias_table.data
            if rel_pos_bias.shape[1] >= num_heads:
                attn_module.relative_position_bias_table = torch.nn.Parameter(rel_pos_bias[:, keep_heads])


def prune_swin_heads_and_update(model, heads_to_prune_dict: Dict[str, List[int]]):
    """
    Apply head pruning to modules in the model based on a dict: {module_name: [heads_to_prune]}.
    """
    for name, module in model.named_modules():
        if hasattr(module, "num_heads") and hasattr(module, "qkv"):
            heads = heads_to_prune_dict.get(name, None)
            if heads:
                prune_window_attention_heads(module, heads)


def get_attn_modules(model, name_regex: Optional[str] = None) -> Dict[str, int]:
    """
    Return a dict of attention-like modules: {module_name: num_heads}.
    Optionally filter by a regex on module name.
    """
    attn_modules = {}
    pattern = re.compile(name_regex) if name_regex else None
    for name, module in model.named_modules():
        if hasattr(module, "num_heads") and hasattr(module, "qkv"):
            if pattern is None or pattern.search(name):
                attn_modules[name] = int(module.num_heads)
    return attn_modules


def mask_attention_head(attn_module, head_idx: int):
    """
    Soft-mask a single head by zeroing its Q, K, V slices and corresponding output projection rows.
    Does not change module shape (reversible by reloading weights).
    """
    embed_dim = attn_module.qkv.in_features  # input dim
    num_heads = attn_module.num_heads
    head_dim = embed_dim // num_heads
    if head_idx < 0 or head_idx >= num_heads:
        raise ValueError(f"Head index {head_idx} out of range (num_heads={num_heads})")

    # Row ranges for the head in Q/K/V blocks
    q_range = range(head_idx * head_dim, (head_idx + 1) * head_dim)
    k_range = range(embed_dim + head_idx * head_dim, embed_dim + (head_idx + 1) * head_dim)
    v_range = range(2 * embed_dim + head_idx * head_dim, 2 * embed_dim + (head_idx + 1) * head_dim)

    with torch.no_grad():
        # Zero QKV weights/bias for this head
        attn_module.qkv.weight[q_range, :] = 0
        attn_module.qkv.weight[k_range, :] = 0
        attn_module.qkv.weight[v_range, :] = 0
        attn_module.qkv.bias[q_range] = 0
        attn_module.qkv.bias[k_range] = 0
        attn_module.qkv.bias[v_range] = 0

        # Optionally zero the output projection rows that correspond to this head
        proj_range = range(head_idx * head_dim, (head_idx + 1) * head_dim)
        attn_module.proj.weight[proj_range, :] = 0
        attn_module.proj.bias[proj_range] = 0


# --------------- Experiment Routines ---------------
def run_head_mask_sweep(
    base_model: nn.Module,
    test_loader: DataLoader,
    device: str,
    name_regex: Optional[str] = None,
    limit_modules: Optional[int] = None,
    limit_heads_per_module: Optional[int] = None,
    progress: bool = True,
):
    """
    For each attention module and each head, mask a single head and evaluate metrics.
    Returns a list of dict rows.
    """
    attn_info = get_attn_modules(base_model, name_regex=name_regex)
    results = []

    # Optionally limit modules
    module_items = list(attn_info.items())
    if limit_modules is not None:
        module_items = module_items[:max(0, int(limit_modules))]

    for mod_name, num_heads in module_items:
        heads_range = range(num_heads)
        if limit_heads_per_module is not None:
            heads_range = range(min(num_heads, int(limit_heads_per_module)))

        for h in heads_range:
            masked_model = copy.deepcopy(base_model).to(device)
            attn_module = dict(masked_model.named_modules())[mod_name]
            mask_attention_head(attn_module, h)

            acc, f1_weighted, f1_macro, f1_benign, f1_malignant = evaluate_detailed(masked_model, test_loader, device=device)
            row = {
                "module": mod_name,
                "masked_head": h,
                "num_heads_module": num_heads,
                "accuracy": acc,
                "f1_weighted": f1_weighted,
                "f1_macro": f1_macro,
                "f1_benign": f1_benign,
                "f1_malignant": f1_malignant,
            }
            results.append(row)
            if progress:
                print(f"[{mod_name} - mask head {h}] Acc={acc:.4f} | F1-Malignant={f1_malignant:.4f}")

    return results


def plot_top_bars(input_csv: str, top_k: int = 10, save_dir: Optional[str] = None, prefix: str = "head_pruning"):
    """
    Quick visualization: two horizontal bar charts for top-k by F1-malignant and F1-benign.
    """
    import matplotlib.pyplot as plt

    df = pd.read_csv(input_csv)
    if df.empty:
        print("[WARN] No data to plot.")
        return

    df_malignant = df.sort_values("f1_malignant", ascending=False).head(top_k)
    df_benign = df.sort_values("f1_benign", ascending=False).head(top_k)

    # Chart 1: top by F1 malignant
    fig1 = plt.figure(figsize=(10, 5))
    plt.barh(
        df_malignant["module"].astype(str) + "_h" + df_malignant["masked_head"].astype(str),
        df_malignant["f1_malignant"],
        color="crimson"
    )
    plt.gca().invert_yaxis()
    plt.xlabel("F1 Malignant")
    plt.title(f"Top {top_k} candidates to mask (maximize F1-Malignant)")
    plt.tight_layout()
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        out1 = os.path.join(save_dir, f"{prefix}_top{top_k}_malignant.png")
        plt.savefig(out1, dpi=150, bbox_inches="tight")
        print(f"[INFO] Saved: {out1}")
        plt.close(fig1)
    else:
        plt.show()

    # Chart 2: top by F1 benign
    fig2 = plt.figure(figsize=(10, 5))
    plt.barh(
        df_benign["module"].astype(str) + "_h" + df_benign["masked_head"].astype(str),
        df_benign["f1_benign"],
        color="seagreen"
    )
    plt.gca().invert_yaxis()
    plt.xlabel("F1 Benign")
    plt.title(f"Top {top_k} candidates to mask (maximize F1-Benign)")
    plt.tight_layout()
    if save_dir:
        out2 = os.path.join(save_dir, f"{prefix}_top{top_k}_benign.png")
        plt.savefig(out2, dpi=150, bbox_inches="tight")
        print(f"[INFO] Saved: {out2}")
        plt.close(fig2)
    else:
        plt.show()


# --------------- CLI ---------------
def main():
    parser = argparse.ArgumentParser(description="Parametrizable head masking sweep for Swin-like attention modules.")

    # Data
    parser.add_argument("--csv-path", type=str, required=True, help="Path to CSV with columns: image_path,label.")
    parser.add_argument("--image-size", type=int, default=224, help="Image size (HxW).")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for evaluation.")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of DataLoader workers.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", choices=["cpu", "cuda"], help="Device for inference.")

    # Model
    parser.add_argument("--model-name", type=str, default="swin_tiny_patch4_window7_224", help="timm model name.")
    parser.add_argument("--num-classes", type=int, default=2, help="Number of classes.")
    parser.add_argument("--pretrained", action="store_true", help="Use pretrained weights when creating the model.")
    parser.add_argument("--dropout-rate", type=float, default=0.0, help="Dropout rate for classifier head (if applicable).")

    # Weights
    parser.add_argument("--checkpoint-path", type=str, default="", help="Local path to a state_dict (.pth).")
    parser.add_argument("--use-wandb", action="store_true", help="Enable Weights & Biases to download artifacts.")
    parser.add_argument("--wandb-project", type=str, default="tu_proyecto", help="W&B project name.")
    parser.add_argument("--wandb-entity", type=str, default="", help="W&B entity (optional).")
    parser.add_argument("--artifact-uri", type=str, default="", help="Artifact URI, e.g., 'owner/project/artifact:version'.")
    parser.add_argument("--artifact-file", type=str, default="", help="Filename inside the artifact, e.g., 'best_model_epoch_4_state_dict.pth'.")

    # Sweep options
    parser.add_argument("--module-name-regex", type=str, default="", help="Regex to filter attention module names (empty = all).")
    parser.add_argument("--limit-modules", type=int, default=None, help="Limit number of modules to test (from the start).")
    parser.add_argument("--limit-heads-per-module", type=int, default=None, help="Limit heads tested per module (from 0..limit-1).")

    # Output
    parser.add_argument("--out-csv", type=str, default="head_pruning_results.csv", help="Output CSV for results.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")

    # Plotting
    parser.add_argument("--plot", action="store_true", help="Generate quick bar plots for top-K by F1.")
    parser.add_argument("--plot-topk", type=int, default=10, help="Top-K to display in plots.")
    parser.add_argument("--plots-dir", type=str, default="", help="Directory to save plots (empty = show instead of saving).")

    args = parser.parse_args()

    # Seed and device
    set_seed(args.seed)
    device = args.device

    # Optional W&B
    run = None
    if args.use_wandb:
        if not WANDB_AVAILABLE:
            raise RuntimeError("wandb is not installed but --use-wandb was provided.")
        wandb_kwargs = dict(project=args.wandb_project)
        if args.wandb_entity:
            wandb_kwargs["entity"] = args.wandb_entity
        run = wandb.init(**wandb_kwargs, job_type="inference", config=vars(args))

    # Dataloader
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
        pin_memory=(device == "cuda"),
    )

    # Model
    model = get_model(
        args.model_name,
        num_classes=args.num_classes,
        pretrained=args.pretrained,
        dropout_rate=args.dropout_rate
    )

    # Load weights
    loaded = False
    if args.checkpoint_path:
        if not os.path.isfile(args.checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint_path}")
        state = torch.load(args.checkpoint_path, map_location="cpu")
        model.load_state_dict(state)
        loaded = True
    elif args.artifact_uri:
        if not args.use_wandb:
            raise ValueError("To load from a W&B artifact, enable --use-wandb.")
        run_local = run or wandb.init(project=args.wandb_project)
        artifact = run_local.use_artifact(args.artifact_uri, type='model')
        artifact_dir = artifact.download()
        artifact_file = args.artifact_file
        if not artifact_file:
            # Try to pick the first .pth if none provided
            candidates = [f for f in os.listdir(artifact_dir) if f.endswith(".pth")]
            if not candidates:
                raise FileNotFoundError("No .pth file found in the artifact. Specify --artifact-file.")
            artifact_file = candidates[0]
        model_path = os.path.join(artifact_dir, artifact_file)
        state = torch.load(model_path, map_location="cpu")
        model.load_state_dict(state)
        loaded = True

    if not loaded:
        print("[WARN] No checkpoint/artifact provided. Using model as initialized.")

    model = model.to(device)

    # Run sweep: mask one head at a time and evaluate
    print("[INFO] Starting head masking sweep...")
    results = run_head_mask_sweep(
        base_model=model,
        test_loader=test_loader,
        device=device,
        name_regex=(args.module_name_regex if args.module_name_regex.strip() else None),
        limit_modules=args.limit_modules,
        limit_heads_per_module=args.limit_heads_per_module,
        progress=True
    )

    # Save results
    df = pd.DataFrame(results)
    df.to_csv(args.out_csv, index=False)
    print(f"[INFO] Saved results to: {args.out_csv}")

    # Select best row by F1 malignant (example selection)
    if not df.empty:
        best_row = df.loc[df['f1_malignant'].idxmax()]
        print(f"\nBest single-head mask: module '{best_row['module']}', head {best_row['masked_head']}")
        print(f"Acc={best_row['accuracy']:.4f} | F1-Malignant={best_row['f1_malignant']:.4f}")

    # Optional plots
    if args.plot:
        plots_dir = args.plots_dir if args.plots_dir.strip() else None
        plot_top_bars(args.out_csv, top_k=args.plot_topk, save_dir=plots_dir)

    if args.use_wandb and run is not None:
        # Log the CSV if desired
        try:
            wandb.save(args.out_csv)
        except Exception:
            pass
        wandb.finish()


if __name__ == "__main__":
    main()
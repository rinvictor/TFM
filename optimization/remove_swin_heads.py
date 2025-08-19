import argparse
import json
import os
import re
import random
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
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
    """
    Evaluate accuracy and F1 metrics.
    Returns: acc, f1_weighted, f1_macro, f1_benign, f1_malignant
    """
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
    f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
    f1_benign = float(f1_per_class[0]) if len(f1_per_class) > 0 else 0.0
    f1_malignant = float(f1_per_class[1]) if len(f1_per_class) > 1 else 0.0
    return acc, f1_weighted, f1_macro, f1_benign, f1_malignant


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
        return [i for h in heads for i in range(start + h * head_dim, start + (h + 1) * head_dim)]

    q_indices = block_indices(0, keep_heads, head_dim)
    k_indices = block_indices(embed_dim, keep_heads, head_dim)
    v_indices = block_indices(2 * embed_dim, keep_heads, head_dim)
    keep_indices = q_indices + k_indices + v_indices

    with torch.no_grad():
        attn_module.qkv.weight = torch.nn.Parameter(attn_module.qkv.weight.data[keep_indices, :])
        attn_module.qkv.bias = torch.nn.Parameter(attn_module.qkv.bias.data[keep_indices])

        new_embed_dim = head_dim * new_num_heads
        attn_module.proj.weight = torch.nn.Parameter(attn_module.proj.weight.data[:new_embed_dim, :new_embed_dim])
        attn_module.proj.bias = torch.nn.Parameter(attn_module.proj.bias.data[:new_embed_dim])

        attn_module.num_heads = new_num_heads
        attn_module.embed_dim = new_embed_dim

        if hasattr(attn_module, "relative_position_bias_table"):
            rel_pos_bias = attn_module.relative_position_bias_table.data
            if rel_pos_bias.dim() == 2 and rel_pos_bias.shape[1] >= num_heads:
                attn_module.relative_position_bias_table = torch.nn.Parameter(rel_pos_bias[:, keep_heads])


def prune_swin_heads_and_update(model, heads_to_prune_dict: Dict[str, List[int]]):
    """
    Apply head pruning to modules in the model based on a dict: {module_name: [heads_to_prune]}.
    """
    name_to_module = dict(model.named_modules())
    for name, heads in heads_to_prune_dict.items():
        if not heads:
            continue
        if name not in name_to_module:
            raise KeyError(f"Module '{name}' not found in model.")
        module = name_to_module[name]
        if not (hasattr(module, "num_heads") and hasattr(module, "qkv")):
            raise TypeError(f"Module '{name}' does not look like a Swin WindowAttention (missing num_heads/qkv).")
        prune_window_attention_heads(module, heads)


def parse_heads_spec(spec: str) -> Dict[str, List[int]]:
    """
    Parse a heads spec string like:
      "layers.2.blocks.0.attn:4,6;layers.3.blocks.0.attn:1"
    into:
      {"layers.2.blocks.0.attn": [4,6], "layers.3.blocks.0.attn": [1]}
    Whitespace around tokens is ignored.
    """
    result: Dict[str, List[int]] = {}
    if not spec.strip():
        return result
    parts = [p for p in spec.split(";") if p.strip() != ""]
    for part in parts:
        if ":" not in part:
            raise ValueError(f"Invalid heads spec segment (missing ':'): '{part}'")
        mod, heads_str = part.split(":", 1)
        mod = mod.strip()
        heads = [h for h in heads_str.split(",") if h.strip() != ""]
        try:
            result[mod] = [int(h.strip()) for h in heads]
        except ValueError:
            raise ValueError(f"Invalid head index in segment: '{part}'")
    return result


def parse_heads_json(json_str: str) -> Dict[str, List[int]]:
    """
    Parse a JSON string mapping module names to lists of head indices.
    Example:
      '{"layers.2.blocks.0.attn":[4],"layers.3.blocks.0.attn":[1,2]}'
    """
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON for --heads-json: {e}")
    if not isinstance(data, dict):
        raise ValueError("Heads JSON must be an object mapping module names to lists of ints.")
    result: Dict[str, List[int]] = {}
    for k, v in data.items():
        if not isinstance(v, list) or not all(isinstance(x, int) for x in v):
            raise ValueError(f"Value for key '{k}' must be a list of integers.")
        result[k] = v
    return result


def main():
    parser = argparse.ArgumentParser(description="Swin head pruning (parametrizable) with evaluation before/after.")

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

    # Heads to prune
    parser.add_argument("--heads", type=str, default="", help="Heads spec: 'module1:h1,h2;module2:h3'.")
    parser.add_argument("--heads-json", type=str, default="", help='JSON mapping of modules to head indices, e.g. \'{"layers.2.blocks.0.attn":[4]}\'.')
    parser.add_argument("--heads-json-file", type=str, default="", help="Path to JSON file mapping modules to head indices.")
    parser.add_argument("--list-attn", action="store_true", help="List attention modules and exit.")
    parser.add_argument("--module-name-regex", type=str, default="", help="Regex to filter attention names when listing.")

    parser.add_argument("--no-eval-before", action="store_true", help="Skip evaluation before pruning.")
    parser.add_argument("--no-eval-after", action="store_true", help="Skip evaluation after pruning.")

    parser.add_argument("--save-path", type=str, default="swin_tiny_pruned.pth", help="Where to save the pruned model state_dict.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")

    args = parser.parse_args()


    set_seed(args.seed)
    device = args.device


    run = None
    if args.use_wandb:
        if not WANDB_AVAILABLE:
            raise RuntimeError("wandb is not installed but --use-wandb was provided.")
        wandb_kwargs = dict(project=args.wandb_project)
        if args.wandb_entity:
            wandb_kwargs["entity"] = args.wandb_entity
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
        pin_memory=(device == "cuda"),
    )

    model = get_model(
        args.model_name,
        num_classes=args.num_classes,
        pretrained=args.pretrained,
        dropout_rate=args.dropout_rate
    ).to(device)

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

    if args.list_attn:
        print("[INFO] Listing attention modules (num_heads) ...")
        info = get_attn_modules(model, name_regex=(args.module_name_regex or None))
        for name, heads in info.items():
            print(f"{name}: {heads}")
        return

    if not args.no_eval_before:
        acc0, f1w0, f1m0, f1b0, f1mal0 = evaluate_detailed(model, test_loader, device=device)
        print("\n>>>>> METRICS BEFORE PRUNING <<<<<")
        print(f"Accuracy:    {acc0:.4f}")
        print(f"F1 macro:    {f1m0:.4f}")
        print(f"F1 weighted: {f1w0:.4f}")
        print(f"F1 benign:   {f1b0:.4f}")
        print(f"F1 malignant:{f1mal0:.4f}")

    heads_to_prune: Dict[str, List[int]] = {}
    if args.heads_json.strip():
        heads_to_prune = parse_heads_json(args.heads_json)
    elif args.heads_json_file.strip():
        if not os.path.isfile(args.heads_json_file):
            raise FileNotFoundError(f"heads JSON file not found: {args.heads_json_file}")
        with open(args.heads_json_file, "r") as f:
            heads_to_prune = parse_heads_json(f.read())
    elif args.heads.strip():
        heads_to_prune = parse_heads_spec(args.heads)
    else:
        heads_to_prune = {"layers.2.blocks.0.attn": [4]}

    print("\n[INFO] Applying head pruning ...")
    prune_swin_heads_and_update(model, heads_to_prune)

    # Evaluate after pruning
    if not args.no_eval_after:
        acc1, f1w1, f1m1, f1b1, f1mal1 = evaluate_detailed(model, test_loader, device=device)
        print("\n>>>>> METRICS AFTER PRUNING <<<<<")
        print(f"Accuracy:    {acc1:.4f}")
        print(f"F1 macro:    {f1m1:.4f}")
        print(f"F1 weighted: {f1w1:.4f}")
        print(f"F1 benign:   {f1b1:.4f}")
        print(f"F1 malignant:{f1mal1:.4f}")

    torch.save(model.state_dict(), args.save_path)
    print(f"\n[INFO] Pruned model saved to '{args.save_path}'.")

    if args.use_wandb and run is not None:
        try:
            wandb.save(args.save_path)
        except Exception:
            pass
        wandb.finish()


if __name__ == "__main__":
    main()
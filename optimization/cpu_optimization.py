import argparse
import os
import time
import copy
import random

import numpy as np
import pandas as pd
import psutil
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import timm
import wandb
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.models as tv_models


def get_model(model_name: str, num_classes: int, pretrained: bool = True, dropout_rate: float = 0.0):
    return timm.create_model(
        model_name,
        pretrained=pretrained,
        num_classes=num_classes,
        drop_rate=dropout_rate
    )


class CustomClassifier(nn.Module):
    def __init__(self, encoder, num_classes, dropout_rate=0.0):
        super(CustomClassifier, self).__init__()
        self.encoder = encoder
        num_features = self._get_encoder_output_features(self.encoder)
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(num_features, num_classes)
        )

    def _get_encoder_output_features(self, encoder):
        return encoder.classifier[1].in_features

    def forward(self, x):
        x = self.encoder.features(x)
        x = self.encoder.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def _get_efficientnet_b3(pretrained: str | None):
    weights = tv_models.EfficientNet_B3_Weights.IMAGENET1K_V1 if pretrained == 'imagenet' else None
    encoder = tv_models.efficientnet_b3(weights=weights)
    return encoder


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


def evaluate(model, loader, device="cpu"):
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

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    acc_global = accuracy_score(y_true, y_pred)

    f1_class_0 = f1_score(y_true, y_pred, labels=[0], average='weighted', zero_division=0)
    f1_class_1 = f1_score(y_true, y_pred, labels=[1], average='weighted', zero_division=0)
    f1_global = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    return {
        "acc_global": acc_global,
        "f1_global": f1_global,
        "f1_class_0": f1_class_0,
        "f1_class_1": f1_class_1
    }


def warmup_model(model, dataloader, steps=3, device="cpu"):
    model.eval()
    data_iter = iter(dataloader)
    with torch.no_grad():
        for _ in range(steps):
            try:
                inputs, _ = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                inputs, _ = next(data_iter)
            _ = model(inputs.to(device, non_blocking=True))
    return model


def real_benchmark(model, loader, save_path, device="cpu"):
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

    torch.save(model.state_dict(), save_path)
    size_disk_mb = os.path.getsize(save_path) / (1024 * 1024)

    cpu_usages, mem_usages, latencies = [], [], []
    model.eval()
    start_total = time.time()
    with torch.inference_mode():
        for batch in loader:
            x = batch[0].to(device, non_blocking=True)
            start = time.time()
            _ = model(x)
            end = time.time()

            latencies.append((end - start) * 1000)
            cpu_usages.append(process.cpu_percent(interval=None))
            mem_usages.append(process.memory_info().rss / 1024 / 1024)
    end_total = time.time()

    total_time = (end_total - start_total) * 1000
    avg_time = float(np.mean(latencies)) if latencies else 0.0
    throughput = (len(loader.dataset) / (end_total - start_total)) if (end_total - start_total) > 0 else 0.0
    p50 = float(np.percentile(latencies, 50)) if latencies else 0.0
    p95 = float(np.percentile(latencies, 95)) if latencies else 0.0
    p99 = float(np.percentile(latencies, 99)) if latencies else 0.0

    print(f"Memory size: {size_memory_mb:.2f} MB | Disk size: {size_disk_mb:.2f} MB")
    if cpu_usages:
        print(f"CPU avg (%): {sum(cpu_usages) / len(cpu_usages):.2f} | CPU max (%): {max(cpu_usages):.2f}")
    if mem_usages:
        print(f"Max memory (MB): {max(mem_usages):.2f}")
    print(f"Total time: {total_time:.2f} ms | Avg latency: {avg_time:.2f} ms | p50: {p50:.2f} | p95: {p95:.2f} | p99: {p99:.2f} | Throughput: {throughput:.2f} img/s")

    return size_memory_mb, size_disk_mb, avg_time, throughput, p95, p99


def try_compile(model, backend="inductor", desc="model"):
    try:
        compiled = torch.compile(model, backend=backend)
        print(f"⚡ Compiled {desc} with backend='{backend}'")
        return compiled, True
    except Exception as e:
        print(f"⚠️ Could not compile {desc} with backend='{backend}': {e}")
        return model, False


def apply_pruning(model, amount=0.07, n=2, dim=0, remove_masks=True):
    pruned = copy.deepcopy(model)
    for _, module in pruned.named_modules():
        if isinstance(module, nn.Conv2d):
            prune.ln_structured(module, name="weight", amount=amount, n=n, dim=dim)
            if remove_masks:
                try:
                    prune.remove(module, "weight")
                except Exception:
                    pass
    return pruned


def quantize_dynamic_linear(model, engine="fbgemm"):
    torch.backends.quantized.engine = engine
    qmodel = torch.ao.quantization.quantize_dynamic(
        copy.deepcopy(model),
        {nn.Linear},
        dtype=torch.qint8
    )
    return qmodel


def evaluate_pipeline(model_cpu, test_loader, args, out_prefix, out_dir):
    results = {}

    if args.run_original:
        print("\n📊 Benchmarking ORIGINAL model")
        original = copy.deepcopy(model_cpu).to(args.device)
        if args.warmup_steps > 0:
            warmup_model(original, test_loader, steps=args.warmup_steps, device=args.device)
        size_mem, size_disk, lat, thr, p95, p99 = real_benchmark(
            original, test_loader, os.path.join(out_dir, f"{out_prefix}_original.pth"), device=args.device
        )
        metrics = evaluate(original, test_loader, device=args.device)
        results["original"] = dict(mem=size_mem, disk=size_disk, lat=lat, p95=p95, p99=p99, throughput=thr,
                                   acc=metrics["acc_global"], f1=metrics["f1_global"],
                                   f1_class_0=metrics["f1_class_0"], f1_class_1=metrics["f1_class_1"])

    if args.run_compiled:
        print("\n⚡ Compiling ORIGINAL model")
        compiled, _ = try_compile(copy.deepcopy(model_cpu).to(args.device), backend=args.compile_backend, desc="original")
        if args.warmup_steps > 0:
            warmup_model(compiled, test_loader, steps=args.warmup_steps, device=args.device)
        print("📊 Benchmarking COMPILED model")
        size_mem, size_disk, lat, thr, p95, p99 = real_benchmark(
            compiled, test_loader, os.path.join(out_dir, f"{out_prefix}_compiled.pth"), device=args.device
        )
        metrics = evaluate(compiled, test_loader, device=args.device)
        results["compiled"] = dict(mem=size_mem, disk=size_disk, lat=lat, p95=p95, p99=p99, throughput=thr,
                                   acc=metrics["acc_global"], f1=metrics["f1_global"],
                                   f1_class_0=metrics["f1_class_0"], f1_class_1=metrics["f1_class_1"])

    if args.run_pruned:
        print("\n🔧 Applying PRUNING")
        pruned = apply_pruning(model_cpu, amount=args.prune_amount, n=args.prune_ln_n, dim=args.prune_ln_dim,
                               remove_masks=True).to(args.device)
        if args.compile_pruned:
            print("⚡ Compiling PRUNED model")
            pruned, _ = try_compile(pruned, backend=args.compile_backend, desc="pruned")
        if args.warmup_steps > 0:
            warmup_model(pruned, test_loader, steps=args.warmup_steps, device=args.device)
        print("📊 Benchmarking PRUNED model")
        size_mem, size_disk, lat, thr, p95, p99 = real_benchmark(
            pruned, test_loader, os.path.join(out_dir, f"{out_prefix}_pruned.pth"), device=args.device
        )
        metrics = evaluate(pruned, test_loader, device=args.device)
        results["pruned"] = dict(mem=size_mem, disk=size_disk, lat=lat, p95=p95, p99=p99, throughput=thr,
                                 acc=metrics["acc_global"], f1=metrics["f1_global"],
                                 f1_class_0=metrics["f1_class_0"], f1_class_1=metrics["f1_class_1"])

    if args.run_quant_fbgemm:
        print("\n🔧 Dynamic quantization (fbgemm, CPU)")
        pruned_for_quant = apply_pruning(model_cpu, amount=args.prune_amount, n=args.prune_ln_n, dim=args.prune_ln_dim,
                                         remove_masks=True)
        q_fbgemm = quantize_dynamic_linear(pruned_for_quant, engine="fbgemm")
        q_device = "cpu"  # dynamic quantization runs on CPU
        if args.compile_quant:
            print("⚡ Compiling QUANTIZED model (fbgemm)")
            q_fbgemm, _ = try_compile(q_fbgemm.to(q_device), backend=args.compile_quant_backend, desc="quant_fbgemm")
        if args.warmup_steps > 0:
            warmup_model(q_fbgemm, test_loader, steps=args.warmup_steps, device=q_device)
        print("📊 Benchmarking QUANTIZED model (fbgemm)")
        size_mem, size_disk, lat, thr, p95, p99 = real_benchmark(
            q_fbgemm, test_loader, os.path.join(out_dir, f"{out_prefix}_quant_fbgemm.pth"), device=q_device
        )
        metrics = evaluate(q_fbgemm, test_loader, device=q_device)
        results["quantized_pruned"] = dict(mem=size_mem, disk=size_disk, lat=lat, p95=p95, p99=p99, throughput=thr,
                                           acc=metrics["acc_global"], f1=metrics["f1_global"],
                                           f1_class_0=metrics["f1_class_0"], f1_class_1=metrics["f1_class_1"])

    if args.run_quant_qnnpack:
        print("\n🔧 Dynamic quantization (qnnpack, CPU/ARM)")
        pruned_for_quant = apply_pruning(model_cpu, amount=args.prune_amount, n=args.prune_ln_n, dim=args.prune_ln_dim,
                                         remove_masks=True)
        q_qnnpack = quantize_dynamic_linear(pruned_for_quant, engine="qnnpack")
        q_device = "cpu"
        if args.compile_quant:
            print("⚡ Compiling QUANTIZED model (qnnpack)")
            q_qnnpack, _ = try_compile(q_qnnpack.to(q_device), backend=args.compile_quant_backend, desc="quant_qnnpack")
        if args.warmup_steps > 0:
            warmup_model(q_qnnpack, test_loader, steps=args.warmup_steps, device=q_device)
        print("📊 Benchmarking QUANTIZED model (qnnpack)")
        size_mem, size_disk, lat, thr, p95, p99 = real_benchmark(
            q_qnnpack, test_loader, os.path.join(out_dir, f"{out_prefix}_quant_qnnpack.pth"), device=q_device
        )
        metrics = evaluate(q_qnnpack, test_loader, device=q_device)
        results["quantized_pruned_qnnpack"] = dict(mem=size_mem, disk=size_disk, lat=lat, p95=p95, p99=p99, throughput=thr,
                                                   acc=metrics["acc_global"], f1=metrics["f1_global"],
                                                   f1_class_0=metrics["f1_class_0"], f1_class_1=metrics["f1_class_1"])

    if args.run_script_fbgemm:
        print("\n🔧 TorchScript (quantized fbgemm)")
        pruned_for_quant = apply_pruning(model_cpu, amount=args.prune_amount, n=args.prune_ln_n, dim=args.prune_ln_dim,
                                         remove_masks=True)
        q_fbgemm = quantize_dynamic_linear(pruned_for_quant, engine="fbgemm")
        try:
            scripted = torch.jit.script(q_fbgemm)
        except Exception as e:
            print(f"⚠️ Could not script the quantized fbgemm model: {e}")
            scripted = q_fbgemm
        q_device = "cpu"
        if args.warmup_steps > 0:
            warmup_model(scripted, test_loader, steps=args.warmup_steps, device=q_device)
        print("📊 Benchmarking TorchScript (quant+pruned, fbgemm)")
        size_mem, size_disk, lat, thr, p95, p99 = real_benchmark(
            scripted, test_loader, os.path.join(out_dir, f"{out_prefix}_script_fbgemm.pth"), device=q_device
        )
        metrics = evaluate(scripted, test_loader, device=q_device)
        results["torchscript_quant_pruned"] = dict(mem=size_mem, disk=size_disk, lat=lat, p95=p95, p99=p99, throughput=thr,
                                                   acc=metrics["acc_global"], f1=metrics["f1_global"],
                                                   f1_class_0=metrics["f1_class_0"], f1_class_1=metrics["f1_class_1"])

    if args.run_script_qnnpack:
        print("\n🔧 TorchScript (quantized qnnpack)")
        pruned_for_quant = apply_pruning(model_cpu, amount=args.prune_amount, n=args.prune_ln_n, dim=args.prune_ln_dim,
                                         remove_masks=True)
        q_qnnpack = quantize_dynamic_linear(pruned_for_quant, engine="qnnpack")
        try:
            scripted = torch.jit.script(q_qnnpack)
        except Exception as e:
            print(f"⚠️ Could not script the quantized qnnpack model: {e}")
            scripted = q_qnnpack
        q_device = "cpu"
        if args.warmup_steps > 0:
            warmup_model(scripted, test_loader, steps=args.warmup_steps, device=q_device)
        print("📊 Benchmarking TorchScript (quant+pruned, qnnpack)")
        size_mem, size_disk, lat, thr, p95, p99 = real_benchmark(
            scripted, test_loader, os.path.join(out_dir, f"{out_prefix}_script_qnnpack.pth"), device=q_device
        )
        metrics = evaluate(scripted, test_loader, device=q_device)
        results["torchscript_quant_pruned_qnnpack"] = dict(mem=size_mem, disk=size_disk, lat=lat, p95=p95, p99=p99, throughput=thr,
                                                           acc=metrics["acc_global"], f1=metrics["f1_global"],
                                                           f1_class_0=metrics["f1_class_0"], f1_class_1=metrics["f1_class_1"])

    return results


def parse_int_list(s: str | None, default=None):
    if s is None or s.strip() == "":
        return default if default is not None else []
    return [int(x.strip()) for x in s.split(",") if x.strip() != ""]


def parse_float_list(s: str | None, default=None):
    if s is None or s.strip() == "":
        return default if default is not None else []
    return [float(x.strip()) for x in s.split(",") if x.strip() != ""]


def main():
    parser = argparse.ArgumentParser(description="Parametrizable benchmark of models (original, compiled, pruned, quant, TorchScript).")

    parser.add_argument("--csv-path", type=str, required=True, help="Path to CSV with columns: image_path,label")
    parser.add_argument("--image-size", type=int, default=224, help="Image size (HxW).")
    parser.add_argument("--pin-memory", action="store_true", help="Enable pin_memory in DataLoader.")

    parser.add_argument("--batch-sizes", type=str, default="1,2,4,8,16,32", help="Comma-separated batch sizes, e.g., '1,2,4'.")
    parser.add_argument("--num-workers-list", type=str, default="", help="Comma-separated num_workers, e.g., '1,4,8'. Empty = auto.")
    parser.add_argument("--prefetch-factors", type=str, default="1,2,4", help="Comma-separated prefetch_factor (requires num_workers>0).")

    parser.add_argument("--arch", type=str, choices=["timm", "efficientnet_b3_custom"], default="efficientnet_b3_custom",
                        help="Architecture type to load.")
    parser.add_argument("--model-name", type=str, default="swin_tiny_patch4_window7_224", help="TIMM model name (if arch=timm).")
    parser.add_argument("--num-classes", type=int, default=2, help="Number of classes.")
    parser.add_argument("--pretrained", action="store_true", help="Use pre-trained weights (timm).")
    parser.add_argument("--dropout-rate", type=float, default=0.0, help="Classifier dropout (timm/custom).")
    parser.add_argument("--efficientnet-pretrained", type=str, default=None, choices=[None, "imagenet"],
                        help="Weights for EfficientNet-B3 (None or 'imagenet').")

    parser.add_argument("--checkpoint-path", type=str, default="", help="Path to state_dict (.pth).")
    parser.add_argument("--use-wandb", action="store_true", help="Enable W&B.")
    parser.add_argument("--wandb-project", type=str, default="tu_proyecto", help="W&B project.")
    parser.add_argument("--wandb-entity", type=str, default="", help="W&B entity.")
    parser.add_argument("--artifact-uri", type=str, default="", help="Artifact URI (owner/project/artifact:version).")
    parser.add_argument("--artifact-file", type=str, default="", help="File inside the artifact (.pth).")

    parser.add_argument("--run-original", action="store_true", help="Run evaluation for the original model.")
    parser.add_argument("--run-compiled", action="store_true", help="Run evaluation for the compiled model.")
    parser.add_argument("--run-pruned", action="store_true", help="Run evaluation for the pruned model.")
    parser.add_argument("--run-quant-fbgemm", action="store_true", help="Run evaluation for the quantized model (fbgemm).")
    parser.add_argument("--run-quant-qnnpack", action="store_true", help="Run evaluation for the quantized model (qnnpack).")
    parser.add_argument("--run-script-fbgemm", action="store_true", help="Run TorchScript evaluation (fbgemm).")
    parser.add_argument("--run-script-qnnpack", action="store_true", help="Run TorchScript evaluation (qnnpack).")

    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", choices=["cpu", "cuda"], help="Inference device for original/compiled/pruned.")
    parser.add_argument("--compile-backend", type=str, default="inductor", help="torch.compile backend for non-quant models.")
    parser.add_argument("--compile-quant-backend", type=str, default="eager", help="torch.compile backend for quantized models (often 'eager').")
    parser.add_argument("--compile-pruned", action="store_true", help="Compile the pruned model before evaluating.")
    parser.add_argument("--compile-quant", action="store_true", help="Try compiling quantized models.")
    parser.add_argument("--prune-amount", type=float, default=0.07, help="Pruning amount (fraction 0-1).")
    parser.add_argument("--prune-ln-n", type=int, default=2, help="Parameter n for ln_structured.")
    parser.add_argument("--prune-ln-dim", type=int, default=0, help="Dimension for ln_structured.")
    parser.add_argument("--warmup-steps", type=int, default=3, help="Warmup steps before benchmarking.")

    parser.add_argument("--output-dir", type=str, default="./outputs_bench", help="Output directory.")
    parser.add_argument("--output-csv", type=str, default="benchmark_results.csv", help="Results CSV filename.")
    parser.add_argument("--output-prefix", type=str, default="model", help="Prefix for saved files.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")

    args = parser.parse_args()

    set_seed(args.seed)

    # W&B
    run = None
    if args.use_wandb:
        wandb_kwargs = dict(project=args.wandb_project)
        if args.wandb_entity:
            wandb_kwargs["entity"] = args.wandb_entity
        run = wandb.init(**wandb_kwargs, job_type="inference", config=vars(args))

    # Dataset / Dataloader configuration
    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor()
    ])
    test_dataset = ImageDataset(args.csv_path, transform=transform)

    # Determine sweep lists
    def _parse_int_list(s: str | None, default=None):
        if s is None or s.strip() == "":
            return default if default is not None else []
        return [int(x.strip()) for x in s.split(",") if x.strip() != ""]

    batch_sizes = _parse_int_list(args.batch_sizes)
    if args.num_workers_list.strip() == "":
        n_cores = os.cpu_count() or 4
        num_workers_list = sorted(list(set([1, max(1, n_cores // 4), max(1, n_cores // 2), n_cores])))
    else:
        num_workers_list = _parse_int_list(args.num_workers_list)
    prefetch_factors = _parse_int_list(args.prefetch_factors)

    # Base model (load weights on CPU by default)
    if args.arch == "timm":
        model = get_model(args.model_name, num_classes=args.num_classes, pretrained=args.pretrained, dropout_rate=args.dropout_rate)
    else:
        encoder = _get_efficientnet_b3(pretrained=args.efficientnet_pretrained)
        model = CustomClassifier(encoder, num_classes=args.num_classes, dropout_rate=args.dropout_rate)

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
            candidates = [f for f in os.listdir(artifact_dir) if f.endswith(".pth")]
            if not candidates:
                raise FileNotFoundError("No .pth file found in the artifact. Specify --artifact-file.")
            artifact_file = candidates[0]
        model_path = os.path.join(artifact_dir, artifact_file)
        state = torch.load(model_path, map_location="cpu")
        model.load_state_dict(state)
        loaded = True

    if not loaded:
        print("No checkpoint provided")

    # If the user did not choose any --run-* flag, run all by default
    if not any([args.run_original, args.run_compiled, args.run_pruned, args.run_quant_fbgemm,
                args.run_quant_qnnpack, args.run_script_fbgemm, args.run_script_qnnpack]):
        args.run_original = True
        args.run_compiled = True
        args.run_pruned = True
        args.run_quant_fbgemm = True
        args.run_quant_qnnpack = True
        args.run_script_fbgemm = True
        args.run_script_qnnpack = True

    os.makedirs(args.output_dir, exist_ok=True)
    all_rows = []

    # Sweep
    for bs in batch_sizes:
        for nw in num_workers_list:
            for pf in prefetch_factors:
                if nw == 0:
                    # prefetch_factor is not valid with num_workers=0
                    dataloader = DataLoader(
                        test_dataset,
                        batch_size=bs,
                        shuffle=False,
                        num_workers=0,
                        pin_memory=args.pin_memory
                    )
                    pf_used = None
                else:
                    dataloader = DataLoader(
                        test_dataset,
                        batch_size=bs,
                        shuffle=False,
                        num_workers=nw,
                        pin_memory=args.pin_memory,
                        prefetch_factor=pf
                    )
                    pf_used = pf

                print(f"\n🔹 Config: batch_size={bs}, num_workers={nw}, prefetch_factor={pf_used}")
                run_id = f"{args.output_prefix}_bs{bs}_nw{nw}_pf{pf if pf_used is not None else 0}"

                metrics_dict = evaluate_pipeline(
                    model_cpu=copy.deepcopy(model).cpu(),
                    test_loader=dataloader,
                    args=args,
                    out_prefix=run_id,
                    out_dir=args.output_dir
                )

                for model_type, vals in metrics_dict.items():
                    all_rows.append({
                        "batch_size": bs,
                        "num_workers": nw,
                        "prefetch_factor": pf_used if pf_used is not None else 0,
                        "model_type": model_type,
                        "mem_MB": vals["mem"],
                        "disk_MB": vals["disk"],
                        "latency_ms": vals["lat"],
                        "p95_ms": vals["p95"],
                        "p99_ms": vals["p99"],
                        "throughput": vals["throughput"],
                        "acc": vals["acc"],
                        "f1": vals["f1"],
                        "f1_class_0": vals["f1_class_0"],
                        "f1_class_1": vals["f1_class_1"]
                    })

    df = pd.DataFrame(all_rows)
    out_csv = os.path.join(args.output_dir, args.output_csv)
    df.to_csv(out_csv, index=False)
    print(f"\nSaved results to {out_csv}")

    if args.use_wandb and run is not None:
        wandb.save(out_csv)
        wandb.finish()


if __name__ == "__main__":
    main()
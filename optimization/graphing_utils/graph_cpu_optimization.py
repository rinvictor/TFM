import argparse
import os
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
try:
    from scipy.interpolate import griddata
except Exception:
    griddata = None


# -----------------------------
# Helpers
# -----------------------------
def ensure_non_empty(df: pd.DataFrame, msg: str) -> bool:
    """Return True if df is non-empty; otherwise print a message and return False."""
    if df.empty:
        print(f"[WARN] No data matched the filter(s). {msg}")
        return False
    return True


def _set_theme(style: str = "whitegrid", font_scale: float = 1.15):
    """Set a seaborn theme globally."""
    sns.set_theme(style=style, font_scale=font_scale)


def _save_or_show(fig: plt.Figure, save_dir: Optional[str], filename: str, fmt: str = "png", dpi: int = 150, show: bool = True):
    """Save the figure if save_dir is provided, otherwise show it (if show=True)."""
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        out_path = os.path.join(save_dir, f"{filename}.{fmt}")
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        print(f"[INFO] Saved figure: {out_path}")
        plt.close(fig)
    else:
        if show:
            plt.show()
        else:
            plt.close(fig)


def _parse_int_list(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip() != ""]


# -----------------------------
# Plots
# -----------------------------
def plot_latency_batch1(
    df_results: pd.DataFrame,
    save_dir: Optional[str] = None,
    filename: str = "latency_batch1",
    fmt: str = "png",
    dpi: int = 150,
    show: bool = True,
):
    """Average latency (s) per model (batch_size=1)."""
    df_b1 = df_results[df_results["batch_size"] == 1].copy()
    if not ensure_non_empty(df_b1, "Try a different batch size or filters."):
        return
    df_b1["latency_s"] = df_b1["latency_ms"] / 1000.0
    df_b1_group = df_b1.groupby("model_type", as_index=False)["latency_s"].mean().sort_values("latency_s")

    _set_theme(style="whitegrid", font_scale=1.15)
    fig = plt.figure(figsize=(15, 8))
    ax = sns.barplot(
        x="latency_s",
        y="model_type",
        data=df_b1_group,
        hue="model_type",
        legend=False,
        palette="viridis",
    )
    for i, (lat_s, model) in enumerate(zip(df_b1_group["latency_s"], df_b1_group["model_type"])):
        plt.text(lat_s + 0.001, i, f"{lat_s:.3f}s", va="center", fontsize=12, fontweight="bold", color="black")

    plt.xlabel("Average Latency (s)")
    plt.ylabel("Model")
    plt.title("Average Latency by Model (Batch Size = 1)")
    plt.tight_layout()
    _save_or_show(fig, save_dir, filename, fmt, dpi, show)


def plot_latency_all_batches(
    df_results: pd.DataFrame,
    model_type: str,
    annotate_batches: Optional[Sequence[int]] = None,
    save_dir: Optional[str] = None,
    filename: str = "latency_vs_batch",
    fmt: str = "png",
    dpi: int = 150,
    show: bool = True,
):
    """Average latency (s) vs batch size for a given model_type."""
    df = df_results.copy()
    df["latency_s"] = df["latency_ms"] / 1000.0
    df_model = df[df["model_type"] == model_type]
    if not ensure_non_empty(df_model, f"Model type: {model_type}"):
        return
    df_group = df_model.groupby("batch_size", as_index=False)["latency_s"].mean()

    _set_theme(style="whitegrid", font_scale=1.25)
    fig = plt.figure(figsize=(15, 8))
    ax = sns.lineplot(
        data=df_group,
        x="batch_size",
        y="latency_s",
        marker="o",
        color="royalblue",
    )

    annot_set = set(annotate_batches or df_group["batch_size"].unique())
    for _, row in df_group.iterrows():
        if row["batch_size"] in annot_set:
            plt.text(
                row["batch_size"], row["latency_s"] + 0.02, f"{row['latency_s']:.3f}s",
                fontsize=11, color="royalblue", ha="center", va="bottom", fontweight="bold"
            )

    plt.xlabel("Batch Size")
    plt.ylabel("Average Latency (s)")
    plt.title(f"Average Latency vs Batch Size ({model_type})")
    plt.xticks(sorted(df_group["batch_size"].unique()))
    plt.tight_layout()
    _save_or_show(fig, save_dir, filename, fmt, dpi, show)


def plot_throughput_batches(
    df_results: pd.DataFrame,
    model_type: str,
    save_dir: Optional[str] = None,
    filename: str = "throughput_vs_batch",
    fmt: str = "png",
    dpi: int = 150,
    show: bool = True,
):
    """Average throughput (img/s) vs batch size for a given model_type."""
    df = df_results.copy()
    df_model = df[df["model_type"] == model_type]
    if not ensure_non_empty(df_model, f"Model type: {model_type}"):
        return
    df_group = df_model.groupby("batch_size", as_index=False)["throughput"].mean()

    _set_theme(style="whitegrid", font_scale=1.25)
    fig = plt.figure(figsize=(12, 6))
    ax = sns.lineplot(
        data=df_group,
        x="batch_size",
        y="throughput",
        marker="o",
        color="seagreen",
    )
    for _, row in df_group.iterrows():
        plt.text(
            row["batch_size"], row["throughput"] + 1, f"{row['throughput']:.1f}",
            fontsize=11, color="seagreen", ha="center", va="bottom", fontweight="bold"
        )
    plt.xlabel("Batch Size")
    plt.ylabel("Throughput (img/s)")
    plt.title(f"Throughput vs Batch Size ({model_type})")
    plt.xticks(sorted(df_group["batch_size"].unique()))
    plt.tight_layout()
    _save_or_show(fig, save_dir, filename, fmt, dpi, show)


def plot_throughput_heatmap(
    df_results: pd.DataFrame,
    model_type: str = "original",
    prefetch_factor: Optional[int] = None,
    aggfunc: str = "mean",
    cmap: str = "YlGnBu",
    save_dir: Optional[str] = None,
    filename: str = "throughput_heatmap",
    fmt: str = "png",
    dpi: int = 150,
    show: bool = True,
):
    """Heatmap of throughput across batch_size (rows) and num_workers (cols), optionally filtered by prefetch_factor."""
    df = df_results[df_results["model_type"] == model_type].copy()
    if prefetch_factor is not None:
        df = df[df["prefetch_factor"] == prefetch_factor]
    if not ensure_non_empty(df, f"Model: {model_type}, Prefetch: {prefetch_factor}"):
        return

    table = df.pivot_table(index="batch_size", columns="num_workers", values="throughput", aggfunc=aggfunc)
    _set_theme(style="white", font_scale=1.25)
    fig = plt.figure(figsize=(9, 7))
    ax = sns.heatmap(table, annot=True, fmt=".1f", cmap=cmap)
    title_pf = f", Prefetch: {prefetch_factor}" if prefetch_factor is not None else ""
    plt.title(f"Throughput (img/s) by Batch Size and Num Workers ({model_type}{title_pf})")
    plt.ylabel("Batch Size")
    plt.xlabel("Num Workers")
    plt.tight_layout()
    _save_or_show(fig, save_dir, filename, fmt, dpi, show)


def plot_throughput_heatmaps_by_prefetch(
    df_results: pd.DataFrame,
    model_type: str,
    graph_name: Optional[str] = None,
    cmap: str = "YlGnBu",
    save_dir: Optional[str] = None,
    filename: str = "throughput_heatmaps_by_prefetch",
    fmt: str = "png",
    dpi: int = 150,
    show: bool = True,
):
    """Create one heatmap per prefetch_factor for a given model_type."""
    df_model = df_results[df_results["model_type"] == model_type].copy()
    if not ensure_non_empty(df_model, f"Model type: {model_type}"):
        return
    prefetch_vals = sorted(df_model["prefetch_factor"].unique())

    _set_theme(style="white", font_scale=1.2)
    ncols = len(prefetch_vals)
    fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 6), sharey=True)
    if ncols == 1:
        axes = [axes]

    for ax, pf in zip(axes, prefetch_vals):
        df_pf = df_model[df_model["prefetch_factor"] == pf]
        table = df_pf.pivot_table(
            index="batch_size",
            columns="num_workers",
            values="throughput",
            aggfunc="mean"
        )
        sns.heatmap(table, annot=True, fmt=".1f", cmap=cmap, ax=ax)
        ax.set_title(f"prefetch_factor = {pf}")
        ax.set_xlabel("Num Workers")
        ax.set_ylabel("Batch Size")

    if graph_name:
        fig.suptitle(graph_name, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    _save_or_show(fig, save_dir, filename, fmt, dpi, show)


def plot_3d_surface(
    df_results: pd.DataFrame,
    model_type: str = "quantized_pruned_qnnpack",
    prefetch_factor: int = 1,
    title: Optional[str] = None,
    save_dir: Optional[str] = None,
    filename: str = "surface3d",
    fmt: str = "png",
    dpi: int = 150,
    show: bool = True,
):
    """3D surface of throughput vs (batch_size, num_workers) for a given model_type and prefetch_factor."""
    df_sel = df_results[(df_results["model_type"] == model_type) & (df_results["prefetch_factor"] == prefetch_factor)].copy()
    if not ensure_non_empty(df_sel, f"Model: {model_type}, Prefetch: {prefetch_factor}"):
        return

    xs = np.sort(df_sel["batch_size"].unique())
    ys = np.sort(df_sel["num_workers"].unique())
    X, Y = np.meshgrid(xs, ys)
    Z = np.zeros_like(X, dtype=float)
    for i, batch in enumerate(xs):
        for j, worker in enumerate(ys):
            val = df_sel[(df_sel["batch_size"] == batch) & (df_sel["num_workers"] == worker)]["throughput"].mean()
            Z[j, i] = val

    max_idx = np.unravel_index(np.nanargmax(Z), Z.shape)
    sweet_batch = X[max_idx]
    sweet_worker = Y[max_idx]
    sweet_value = Z[max_idx]

    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(X, Y, Z, cmap="viridis", alpha=0.9)
    ax.scatter(sweet_batch, sweet_worker, sweet_value, color="red", s=120, label="Best configuration", zorder=150)
    ax.text(sweet_batch, sweet_worker, sweet_value + 1,
            f"Max: {sweet_value:.1f}\nBatch: {sweet_batch}\nWorkers: {sweet_worker}",
            color="red", weight="bold", fontsize=10, zorder=150)

    ax.set_xlabel("Batch Size")
    ax.set_ylabel("Num Workers")
    ax.set_zlabel("Throughput (img/s)")
    ttl = title or f"Throughput 3D Surface ({model_type}) - Prefetch: {prefetch_factor}"
    ax.set_title(ttl)
    fig.colorbar(surf, shrink=0.5, aspect=6, label="Throughput")
    ax.legend(loc="upper left", bbox_to_anchor=(1.05, 1), fontsize=10)
    plt.tight_layout()
    _save_or_show(fig, save_dir, filename, fmt, dpi, show)


def plot_3d_surface_smooth(
    df_results: pd.DataFrame,
    model_type: str = "quantized_pruned_qnnpack",
    prefetch_factor: int = 1,
    title: Optional[str] = None,
    save_dir: Optional[str] = None,
    filename: str = "surface3d_smooth",
    fmt: str = "png",
    dpi: int = 150,
    show: bool = True,
):
    """Smooth 3D surface using interpolation (requires SciPy)."""
    if griddata is None:
        print("[WARN] SciPy is not available; smooth 3D surface cannot be generated. Install scipy to enable this plot.")
        return

    df_sel = df_results[(df_results["model_type"] == model_type) & (df_results["prefetch_factor"] == prefetch_factor)].copy()
    if not ensure_non_empty(df_sel, f"Model: {model_type}, Prefetch: {prefetch_factor}"):
        return

    xs = np.sort(df_sel["batch_size"].unique())
    ys = np.sort(df_sel["num_workers"].unique())

    points = np.array([(row["batch_size"], row["num_workers"]) for _, row in df_sel.iterrows()])
    values = df_sel["throughput"].values

    xs_dense = np.linspace(xs.min(), xs.max(), 100)
    ys_dense = np.linspace(ys.min(), ys.max(), 100)
    X_dense, Y_dense = np.meshgrid(xs_dense, ys_dense)
    Z_dense = griddata(points, values, (X_dense, Y_dense), method="cubic")

    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(
        X_dense, Y_dense, Z_dense,
        cmap="viridis",
        alpha=1.0,
        linewidth=0,
        antialiased=False,
        shade=False
    )
    surf.set_edgecolor("none")

    max_idx = np.nanargmax(values)
    sweet_batch = df_sel.iloc[max_idx]["batch_size"]
    sweet_worker = df_sel.iloc[max_idx]["num_workers"]
    sweet_value = values[max_idx]

    ax.text(sweet_batch, sweet_worker, sweet_value + 1,
            f"Max: {sweet_value:.1f}\nBatch: {sweet_batch}\nWorkers: {sweet_worker}",
            color="red", weight="bold", fontsize=10, zorder=200)

    ax.set_xlabel("Batch Size")
    ax.set_ylabel("Num Workers")
    ax.set_zlabel("Throughput (img/s)")
    ax.set_title(title or f"Throughput Smooth Surface ({model_type}) - Prefetch: {prefetch_factor}")
    fig.colorbar(surf, shrink=0.5, aspect=6, label="Throughput")
    plt.tight_layout()
    _save_or_show(fig, save_dir, filename, fmt, dpi, show)


def plot_latency_percentiles_custom(
    df_results: pd.DataFrame,
    model_original: str = "original",
    batch_orig: int = 4,
    workers_orig: int = 6,
    prefetch_orig: int = 4,
    model_quantized: str = "torchscript_quant_pruned",
    batch_quant: int = 4,
    workers_quant: int = 6,
    prefetch_quant: int = 4,
    save_dir: Optional[str] = None,
    filename: str = "latency_percentiles",
    fmt: str = "png",
    dpi: int = 150,
    show: bool = True,
):
    """Compare mean, p95 and p99 latency between a chosen 'original' and a 'quantized' (or other) model configuration."""
    df_orig = df_results[
        (df_results["model_type"] == model_original) &
        (df_results["batch_size"] == batch_orig) &
        (df_results["num_workers"] == workers_orig) &
        (df_results["prefetch_factor"] == prefetch_orig)
    ].copy()

    df_quant = df_results[
        (df_results["model_type"] == model_quantized) &
        (df_results["batch_size"] == batch_quant) &
        (df_results["num_workers"] == workers_quant) &
        (df_results["prefetch_factor"] == prefetch_quant)
    ].copy()

    res_list = []
    if not df_orig.empty:
        res = df_orig.iloc[0]
        res_list.append({
            "Model": "Original Model",
            "Mean": res["latency_ms"],
            "P95": res["p95_ms"],
            "P99": res["p99_ms"],
        })
    else:
        print("[WARN] Original model configuration not found.")

    if not df_quant.empty:
        res = df_quant.iloc[0]
        res_list.append({
            "Model": "Quantized Model",
            "Mean": res["latency_ms"],
            "P95": res["p95_ms"],
            "P99": res["p99_ms"],
        })
    else:
        print("[WARN] Quantized model configuration not found.")

    if not res_list:
        print("[WARN] Nothing to plot for latency percentiles.")
        return

    df_summary = pd.DataFrame(res_list)
    df_long = pd.melt(df_summary, id_vars="Model", value_vars=["Mean", "P95", "P99"],
                      var_name="Metric", value_name="Latency (ms)")

    _set_theme(style="whitegrid", font_scale=1.2)
    fig = plt.figure(figsize=(7, 5))
    ax = sns.barplot(
        data=df_long,
        x="Model",
        y="Latency (ms)",
        hue="Metric",
        palette="Set2"
    )
    plt.title("Mean, P95 and P99 Latency\nOriginal vs Quantized")
    plt.xlabel("")
    plt.ylabel("Latency (ms)")
    plt.legend(title="Metric", loc="upper left", bbox_to_anchor=(1.05, 1))
    plt.tight_layout()
    _save_or_show(fig, save_dir, filename, fmt, dpi, show)


def plot_radar_f1_throughput(
    df_results: pd.DataFrame,
    model_original: str = "original",
    batch_orig: int = 4,
    workers_orig: int = 6,
    prefetch_orig: int = 4,
    model_quantized: str = "torchscript_quant_pruned",
    batch_quant: int = 4,
    workers_quant: int = 6,
    prefetch_quant: int = 4,
    save_dir: Optional[str] = None,
    filename: str = "radar_f1_throughput",
    fmt: str = "png",
    dpi: int = 150,
    show: bool = True,
):
    """Radar chart comparing F1 metrics and throughput between two configurations."""
    metrics = ["f1", "f1_class_0", "f1_class_1", "throughput"]
    labels = ["F1 Global", "F1 Benign", "F1 Malignant", "Throughput"]

    df_orig = df_results[
        (df_results["model_type"] == model_original) &
        (df_results["batch_size"] == batch_orig) &
        (df_results["num_workers"] == workers_orig) &
        (df_results["prefetch_factor"] == prefetch_orig)
    ]
    df_quant = df_results[
        (df_results["model_type"] == model_quantized) &
        (df_results["batch_size"] == batch_quant) &
        (df_results["num_workers"] == workers_quant) &
        (df_results["prefetch_factor"] == prefetch_quant)
    ]

    orig = [df_orig.iloc[0][m] if not df_orig.empty else 0 for m in metrics]
    quant = [df_quant.iloc[0][m] if not df_quant.empty else 0 for m in metrics]

    all_vals = np.array([orig, quant])
    maxs = all_vals.max(axis=0)
    maxs_safe = np.where(maxs == 0, 1, maxs)
    orig_n = np.array(orig) / maxs_safe
    quant_n = np.array(quant) / maxs_safe

    orig_n_closed = np.concatenate((orig_n, [orig_n[0]]))
    quant_n_closed = np.concatenate((quant_n, [quant_n[0]]))

    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles_closed = angles + [angles[0]]

    fig, ax = plt.subplots(figsize=(8, 6), subplot_kw=dict(polar=True))

    ax.plot(angles_closed, orig_n_closed, "o-", label="Original", color="#6db5fc")
    ax.plot(angles_closed, quant_n_closed, "o-", label="Quantized", color="#ff9393")
    ax.fill(angles_closed, orig_n_closed, alpha=0.15, color="#6db5fc")
    ax.fill(angles_closed, quant_n_closed, alpha=0.15, color="#ff9393")

    ax.set_yticklabels([])
    ax.set_ylim(0, 1.2)

    ax.set_thetagrids(np.degrees(angles), [""] * len(labels))
    for i, lab in enumerate(labels):
        rot = 90 if i in (0, 2) else 0
        ax.text(angles[i], 1.03, lab,
                transform=ax.get_xaxis_transform(),
                rotation=rot, rotation_mode="anchor",
                ha="center", va="center", fontsize=11,
                clip_on=False, zorder=6)

    def fmt_val(metric, v):
        return f"{v:.1f}" if "throughput" in metric.lower() else f"{v:.3f}"

    delta_deg = 6
    delta = np.deg2rad(delta_deg)
    dr_orig = 0.06
    dr_quant = 0.06

    bbox_kw_o = dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85)
    bbox_kw_q = dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85)

    for i, ang in enumerate(angles):
        r_o = np.clip(orig_n[i] + dr_orig, 0, 1.18)
        r_q = np.clip(quant_n[i] + dr_quant, 0, 1.18)

        if i in (0, 2):
            ang_o = ang - delta
            ang_q = ang + delta
            ax.annotate(fmt_val(metrics[i], orig[i]),
                        xy=(ang, orig_n[i]),
                        xytext=(ang_o, r_o),
                        textcoords="data",
                        ha="center", va="center",
                        color="#2c76c0", fontsize=9, bbox=bbox_kw_o, zorder=7)
            ax.annotate(fmt_val(metrics[i], quant[i]),
                        xy=(ang, quant_n[i]),
                        xytext=(ang_q, r_q),
                        textcoords="data",
                        ha="center", va="center",
                        color="#c02c2c", fontsize=9, bbox=bbox_kw_q, zorder=7)

        elif i == 1:
            r_o_below = np.clip(orig_n[i] - 0.10, 0.02, 1.18)
            r_q_above = np.clip(quant_n[i] + 0.08, 0, 1.18)
            ax.annotate(fmt_val(metrics[i], orig[i]),
                        xy=(ang, orig_n[i]),
                        xytext=(ang, r_o_below),
                        textcoords="data",
                        ha="center", va="center",
                        color="#2c76c0", fontsize=9, bbox=bbox_kw_o, zorder=7)
            ax.annotate(fmt_val(metrics[i], quant[i]),
                        xy=(ang, quant_n[i]),
                        xytext=(ang, r_q_above),
                        textcoords="data",
                        ha="center", va="center",
                        color="#c02c2c", fontsize=9, bbox=bbox_kw_q, zorder=7)

        else:
            r_q2 = np.clip(r_q + 0.03, 0, 1.18)
            ax.annotate(fmt_val(metrics[i], orig[i]),
                        xy=(ang, orig_n[i]),
                        xytext=(ang, r_o),
                        textcoords="data",
                        ha="center", va="center",
                        color="#2c76c0", fontsize=9, bbox=bbox_kw_o, zorder=7)
            ax.annotate(fmt_val(metrics[i], quant[i]),
                        xy=(ang, quant_n[i]),
                        xytext=(ang, r_q2),
                        textcoords="data",
                        ha="center", va="center",
                        color="#c02c2c", fontsize=9, bbox=bbox_kw_q, zorder=7)

    fig.suptitle("F1 and Throughput Trade-off\nOriginal vs Quantized", y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.98])

    plt.legend(loc="upper right", bbox_to_anchor=(1.2, 1))
    _save_or_show(fig, save_dir, filename, fmt, dpi, show)


def main():
    parser = argparse.ArgumentParser(description="Parametrizable plots for benchmark results CSV.")
    parser.add_argument("--csv", type=str, required=True, help="Path to benchmark results CSV.")
    parser.add_argument("--save-dir", type=str, default="", help="Directory to save figures. If empty, figures are shown instead.")
    parser.add_argument("--fmt", type=str, default="png", help="Image format for saved figures (e.g., png, pdf).")
    parser.add_argument("--dpi", type=int, default=150, help="DPI for saved figures.")
    parser.add_argument("--no-show", action="store_true", help="Do not show figures (useful when saving only).")

    # Theme
    parser.add_argument("--theme-style", type=str, default="whitegrid", help="Seaborn theme style.")
    parser.add_argument("--font-scale", type=float, default=1.15, help="Seaborn font scale.")

    # Plots to run
    parser.add_argument("--run-latency-b1", action="store_true", help="Run: average latency per model (batch_size=1).")
    parser.add_argument("--run-latency-vs-batch", action="store_true", help="Run: latency vs batch for one model.")
    parser.add_argument("--latency-model", type=str, default="original", help="Model type to use for latency vs batch.")
    parser.add_argument("--annotate-batches", type=str, default="", help="Batch sizes to annotate, e.g. '1,2,4,8'. Empty = all.")

    parser.add_argument("--run-throughput-vs-batch", action="store_true", help="Run: throughput vs batch for one model.")
    parser.add_argument("--throughput-model", type=str, default="original", help="Model type to use for throughput vs batch.")

    parser.add_argument("--run-throughput-heatmap", action="store_true", help="Run: throughput heatmap.")
    parser.add_argument("--heatmap-model", type=str, default="original", help="Model type for throughput heatmap.")
    parser.add_argument("--heatmap-prefetch", type=int, default=None, help="Prefetch factor to filter in heatmap. None = no filter.")

    parser.add_argument("--run-heatmaps-by-prefetch", action="store_true", help="Run: one throughput heatmap per prefetch.")
    parser.add_argument("--hbpf-model", type=str, default="quantized_pruned_qnnpack", help="Model for heatmaps by prefetch.")
    parser.add_argument("--hbpf-title", type=str, default="Throughput by Batch Size, Num Workers and Prefetch", help="Title for heatmaps by prefetch figure.")

    parser.add_argument("--run-surface3d", action="store_true", help="Run: 3D throughput surface.")
    parser.add_argument("--s3d-model", type=str, default="quantized_pruned_qnnpack", help="Model type for 3D surface.")
    parser.add_argument("--s3d-prefetch", type=int, default=1, help="Prefetch factor for 3D surface.")
    parser.add_argument("--s3d-title", type=str, default="", help="Optional title for 3D surface.")

    parser.add_argument("--run-surface3d-smooth", action="store_true", help="Run: smoothed 3D throughput surface (requires SciPy).")
    parser.add_argument("--s3ds-model", type=str, default="quantized_pruned_qnnpack", help="Model type for smooth 3D surface.")
    parser.add_argument("--s3ds-prefetch", type=int, default=1, help="Prefetch factor for smooth 3D surface.")
    parser.add_argument("--s3ds-title", type=str, default="", help="Optional title for smooth 3D surface.")

    parser.add_argument("--run-latency-percentiles", action="store_true", help="Run: latency percentiles comparison plot.")
    parser.add_argument("--lp-orig-model", type=str, default="original")
    parser.add_argument("--lp-orig-batch", type=int, default=4)
    parser.add_argument("--lp-orig-workers", type=int, default=6)
    parser.add_argument("--lp-orig-prefetch", type=int, default=4)
    parser.add_argument("--lp-quant-model", type=str, default="torchscript_quant_pruned")
    parser.add_argument("--lp-quant-batch", type=int, default=4)
    parser.add_argument("--lp-quant-workers", type=int, default=6)
    parser.add_argument("--lp-quant-prefetch", type=int, default=4)

    parser.add_argument("--run-radar", action="store_true", help="Run: radar plot F1 and throughput.")
    parser.add_argument("--rad-orig-model", type=str, default="original")
    parser.add_argument("--rad-orig-batch", type=int, default=4)
    parser.add_argument("--rad-orig-workers", type=int, default=6)
    parser.add_argument("--rad-orig-prefetch", type=int, default=4)
    parser.add_argument("--rad-quant-model", type=str, default="torchscript_quant_pruned")
    parser.add_argument("--rad-quant-batch", type=int, default=4)
    parser.add_argument("--rad-quant-workers", type=int, default=6)
    parser.add_argument("--rad-quant-prefetch", type=int, default=4)

    args = parser.parse_args()

    # Set theme globally (some plots also set it again with different font scale)
    _set_theme(style=args.theme_style, font_scale=args.font_scale)

    df_results = pd.read_csv(args.csv)

    save_dir = args.save_dir if args.save_dir.strip() else None
    show = not args.no_show

    if args.run_latency_b1:
        plot_latency_batch1(df_results, save_dir, "latency_batch1", args.fmt, args.dpi, show)

    if args.run_latency_vs_batch:
        annot = _parse_int_list(args.annotate_batches) if args.annotate_batches else None
        plot_latency_all_batches(
            df_results,
            model_type=args.latency_model,
            annotate_batches=annot,
            save_dir=save_dir,
            filename=f"latency_vs_batch_{args.latency_model}",
            fmt=args.fmt,
            dpi=args.dpi,
            show=show,
        )

    if args.run_throughput_vs_batch:
        plot_throughput_batches(
            df_results,
            model_type=args.throughput_model,
            save_dir=save_dir,
            filename=f"throughput_vs_batch_{args.throughput_model}",
            fmt=args.fmt,
            dpi=args.dpi,
            show=show,
        )

    if args.run_throughput_heatmap:
        plot_throughput_heatmap(
            df_results,
            model_type=args.heatmap_model,
            prefetch_factor=args.heatmap_prefetch,
            save_dir=save_dir,
            filename=f"throughput_heatmap_{args.heatmap_model}"
                     f"{'' if args.heatmap_prefetch is None else f'_pf{args.heatmap_prefetch}'}",
            fmt=args.fmt,
            dpi=args.dpi,
            show=show,
        )

    if args.run_heatmaps_by_prefetch:
        plot_throughput_heatmaps_by_prefetch(
            df_results,
            model_type=args.hbpf_model,
            graph_name=args.hbpf_title,
            save_dir=save_dir,
            filename=f"throughput_heatmaps_by_prefetch_{args.hbpf_model}",
            fmt=args.fmt,
            dpi=args.dpi,
            show=show,
        )

    if args.run_surface3d:
        plot_3d_surface(
            df_results,
            model_type=args.s3d_model,
            prefetch_factor=args.s3d_prefetch,
            title=(args.s3d_title or None),
            save_dir=save_dir,
            filename=f"surface3d_{args.s3d_model}_pf{args.s3d_prefetch}",
            fmt=args.fmt,
            dpi=args.dpi,
            show=show,
        )

    if args.run_surface3d_smooth:
        plot_3d_surface_smooth(
            df_results,
            model_type=args.s3ds_model,
            prefetch_factor=args.s3ds_prefetch,
            title=(args.s3ds_title or None),
            save_dir=save_dir,
            filename=f"surface3d_smooth_{args.s3ds_model}_pf{args.s3ds_prefetch}",
            fmt=args.fmt,
            dpi=args.dpi,
            show=show,
        )

    if args.run_latency_percentiles:
        plot_latency_percentiles_custom(
            df_results,
            model_original=args.lp_orig_model,
            batch_orig=args.lp_orig_batch,
            workers_orig=args.lp_orig_workers,
            prefetch_orig=args.lp_orig_prefetch,
            model_quantized=args.lp_quant_model,
            batch_quant=args.lp_quant_batch,
            workers_quant=args.lp_quant_workers,
            prefetch_quant=args.lp_quant_prefetch,
            save_dir=save_dir,
            filename=f"latency_percentiles_{args.lp_orig_model}_vs_{args.lp_quant_model}",
            fmt=args.fmt,
            dpi=args.dpi,
            show=show,
        )

    if args.run_radar:
        plot_radar_f1_throughput(
            df_results,
            model_original=args.rad_orig_model,
            batch_orig=args.rad_orig_batch,
            workers_orig=args.rad_orig_workers,
            prefetch_orig=args.rad_orig_prefetch,
            model_quantized=args.rad_quant_model,
            batch_quant=args.rad_quant_batch,
            workers_quant=args.rad_quant_workers,
            prefetch_quant=args.rad_quant_prefetch,
            save_dir=save_dir,
            filename=f"radar_{args.rad_orig_model}_vs_{args.rad_quant_model}",
            fmt=args.fmt,
            dpi=args.dpi,
            show=show,
        )


if __name__ == "__main__":
    main()
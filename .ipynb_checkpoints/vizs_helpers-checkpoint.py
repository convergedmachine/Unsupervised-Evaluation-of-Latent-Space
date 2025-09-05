import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional

DATASETS = [
    'mnist_basic',
    'mnist_rotated',    
    'mnist_background_random',
    'mnist_background_images',
    'fashion_mnist',
    'cifar10',
    'cifar100'
]

from sklearn.exceptions import ConvergenceWarning
ConvergenceWarning('ignore')

def _to_rgb(color):
    return np.array(mcolors.to_rgb(color), dtype=float)

def _blend_to_white(base_rgb: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Blend from white (t=0) to base_rgb (t=1). t shape: (N,)
    Returns (N, 3) RGB array.
    """
    white = np.ones(3, dtype=float)
    # ensure t is in [0,1]
    t = np.clip(t, 0.0, 1.0)
    return (1.0 - t)[:, None] * white + t[:, None] * base_rgb

def plot_accuracy_vs_param_panels(
    parzen_dict: Dict[str, pd.DataFrame],
    random_dict: Dict[str, pd.DataFrame],
    acc_col: str = "accuracy",
    param_col: str = "param",
    color_set: str = "params_nhid1",
    figsize: Tuple[int, int] = (35, 5),
    save_path: Optional[str] = None
):
    datasets = DATASETS  # assumed defined elsewhere

    fig, axes = plt.subplots(1, 7, figsize=figsize, sharey=False)
    axes = axes.flatten()

    # Base colors (matplotlib default blue/orange)
    PARZEN_BASE = _to_rgb("#1f77b4")
    RANDOM_BASE = _to_rgb("#ff7f0e")

    sampler_specs = [
        ("Parzen", parzen_dict, "o", PARZEN_BASE),
        ("Random", random_dict, "^", RANDOM_BASE),
    ]

    # ---- global normalization over color_set so shading is comparable across panels ----
    all_nhid = []
    for _, dct, _, _ in sampler_specs:
        for ds, df in dct.items():
            if {acc_col, param_col, color_set}.issubset(df.columns):
                par = pd.to_numeric(df[param_col], errors="coerce").to_numpy()
                acc = pd.to_numeric(df[acc_col], errors="coerce").to_numpy()
                nh  = pd.to_numeric(df[color_set], errors="coerce").to_numpy()
                mask = np.isfinite(par) & np.isfinite(acc) & np.isfinite(nh)
                if mask.any():
                    all_nhid.append(nh[mask])
    if len(all_nhid):
        stacked = np.concatenate(all_nhid)
        vmin, vmax = float(np.nanmin(stacked)), float(np.nanmax(stacked))
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
            vmin, vmax = 0.0, 1.0  # fallback
    else:
        vmin, vmax = 0.0, 1.0

    def _norm(nh):
        return (nh - vmin) / (vmax - vmin + 1e-12)

    for ax, ds in zip(axes, datasets):
        ax.set_title(ds.replace("_", " "))

        for label, dct, marker, base_rgb in sampler_specs:
            if ds not in dct:
                continue
            df = dct[ds]
            if acc_col not in df.columns or param_col not in df.columns:
                continue

            par = pd.to_numeric(df[param_col], errors="coerce").to_numpy()
            acc = pd.to_numeric(df[acc_col], errors="coerce").to_numpy()
            mask = np.isfinite(par) & np.isfinite(acc)

            colors = None
            if color_set in df.columns:
                nh = pd.to_numeric(df[color_set], errors="coerce").to_numpy()
                mask = mask & np.isfinite(nh)
                t = _norm(nh[mask])  # 0..1
                colors = _blend_to_white(base_rgb, t)  # (N,3)

            x, y = par[mask], acc[mask]

            ax.set_xlabel("ρ-Parameter")
            ax.set_ylabel("Accuracy" if "val_recon_loss" not in acc_col else "Recon Loss")
            #ax.set_xlim(left=0, right=0.015)

            ax.scatter(
                x, y,
                alpha=0.9, s=24, marker=marker,
                label=label,
                c=colors if colors is not None else base_rgb[None, :],
                edgecolors="none"
            )

        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9, loc="best", framealpha=0.9)

    for j in range(len(datasets), len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from typing import Dict, List, Optional, Tuple
from scipy import stats

# ----------------------- utilities -----------------------

def _to_rgb(color_str: str) -> np.ndarray:
    return np.array(mcolors.to_rgb(color_str), dtype=float)

def _blend_to_white(base_rgb: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Blend from white (t=0) to base_rgb (t=1). t shape: (N,)
    Returns (N, 3) RGB array.
    """
    t = np.clip(np.asarray(t, dtype=float), 0.0, 1.0)
    white = np.ones(3, dtype=float)
    return (1.0 - t)[:, None] * white + t[:, None] * base_rgb

def _extract_xy(df: pd.DataFrame, nhid_col: str, acc_col: str) -> Tuple[np.ndarray, np.ndarray]:
    x = pd.to_numeric(df[nhid_col], errors="coerce").to_numpy()
    y = pd.to_numeric(df[acc_col], errors="coerce").to_numpy()
    m = np.isfinite(x) & np.isfinite(y)
    return x[m], y[m]

def _corr_and_fit(x: np.ndarray, y: np.ndarray) -> dict:
    if x.size < 3:
        return dict(pearson_r=np.nan, pearson_p=np.nan,
                    spearman_r=np.nan, spearman_p=np.nan,
                    slope=np.nan, intercept=np.nan, n=int(x.size))
    pr = stats.pearsonr(x, y)
    sr = stats.spearmanr(x, y)
    slope, intercept, _, _, _ = stats.linregress(x, y)
    return dict(
        pearson_r=float(getattr(pr, "statistic", pr[0])),
        pearson_p=float(getattr(pr, "pvalue", pr[1])),
        spearman_r=float(getattr(sr, "correlation", sr[0])),
        spearman_p=float(getattr(sr, "pvalue", sr[1])),
        slope=float(slope),
        intercept=float(intercept),
        n=int(x.size),
    )

# ----------------------- analysis table -----------------------

def analyze_nhid1_vs_value(
    parzen_dict: Dict[str, pd.DataFrame],
    random_dict: Dict[str, pd.DataFrame],
    acc_col: str = "value",
    nhid_col: str = "params_nhid1",
    datasets: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Build a tidy table with Pearson/Spearman correlations and OLS slopes
    per dataset and overall, for both samplers.
    """
    rows = []
    for sampler_name, dct in [("Parzen", parzen_dict), ("Random", random_dict)]:
        keys = datasets if datasets is not None else list(dct.keys())

        all_x, all_y = [], []
        for ds in keys:
            if ds not in dct or acc_col not in dct[ds].columns or nhid_col not in dct[ds].columns:
                continue
            x, y = _extract_xy(dct[ds], nhid_col, acc_col)
            stats_row = _corr_and_fit(x, y)
            stats_row.update({"sampler": sampler_name, "dataset": ds})
            rows.append(stats_row)
            if x.size:
                all_x.append(x); all_y.append(y)

        # overall across datasets
        if len(all_x):
            X = np.concatenate(all_x); Y = np.concatenate(all_y)
            stats_row = _corr_and_fit(X, Y)
            stats_row.update({"sampler": sampler_name, "dataset": "__OVERALL__"})
            rows.append(stats_row)

    return pd.DataFrame(rows).sort_values(["sampler", "dataset"]).reset_index(drop=True)

# ----------------------- plotting -----------------------

def plot_nhid1_vs_value_panels(
    parzen_dict: Dict[str, pd.DataFrame],
    random_dict: Dict[str, pd.DataFrame],
    datasets: List[str] = DATASETS,
    acc_col: str = "value",
    nhid_col: str = "params_nhid1",
    n_cols: int = 7,
    figsize: Tuple[int, int] = (35, 5),
    save_path: Optional[str] = None,
    show_regression: bool = True,
    annotate_corr: bool = True,
):
    """
    Multi-panel plot: x = params_nhid1, y = acc_col.
    Parzen (blue o) & Random (orange ^) with degradé shading from white to base color.
    """
    # base colors (matplotlib default blue/orange)
    PARZEN_BASE = _to_rgb("#1f77b4")
    RANDOM_BASE = _to_rgb("#ff7f0e")

    # normalize nhid globally over both samplers so shading is comparable
    all_nhid = []
    for dct in (parzen_dict, random_dict):
        for ds in datasets:
            if ds in dct and nhid_col in dct[ds].columns:
                all_nhid.append(pd.to_numeric(dct[ds][nhid_col], errors="coerce").to_numpy())
    if len(all_nhid):
        stacked = np.concatenate(all_nhid)
        vmin = float(np.nanmin(stacked)) if np.isfinite(np.nanmin(stacked)) else 0.0
        vmax = float(np.nanmax(stacked)) if np.isfinite(np.nanmax(stacked)) else 1.0
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
            vmin, vmax = 0.0, 1.0
    else:
        vmin, vmax = 0.0, 1.0

    def _norm(z):
        return (z - vmin) / (vmax - vmin + 1e-12)

    # layout
    n = len(datasets)
    n_cols = min(n_cols, n) if n > 0 else 1
    n_rows = (n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(figsize[0], max(figsize[1], 5*n_rows)), squeeze=False)

    # iterate datasets
    for i, ds in enumerate(datasets):
        ax = axes[i // n_cols, i % n_cols]
        ax.set_title(ds.replace("_", " "))

        # helper to scatter per sampler
        def scatter_sampler(dct, base_rgb, marker, label):
            if ds not in dct or acc_col not in dct[ds].columns or nhid_col not in dct[ds].columns:
                return None, None, None
            df = dct[ds]
            x, y = _extract_xy(df, nhid_col, acc_col)
            if x.size == 0:
                return None, None, None
            t = _norm(x)
            colors = _blend_to_white(base_rgb, t)
            ax.scatter(x, y, c=colors, marker=marker, s=30, alpha=0.9, edgecolors="none", label=label)

            # regression
            fit = None
            if show_regression and x.size >= 2:
                slope, intercept, _, _, _ = stats.linregress(x, y)
                xx = np.linspace(np.nanmin(x), np.nanmax(x), 100)
                yy = slope * xx + intercept
                ax.plot(xx, yy, color=base_rgb, linewidth=2)
                fit = (slope, intercept)

            # correlations
            metrics = _corr_and_fit(x, y)
            return metrics, fit, (x, y)

        parzen_metrics, parzen_fit, _ = scatter_sampler(parzen_dict, PARZEN_BASE, "o", "Parzen")
        random_metrics, random_fit, _ = scatter_sampler(random_dict, RANDOM_BASE, "^", "Random")

        ax.set_xlabel("params_nhid1")
        ax.set_ylabel(acc_col.capitalize())
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9, loc="best", framealpha=0.9)

        if annotate_corr:
            y_anchor = 0.98
            if parzen_metrics is not None:
                ax.text(0.02, y_anchor,
                        f"Parzen r={parzen_metrics['pearson_r']:.2f}  ρ={parzen_metrics['spearman_r']:.2f}",
                        transform=ax.transAxes, fontsize=9, va="top", ha="left",
                        bbox=dict(boxstyle="round,pad=0.2", alpha=0.15))
                y_anchor -= 0.11
            if random_metrics is not None:
                ax.text(0.02, y_anchor,
                        f"Random r={random_metrics['pearson_r']:.2f}  ρ={random_metrics['spearman_r']:.2f}",
                        transform=ax.transAxes, fontsize=9, va="top", ha="left",
                        bbox=dict(boxstyle="round,pad=0.2", alpha=0.15))

    # hide empty axes
    for j in range(i + 1, n_rows * n_cols):
        axes[j // n_cols, j % n_cols].axis("off")

    plt.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()

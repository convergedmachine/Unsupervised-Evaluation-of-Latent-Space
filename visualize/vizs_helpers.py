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
    return (1.0 - t)[:, None] * white + t[:, None] * base_rgb

def plot_accuracy_vs_param_panels(
    parzen_dict: Dict[str, pd.DataFrame],
    random_dict: Dict[str, pd.DataFrame],
    acc_col: str = "accuracy",
    param_col: str = "param",
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
            if {acc_col, param_col, "params_nhid1"}.issubset(df.columns):
                par = pd.to_numeric(df[param_col], errors="coerce").to_numpy()
                acc = pd.to_numeric(df[acc_col], errors="coerce").to_numpy()
                nh = pd.to_numeric(df["params_nhid1"], errors="coerce").to_numpy()
                if "params_nhid2" in df.columns:
                    nh2 = pd.to_numeric(df["params_nhid2"], errors="coerce").to_numpy()
                    nh = nh * nh2
                if "params_nhid3" in df.columns:
                    nh3 = pd.to_numeric(df["params_nhid3"], errors="coerce").to_numpy()
                    nh += nh2 * nh3
                if "params_ncode" in df.columns:
                    ncode = pd.to_numeric(df["params_ncode"], errors="coerce").to_numpy()
                    nh += ncode * nh3                
                mask = np.isfinite(par) & np.isfinite(acc) & np.isfinite(nh)
                if mask.any():
                    all_nhid.append(nh)
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

            colors = None
            if "params_nhid1" in df.columns:
                nh = pd.to_numeric(df["params_nhid1"], errors="coerce").to_numpy()
                if "params_nhid2" in df.columns:
                    nh2 = pd.to_numeric(df["params_nhid2"], errors="coerce").to_numpy()
                    nh = nh * nh2
                if "params_nhid3" in df.columns:
                    nh3 = pd.to_numeric(df["params_nhid3"], errors="coerce").to_numpy()
                    nh += nh2 * nh3
                if "params_ncode" in df.columns:
                    ncode = pd.to_numeric(df["params_ncode"], errors="coerce").to_numpy()
                    nh += ncode * nh3                    
                t = _norm(nh)  # 0..1
                colors = _blend_to_white(base_rgb, t)

            x, y = par, acc

            ax.set_xlabel("ρ-Parameter")
            ax.set_ylabel("Accuracy" if "val_recon_loss" not in acc_col else "Recon Loss")
            ax.set_xlim(left=0, right=0.015)

            ax.scatter(
                x, y,
                alpha=0.9, s=64, marker=marker,
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

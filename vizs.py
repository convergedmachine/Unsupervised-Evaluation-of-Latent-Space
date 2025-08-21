import os
import pandas as pd
from matplotlib.ticker import FuncFormatter
from typing import Dict, List, Tuple, Optional, Iterable
import math
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------
# Global config
# ---------------------------
DATASETS = [
    'mnist_background_images',
    'mnist_rotated',
    'rectangles',
    'rectangles_images',
    'mnist_rotated_background_images',
    'mnist_background_random',
    'convex',
    'mnist_basic'
]

# ---------------------------
# Utilities
# ---------------------------
def best_of_k_statistics(values: np.ndarray, ks: Iterable[int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute mean and 95% CI of best-of-k across contiguous blocks.
    Returns (k_list, mean_best, ci95).
    """
    k_list, mean_best, ci95 = [], [], []
    n = len(values)
    if n == 0:
        return np.array([]), np.array([]), np.array([])
    for k in ks:
        if k > n:
            continue
        n_blocks = n // k
        if n_blocks == 0:
            continue
        block_bests = [values[i*k:(i+1)*k].max() for i in range(n_blocks)]
        block_bests = np.array(block_bests, dtype=float)
        mu = float(block_bests.mean())
        
        # Standard error of the mean across blocks
        sem = block_bests.std(ddof=1) / math.sqrt(max(1, len(block_bests)))
        ci = 1.96 * sem
        k_list.append(k)
        mean_best.append(mu)
        ci95.append(ci)
    return np.array(k_list), np.array(mean_best), np.array(ci95)

def plot_efficiency_curves_best_of_N(parzen_dict, grid_dict, random_dict,
                                     exp_ks=(1,2,4,8,16,32,64)):
    fig, axs = plt.subplots(2, 4, figsize=(20, 10), sharey=True)
    axs = axs.ravel()

    styles = {
        "Parzen": {"color": "tab:blue", "marker": "o"},
        "Grid": {"color": "tab:orange", "marker": "s"},
        "Random": {"color": "tab:green", "marker": "^"},
    }

    for i, ds in enumerate(DATASETS):
        ax = axs[i]
        for label, key in [("Parzen","parzen"), ("Grid","grid"), ("Random","random")]:
            dct = {"parzen": parzen_dict, "grid": grid_dict, "random": random_dict}[key]
            if ds not in dct: 
                continue
            df = dct[ds]
            vals = df["value"].to_numpy(dtype=float) if not df.empty else np.array([])
            K, mean_best, ci95 = best_of_k_statistics(vals, exp_ks)

            if len(K) == 0:
                continue

            # Plot mean line
            ax.plot(K, mean_best, label=label,
                    color=styles[label]["color"],
                    marker=styles[label]["marker"],
                    linewidth=1.8, markersize=4)

            # Shaded CI band
            ax.fill_between(K,
                            mean_best - ci95,
                            mean_best + ci95,
                            color=styles[label]["color"],
                            alpha=0.2)

        ax.set_xscale("log", base=2)
        ax.set_xticks(list(exp_ks))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda val, pos: f"{int(val)}"))
        ax.set_title(ds, fontsize=12)
        ax.set_xlabel("Budget K (best-of-K)")
        if i % 4 == 0:
            ax.set_ylabel("Mean best (±95% CI)")
        ax.grid(True, which="both", alpha=0.25, zorder=0)
        ax.legend(fontsize=9, frameon=False, loc="lower right")

    fig.suptitle("Efficiency curves (best-of-K) with 95% CI", fontsize=16, y=0.98)
    fig.tight_layout()
    out_path = os.path.join("eff_curves_2x4_panels.png")
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


import os
import math
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import ARDRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted
from sklearn.utils import resample
import glob

from sklearn.exceptions import ConvergenceWarning
ConvergenceWarning('ignore')

# ----------------------------
# Globals
# ----------------------------
DATASETS = [
    'mnist_background_images',
    'mnist_rotated',
    'rectangles',
    'rectangles_images',
    'mnist_rotated_background_images',
    'mnist_background_random',
    'convex',
    'mnist_basic'
]

# Hyperparameter groups -> columns to look for (first found columns are used).
# (You can add/rename here if your CSVs use different param names.)
HYPER_GROUPS = {
    'h.u.':  dict(names=['params_nhid1'],                                  full='n. hidden units',   color='#777777'),
    'a.f.':  dict(names=['params_squash'],                                  full='activation fn.',    color='#E69F00'),
    'w.a.':  dict(names=['params_dist1'],                                   full='initial W algo.',   color='#9467BD'),
    'w.n.':  dict(names=['params_scale_heur1', 'params_scale_mult1'],       full='initial W norm',    color='#1F77B4'),
    'l.r.':  dict(names=['params_lr'],                                      full='learning rate',     color='#D62728'),
}

# Fallback list of budgets for efficiency curves
BUDGETS_DEFAULT = [1, 2, 4, 8, 16, 32, 64]

# ----------------------------
# ARD relevance (Bayesian linear with bootstrapping)
# ----------------------------
def _build_encoder(df: pd.DataFrame, hp_cols: List[str]) -> Tuple[ColumnTransformer, List[str], List[str]]:
    """
    Build a ColumnTransformer that one-hot encodes categorical/bool and scales numeric.
    Returns (ct, numeric_cols, categorical_cols)
    """
    num_cols, cat_cols = [], []
    for c in hp_cols:
        if c not in df.columns:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            num_cols.append(c)
        else:
            cat_cols.append(c)

    transformers = []
    if cat_cols:
        transformers.append(("cat", OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols))
    if num_cols:
        transformers.append(("num", StandardScaler(), num_cols))

    ct = ColumnTransformer(transformers, remainder='drop', sparse_threshold=0.0)
    return ct, num_cols, cat_cols


def _transformed_feature_groups(ct: ColumnTransformer, num_cols: List[str], cat_cols: List[str]) -> Tuple[List[str], List[str]]:
    """
    After ct is fit, recover transformed feature names and the group-key for each transformed column
    (group-key is based on original hyperparameter column's group abbreviation like 'h.u.', 'a.f.', etc.)

    Returns (feat_names, feat_groups) of equal length.
    """
    feat_names: List[str] = []
    feat_groups: List[str] = []

    # helper: map base column name -> group key
    base_to_group = {}
    for key, spec in HYPER_GROUPS.items():
        for nm in spec['names']:
            base_to_group[nm] = key

    # cat features (one-hot)
    if 'cat' in dict(ct.named_transformers_):
        ohe: OneHotEncoder = ct.named_transformers_['cat']
        try:
            cat_names = list(ohe.get_feature_names_out(ct.transformers_[0][2]))
        except Exception:
            # fallback names (not ideal)
            cat_names = []
            for base, cats in zip(ct.transformers_[0][2], ohe.categories_):
                for cat in cats:
                    cat_names.append(f"{base}_{cat}")
        for name in cat_names:
            # original base column is before the last underscore that separates OHE category
            base = name.split('_')[0]
            # Some names keep the full base as-is; ensure we map correctly if base not found:
            if base not in base_to_group:
                # try to find the longest matching base among known ones
                matches = [b for b in base_to_group if name.startswith(b)]
                base = max(matches, key=len) if matches else base
            feat_names.append(name)
            feat_groups.append(base_to_group.get(base, base))  # default to base if not mapped

    # numeric features
    if 'num' in dict(ct.named_transformers_):
        for base in ct.transformers_[1][2] if len(ct.transformers_) > 1 else []:
            feat_names.append(base)
            feat_groups.append(base_to_group.get(base, base))

    return feat_names, feat_groups


def _compute_ard_relevance_distributions(df: pd.DataFrame, hp_cols: List[str], n_boot: int = 64, frac: float = 0.8,
                                         random_state: int = 42) -> Dict[str, List[float]]:
    """
    Bootstrap ARDRegression to get a distribution of group-wise relevances.
    Returns dict: group_key -> list of relevance values
    """
    rng = np.random.RandomState(random_state)

    # Build encoder
    ct, num_cols, cat_cols = _build_encoder(df, hp_cols)

    # Fit once to learn shapes and feature name mapping
    X = df[hp_cols]
    y = df['value'].astype(float).to_numpy()
    X_t = ct.fit_transform(X)
    feat_names, feat_groups = _transformed_feature_groups(ct, num_cols, cat_cols)
    n_features = X_t.shape[1]

    # Storage
    group_values = {k: [] for k in HYPER_GROUPS.keys()}

    # Bootstrap
    for b in range(n_boot):
        # sample rows with replacement (bootstrap)
        boot_idx = rng.randint(0, len(df), size=max(8, int(len(df) * frac)))
        Xb = X.iloc[boot_idx]
        yb = y[boot_idx]
        Xb_t = ct.transform(Xb)

        # ARDRegression on transformed features
        ard = ARDRegression(max_iter=300, tol=1e-3, fit_intercept=True, compute_score=False)
        try:
            ard.fit(Xb_t, yb)
        except Exception:
            # If singular or numeric issues, skip this bootstrap
            continue

        # lambda_ are precisions per weight; larger => more shrinkage => less relevant
        check_is_fitted(ard)
        if ard.lambda_.shape[0] != n_features:
            # dimension mismatch; skip
            continue

        feature_relevance = 1.0 / np.sqrt(np.maximum(1e-12, ard.lambda_))  # ~ "1/length-scale" analogue

        # Aggregate to hyperparameter groups (max across one-hot columns)
        for feat_val, grp in zip(feature_relevance, feat_groups):
            if grp in group_values:
                group_values[grp].append(float(feat_val))

    # Ensure all groups exist (possibly empty lists)
    for k in HYPER_GROUPS.keys():
        group_values.setdefault(k, [])
    return group_values


def _collect_pooled_df(parzen_df: Optional[pd.DataFrame], grid_df: Optional[pd.DataFrame], random_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Concatenate available frames (ignore None)."""
    frames = [d for d in [parzen_df, grid_df, random_df] if d is not None]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)

def _select_hp_columns(df: pd.DataFrame) -> List[str]:
    """Return available hyperparameter columns from HYPER_GROUPS in `df` (first match per group)."""
    cols = []
    for grp in HYPER_GROUPS.values():
        for name in grp['names']:
            if name in df.columns:
                cols.append(name)
                break  # only first matching name per group
    # ensure uniqueness and keep order
    seen, uniq = set(), []
    for c in cols:
        if c not in seen:
            seen.add(c)
            uniq.append(c)
    return uniq

def plot_ard_panels(
    parzen_dict: Dict[str, pd.DataFrame],
    grid_dict: Dict[str, pd.DataFrame],
    random_dict: Dict[str, pd.DataFrame],
    n_boot: int = 64
) -> Tuple[str, str]:
    """
    Build a 2x4 panel of ARD relevance boxplots (one panel per dataset), pooling trials
    from Parzen, Grid, and Random for stronger estimates (matches the style of your figure).
    Also writes a separate legend figure.

    Returns (panel_path, legend_path).
    """
    # Prepare figure
    fig, axs = plt.subplots(2, 4, figsize=(20, 10), sharex=True)
    axs = axs.flatten()

    # Order of groups to show on y-axis (from top to bottom)
    group_order = list(HYPER_GROUPS.keys())
    group_colors = [HYPER_GROUPS[k]['color'] for k in group_order]

    for i, dataset in enumerate(DATASETS):
        ax = axs[i]
        ax.set_title(dataset, fontsize=12)

        # Combine samplers
        p_df = parzen_dict.get(dataset)
        g_df = grid_dict.get(dataset)
        r_df = random_dict.get(dataset)
        df_all = _collect_pooled_df(p_df, g_df, r_df)
        if df_all.empty:
            ax.text(0.5, 0.5, "No data", ha='center', va='center')
            ax.set_yticks(range(len(group_order)))
            ax.set_yticklabels(group_order)
            continue

        hp_cols = _select_hp_columns(df_all)
        if len(hp_cols) == 0:
            ax.text(0.5, 0.5, "No recognized hyperparameter columns", ha='center', va='center')
            ax.set_yticks(range(len(group_order)))
            ax.set_yticklabels(group_order)
            continue

        # Compute ARD relevance distributions
        group_to_vals = _compute_ard_relevance_distributions(df_all, hp_cols, n_boot=n_boot)

        # Gather data in plotting order; ensure non-empty lists
        data = [group_to_vals.get(k, []) for k in group_order]

        # Horizontal boxplots per group
        bp = ax.boxplot(
            data,
            vert=False,
            patch_artist=True,
            widths=0.6,
            showfliers=False  # matches your figure (no black circles)
        )

        # Color styling
        for j, (box, whiskers, caps, med) in enumerate(zip(
            bp['boxes'],
            zip(bp['whiskers'][0::2], bp['whiskers'][1::2]),
            zip(bp['caps'][0::2], bp['caps'][1::2]),
            bp['medians']
        )):
            col = group_colors[j]
            box.set(edgecolor=col, facecolor='none', linewidth=1.8)
            for w in whiskers:
                w.set(color=col, linewidth=1.2, linestyle='--', dashes=(4, 4))
            for c in caps:
                c.set(color=col, linewidth=1.2)
            med.set(color=col, linewidth=2.0)

        ax.set_yticks(np.arange(1, len(group_order) + 1))
        ax.set_yticklabels(group_order, fontsize=11)
        ax.set_xlabel('relevance (1 / length scale)')
        ax.grid(True, axis='x', alpha=0.25)

        # Optional: robust x-limits using Tukey whiskers across all groups (Q1 - .5*IQR to Q3 + 2.5*IQR)
        all_vals = np.concatenate([np.array(v, dtype=float) for v in data if len(v) > 0]) if any(len(v) > 0 for v in data) else np.array([0.0])
        if all_vals.size > 1:
            q1, q3 = np.percentile(all_vals, [25, 75])
            iqr = q3 - q1
            lower = max(0.0, q1 - 0.5 * iqr)
            upper = q3 + 2.5 * iqr
            if upper <= lower:
                upper = lower + 1.0
            ax.set_xlim(lower, upper)

    fig.tight_layout()
    panel_path = os.path.join("ard_2x4_panels.png")
    fig.savefig(panel_path, dpi=200)
    plt.close(fig)

# ------------- helpers -------------
def _ensure_xy(df: pd.DataFrame,
               acc_col: str = "accuracy",
               param_col: str = "param") -> Tuple[np.ndarray, np.ndarray]:
    """Return (x=param, y=accuracy) from a DF with flexible column names."""
    cols_lower = [c.lower() for c in df.columns]

    # Direct hit
    if acc_col in df.columns and param_col in df.columns:
        acc = df[acc_col].astype(float).to_numpy()
        par = df[param_col].astype(float).to_numpy()
        mask = np.isfinite(acc) & np.isfinite(par)
        return par[mask], acc[mask]

    # Heuristics
    acc_candidates = [c for c in df.columns if "acc" in c.lower()]
    par_candidates  = [c for c in df.columns if "param" in c.lower() or "lambda" in c.lower() or "alpha" in c.lower()]
    if acc_candidates and par_candidates:
        acc = df[acc_candidates[0]].astype(float).to_numpy()
        par = df[par_candidates[0]].astype(float).to_numpy()
        mask = np.isfinite(acc) & np.isfinite(par)
        return par[mask], acc[mask]

    # Fallback: first two numeric columns
    numeric = df.select_dtypes(include=[np.number])
    if numeric.shape[1] >= 2:
        par = numeric.iloc[:, 1].to_numpy()
        acc = numeric.iloc[:, 0].to_numpy()
        mask = np.isfinite(acc) & np.isfinite(par)
        return par[mask], acc[mask]

    raise ValueError("Could not infer 'accuracy' and 'param' columns from the DataFrame.")

def _pearson_spearman(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    if len(x) < 2:
        return (np.nan, np.nan)
    r = float(np.corrcoef(x, y)[0, 1])
    rho = pd.Series(y).rank().corr(pd.Series(x).rank(), method="pearson")
    return r, float(rho)

# ------------- main plotter -------------
def plot_accuracy_vs_param_panels(
    parzen_dict: Dict[str, pd.DataFrame],
    grid_dict: Dict[str, pd.DataFrame],
    random_dict: Dict[str, pd.DataFrame],
    datasets: Optional[List[str]] = None,
    acc_col: str = "accuracy",
    param_col: str = "param",
    add_fit_lines: bool = True,
    figsize: Tuple[int, int] = (18, 9),
    save_path: Optional[str] = None,
    plot_mode: str = "scatter",  # "scatter" or "bland_altman"
):
    """
    Create 8 Accuracy-vs-Parameter panels or Bland–Altman plots.

    plot_mode = "scatter": overlay Parzen/Grid/Random with correlations.
    plot_mode = "bland_altman": show Bland–Altman agreement plots.
    """
    if datasets is None:
        datasets = DATASETS

    fig, axes = plt.subplots(2, 4, figsize=figsize, sharey=False)
    axes = axes.flatten()

    sampler_specs = [
        ("Parzen", parzen_dict, "o"),
        ("Grid",   grid_dict,   "s"),
        ("Random", random_dict, "x"),
    ]

    records = []

    for ax, ds in zip(axes, datasets):
        ax.set_title(ds.replace("_", " "))

        for label, dct, marker in sampler_specs:
            if ds not in dct or dct[ds] is None or len(dct[ds]) == 0:
                continue

            df = dct[ds]
            try:
                x, y = _ensure_xy(df, acc_col=acc_col, param_col=param_col)
            except Exception:
                continue

            if len(x) == 0:
                continue

            if plot_mode == "scatter":
                # Correlations
                r, rho = _pearson_spearman(x, y)
                records.append({"dataset": ds, "sampler": label,
                                "pearson_r": r, "spearman_rho": rho, "r2": r**2})

                ax.set_xlabel("Parameter") # "Max Off-diagonal")
                ax.set_ylabel("Accuracy")
                ax.scatter(x, y, alpha=0.8, s=22, marker=marker,
                           label=f"{label} (r={r:.2f}, ρ={rho:.2f})")
                if add_fit_lines and len(x) >= 2:
                    m, b = np.polyfit(x, y, 1)
                    ax.plot(np.sort(x), m*np.sort(x)+b, linewidth=1.6)

            elif plot_mode == "bland_altman":
                # Bland–Altman
                A = y
                B = x
                mean_vals = (A + B) / 2
                diff_vals = A - B
                mean_diff = diff_vals.mean()
                sd_diff = diff_vals.std(ddof=1)

                ax.set_xlabel("Mean of Accuracy and Param")
                ax.set_ylabel("Difference (Accuracy - Param)")
                ax.scatter(mean_vals, diff_vals, alpha=0.6, marker=marker, label=label)
                ax.axhline(mean_diff, color="gray", linestyle="--")
                ax.axhline(mean_diff+1.96*sd_diff, color="red", linestyle="--")
                ax.axhline(mean_diff-1.96*sd_diff, color="red", linestyle="--")

        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="best", framealpha=0.9)

    for j in range(len(datasets), len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")

    corr_df = pd.DataFrame.from_records(records).sort_values(["dataset","sampler"]) if records else None
    return fig, corr_df

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple

# ---------------- I/O ----------------
def load_histories_twohidden(root="../records/two_hidden_mlp_records/histories", dataset="cifar100"):
    """
    Loader for the PREVIOUS program's outputs under:
        <root>/<dataset>/h1_<H1>_h2_<H2>/*.json

    Returns:
      Dict[(h1, h2)] -> List[run_json_dict]
    """
    hist_dir = os.path.join(root, dataset)
    data: Dict[Tuple[int, int], List[dict]] = {}
    if not os.path.isdir(hist_dir):
        return data

    for conf in sorted(os.listdir(hist_dir)):
        if not conf.startswith("h1_"):
            continue
        try:
            parts = conf.split("_")
            # expected: ["h1", "<H1>", "h2", "<H2>"]
            h1 = int(parts[1]); h2 = int(parts[3])
        except Exception:
            continue

        conf_dir = os.path.join(hist_dir, conf)
        if not os.path.isdir(conf_dir):
            continue

        runs = []
        for f in sorted(os.listdir(conf_dir)):
            if f.endswith(".json"):
                with open(os.path.join(conf_dir, f), "r") as fp:
                    runs.append(json.load(fp))
        if runs:
            data[(h1, h2)] = runs
    return data

# ---------------- helpers: extract & stack ----------------
def _val_from_run(run: dict) -> np.ndarray:
    """Return per-epoch Val Acc. trace (NaN-safe)."""
    return np.asarray(run.get("val_hist", []), dtype=float)

def _ed_c12_from_run(run: dict) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return per-epoch ED traces for C1 and C2 with NaN-safe handling and time alignment.
    If missing, return empty arrays.
    """
    ed = run.get("ed_hist", {}) or {}
    c1 = np.asarray(ed.get("C1", []), dtype=float)
    c2 = np.asarray(ed.get("C2", []), dtype=float)

    lengths = [len(a) for a in (c1, c2) if len(a)]
    if lengths:
        L = min(lengths)
        c1 = c1[:L] if len(c1) else np.full(L, np.nan)
        c2 = c2[:L] if len(c2) else np.full(L, np.nan)
    else:
        c1 = np.array([], dtype=float)
        c2 = np.array([], dtype=float)
    return c1, c2

def _stack_runs_timealigned_val(runs: List[dict]):
    """Stack Val Acc. across repetitions; return (mean, std, L)."""
    vals = []
    for r in runs:
        v = _val_from_run(r)
        if v.size and np.isfinite(v).any():
            vals.append(v)
    if not vals:
        return None, None, 0
    L = min(map(len, vals))
    A = np.stack([s[:L] for s in vals], axis=0)
    return np.nanmean(A, axis=0), np.nanstd(A, axis=0), L

def _stack_runs_timealigned_ed(runs: List[dict]):
    """
    Stack C1/C2 ED traces across repetitions;
    return (c1_mean, c1_std, c2_mean, c2_std, L).
    """
    c1_list, c2_list = [], []
    for r in runs:
        c1, c2 = _ed_c12_from_run(r)
        if c1.size and np.isfinite(c1).any(): c1_list.append(c1)
        if c2.size and np.isfinite(c2).any(): c2_list.append(c2)
    if not (c1_list or c2_list):
        return None, None, None, None, 0

    def min_len(lst_list: List[np.ndarray]) -> Optional[int]:
        return min(map(len, lst_list)) if lst_list else None

    lengths = [L for L in [min_len(c1_list), min_len(c2_list)] if L is not None]
    L = min(lengths) if lengths else 0
    if L == 0:
        return None, None, None, None, 0

    if not c1_list: c1_list = [np.full(L, np.nan)]
    if not c2_list: c2_list = [np.full(L, np.nan)]

    A1 = np.stack([s[:L] for s in c1_list], axis=0)
    A2 = np.stack([s[:L] for s in c2_list], axis=0)

    return (
        np.nanmean(A1, axis=0), np.nanstd(A1, axis=0),
        np.nanmean(A2, axis=0), np.nanstd(A2, axis=0),
        L
    )

# ---------------- global ranges (optional auto scaling) ----------------
def _compute_global_ranges(histories: Dict[Tuple[int, int], List[dict]], average_reps: bool):
    """
    Compute global (vmin, vmax, emin, emax) across all (h1,h2).
    If no data found, return sensible defaults.
    """
    vmins, vmaxs, emins, emaxs = [], [], [], []
    for runs in histories.values():
        if not runs:
            continue

        if average_reps:
            vmean, vstd, L_v = _stack_runs_timealigned_val(runs)
            c1m, c1s, c2m, c2s, L_e = _stack_runs_timealigned_ed(runs)
            if (vmean is None) or (c1m is None and c2m is None) or L_v == 0 or L_e == 0:
                continue
            L = min(L_v, L_e)
            if L <= 0:
                continue
            # Validation bounds
            if np.isfinite(vmean[:L]).any():
                vm = (vmean - (vstd if vstd is not None else 0))[:L]
                vM = (vmean + (vstd if vstd is not None else 0))[:L]
                if np.isfinite(vm).any(): vmins.append(np.nanmin(vm))
                if np.isfinite(vM).any(): vmaxs.append(np.nanmax(vM))
            # ED bounds
            ED_low, ED_high = [], []
            if c1m is not None:
                ED_low.append((c1m - (c1s if c1s is not None else 0))[:L])
                ED_high.append((c1m + (c1s if c1s is not None else 0))[:L])
            if c2m is not None:
                ED_low.append((c2m - (c2s if c2s is not None else 0))[:L])
                ED_high.append((c2m + (c2s if c2s is not None else 0))[:L])
            if ED_low and ED_high:
                lo = np.nanmin(np.concatenate(ED_low))
                hi = np.nanmax(np.concatenate(ED_high))
                if np.isfinite(lo): emins.append(lo)
                if np.isfinite(hi): emaxs.append(hi)
        else:
            r = runs[0]
            v = _val_from_run(r)
            c1, c2 = _ed_c12_from_run(r)
            arrays = [a for a in [v, c1, c2] if a.size]
            if not arrays:
                continue
            L = min(map(len, arrays))
            v = v[:L] if v.size else np.full(L, np.nan)
            c1 = c1[:L] if c1.size else np.full(L, np.nan)
            c2 = c2[:L] if c2.size else np.full(L, np.nan)
            if np.isfinite(v).any():
                vmins.append(np.nanmin(v))
                vmaxs.append(np.nanmax(v))
            ED = np.stack([c1, c2], axis=0)
            if np.isfinite(ED).any():
                emins.append(np.nanmin(ED))
                emaxs.append(np.nanmax(ED))

    if not vmins or not vmaxs:
        vmin, vmax = 0.925, 1.0
    else:
        vmin, vmax = float(np.nanmin(vmins)), float(np.nanmax(vmaxs))
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
            vmin, vmax = 0.925, 1.0

    if not emins or not emaxs:
        emin, emax = -0.001, 0.007
    else:
        emin, emax = float(np.nanmin(emins)), float(np.nanmax(emaxs))
        if not np.isfinite(emin) or not np.isfinite(emax) or emin == emax:
            emin, emax = -0.001, 0.007

    vpad = 0.02 * max(1e-6, (vmax - vmin))
    epad = 0.10 * max(1e-9, (emax - emin))  # a bit more padding for ED
    return (vmin - vpad, vmax + vpad, emin - epad, emax + epad)

# ---------------- plotting ----------------
def plot_grid(dataset: str,
              histories: Dict[Tuple[int, int], List[dict]],
              out_dir="two_hidden_mlp_records/figures_2layer_view",
              average_reps=True,
              show_band=True,
              use_minmax=True,
              n_rows: int = 4,
              n_cols: int = 4):
    """
    Mosaic grid:
      rows x cols : (h1,h2) pairs, laid out in sorted order
    Left y-axis : mean Val Acc. (± std band if enabled)
    Right y-axis: mean ED for C1, C2 (± std bands if enabled)
    """
    if not histories:
        print(f"[WARN] plot_grid: empty histories for dataset={dataset}. Skipping.")
        return

    os.makedirs(out_dir, exist_ok=True)

    # Collect (h1,h2) pairs
    all_h1h2 = sorted(histories.keys())
    N = len(all_h1h2)

    # If user expects 4x4=16 but N differs, adapt layout
    if n_rows * n_cols != N:
        # Try to pick a compact grid
        n_cols = int(np.ceil(np.sqrt(N)))
        n_rows = int(np.ceil(N / n_cols))

    # Style (no seaborn dependency required)
    plt.rcParams.update({
        "axes.grid": True,
        "grid.linestyle": "--",
        "grid.alpha": 0.3,
        "font.size": 8,
    })

    # Size scales with columns/rows
    fig_w = 10
    fig_h = 10
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(fig_w, fig_h),
        sharex=True, sharey=False, squeeze=False
    )

    # Colors
    colors = {
        'val_acc': '#1f77b4',  # Blue
        'ed_c1':   '#ff7f0e',  # Orange
        'ed_c2':   '#2ca02c',  # Green
    }
    right_axis_color = "#444444"  # neutral for right axis ticks/label

    # Axis limits
    if use_minmax:
        vmin, vmax, emin, emax = _compute_global_ranges(histories, average_reps=average_reps)
    else:
        vmin, vmax = 0.0, 0.4
        emin, emax = -0.001, 0.007

    # Determine legend necessity by scanning once
    any_c1, any_c2 = False, False
    for runs in histories.values():
        for r in runs:
            c1, c2 = _ed_c12_from_run(r)
            if c1.size and np.isfinite(c1).any(): any_c1 = True
            if c2.size and np.isfinite(c2).any(): any_c2 = True
        if any_c1 and any_c2:
            break

    # Plot every pair in order
    for idx, (h1, h2) in enumerate(all_h1h2):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]

        runs = histories[(h1, h2)]
        best_acc = None

        if average_reps:
            val_mean, val_std, L_v = _stack_runs_timealigned_val(runs)
            c1_mean, c1_std, c2_mean, c2_std, L_e = _stack_runs_timealigned_ed(runs)
            if (val_mean is None) or (c1_mean is None and c2_mean is None) or L_v == 0 or L_e == 0:
                ax.axis("on")
                ax.set_xticks([]); ax.set_yticks([])
                ax.text(0.5, 0.5, "empty", ha="center", va="center", fontsize=8, alpha=0.6)
                continue
            L = min(L_v, L_e)
            epochs = np.arange(1, L + 1)

            if val_mean is not None and np.isfinite(val_mean[:L]).any():
                best_acc = float(np.nanmax(val_mean[:L]))

            # Left axis: Val Acc.
            ax.set_ylim(vmin, vmax)
            ax.plot(epochs, val_mean[:L], color=colors['val_acc'], lw=2, label='Val Acc')
            if show_band and (val_std is not None):
                ax.fill_between(
                    epochs,
                    (val_mean - val_std)[:L],
                    (val_mean + val_std)[:L],
                    color=colors['val_acc'], alpha=0.2
                )

            # Right axis: ED (C1, C2)
            ax2 = ax.twinx()
            ax2.set_ylim(emin, emax)
            if c1_mean is not None and np.isfinite(c1_mean[:L]).any():
                ax2.plot(epochs, c1_mean[:L], color=colors['ed_c1'], linestyle='--', lw=1.5, label='ρ-param layer-1')
                if show_band and (c1_std is not None):
                    ax2.fill_between(epochs, (c1_mean - c1_std)[:L], (c1_mean + c1_std)[:L],
                                     color=colors['ed_c1'], alpha=0.15)
            if c2_mean is not None and np.isfinite(c2_mean[:L]).any():
                ax2.plot(epochs, c2_mean[:L], color=colors['ed_c2'], linestyle='-.', lw=1.5, label='ρ-param layer-ρ2')
                if show_band and (c2_std is not None):
                    ax2.fill_between(epochs, (c2_mean - c2_std)[:L], (c2_mean + c2_std)[:L],
                                     color=colors['ed_c2'], alpha=0.15)
        else:
            run = runs[0]
            v = _val_from_run(run)
            c1, c2 = _ed_c12_from_run(run)
            arrays = [a for a in [v, c1, c2] if a.size]
            if not arrays:
                ax.axis("on"); ax.set_xticks([]); ax.set_yticks([])
                ax.text(0.5, 0.5, "empty", ha="center", va="center", fontsize=8, alpha=0.6)
                continue
            L = min(map(len, arrays))
            v, c1, c2 = v[:L], c1[:L], c2[:L]
            epochs = np.arange(1, L + 1)

            if v.size and np.isfinite(v).any():
                best_acc = float(np.nanmax(v))

            ax.set_ylim(vmin, vmax)
            ax.plot(epochs, v, color=colors['val_acc'], lw=2, label='Val Acc')
            ax2 = ax.twinx()
            ax2.set_ylim(emin, emax)
            if c1.size:
                ax2.plot(epochs, c1, color=colors['ed_c1'], linestyle='--', lw=1.5, label='ρ-param layer-1')
            if c2.size:
                ax2.plot(epochs, c2, color=colors['ed_c2'], linestyle='-.', lw=1.5, label='ρ-param layer-2')

        # Titles / labels
        ax.set_title(f"h1={h1}, h2={h2}", fontsize=9, pad=5)

        # Left y labels only on leftmost column
        if col == 0:
            ax.set_ylabel("Val Acc", color=colors['val_acc'], fontsize=8)
            ax.tick_params(axis='y', labelcolor=colors['val_acc'], labelsize=7)
        else:
            ax.tick_params(axis='y', labelsize=0, length=0)

        # Right y labels only on rightmost column
        if col == n_cols - 1:
            ax2.set_ylabel("ρ-param", color=right_axis_color, fontsize=8)
            ax2.tick_params(axis='y', labelcolor=right_axis_color, labelsize=7)
        else:
            ax2.tick_params(axis='y', labelsize=0, length=0)

        # X labels only on bottom row
        #ax.set_xlabel("Epoch", fontsize=8)
        ax.tick_params(axis='x', labelsize=7)

        ax.grid(True, alpha=0.3, linestyle='--')

        # Best-acc box
        if best_acc is not None and np.isfinite(best_acc):
            ax.text(0.5, 0.5, f"{best_acc:.3f}",
                    transform=ax.transAxes, fontsize=17,
                    ha='center', va='center',
                    bbox=dict(facecolor='white', alpha=0.85, edgecolor='none'))

    # Dynamic figure-level legend (only show what appears anywhere)
    lines = [plt.Line2D([0], [0], lw=2, color=colors['val_acc'], label='Val Acc.')]
    if any_c1:
        lines.append(plt.Line2D([0], [0], lw=1.5, color=colors['ed_c1'], linestyle='--', label='ρ-param layer-1'))
    if any_c2:
        lines.append(plt.Line2D([0], [0], lw=1.5, color=colors['ed_c2'], linestyle='-.', label='ρ-param layer-2'))

    fig.legend(handles=lines, loc="upper center", frameon=True,
               fontsize=9, ncol=min(4, len(lines)), bbox_to_anchor=(0.5, 0.98),
               framealpha=0.95, edgecolor='gray')

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    base = os.path.join(out_dir, f"{dataset}_grid_2hidden")
    plt.savefig(base + ".png", dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {base}.png")

# ---------------- main ----------------
if __name__ == "__main__":
    datasets = ["cifar100"]
    for d in datasets:
        hists = load_histories_twohidden("../records/two_hidden_mlp_records/histories", dataset=d)
        if not hists:
            print(f"[WARN] No histories found for dataset={d}. Nothing to plot.")
            continue
        # Force 4x4 if you KNOW there are 16 pairs; otherwise it will auto-adapt.
        plot_grid(
            dataset=d,
            histories=hists,
            out_dir=".",
            average_reps=False,
            show_band=True,
            use_minmax=True,
            n_rows=4,
            n_cols=4
        )

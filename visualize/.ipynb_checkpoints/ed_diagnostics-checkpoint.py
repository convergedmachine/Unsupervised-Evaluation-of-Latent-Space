#!/usr/bin/env python3
import os, math, json, argparse
from typing import List, Tuple, Dict, Optional
import numpy as np
import torch
import matplotlib.pyplot as plt

# -----------------------------
# Core utilities (float64 throughout)
# -----------------------------
@torch.no_grad()
def _phi(x: torch.Tensor) -> torch.Tensor:
    x = x.to(torch.float64)
    return torch.exp(-0.5 * x**2) / math.sqrt(2 * math.pi)

@torch.no_grad()
def _Phi(x: torch.Tensor) -> torch.Tensor:
    x = x.to(torch.float64)
    return 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

@torch.no_grad()
def robust_standardize(x: torch.Tensor) -> torch.Tensor:
    x = x.to(torch.float64)
    med = x.median()
    mad = (x - med).abs().median().clamp(min=1e-12)
    return (x - med) / (1.4826 * mad)

@torch.no_grad()
def meanstd_standardize(x: torch.Tensor) -> torch.Tensor:
    x = x.to(torch.float64)
    mu = x.mean()
    sd = x.std(unbiased=True).clamp(min=1e-12)
    return (x - mu) / sd

@torch.no_grad()
def offdiag_values(C: torch.Tensor, mode: str = "all") -> torch.Tensor:
    n = C.shape[0]
    if mode == "upper":
        iu = torch.triu_indices(n, n, offset=1, device=C.device)
        return C[iu[0], iu[1]].to(torch.float64)
    elif mode == "all":
        mask = ~torch.eye(n, dtype=torch.bool, device=C.device)
        return C[mask].flatten().to(torch.float64)
    else:
        raise ValueError("mode must be 'upper' or 'all'")

@torch.no_grad()
def expected_abs_diff_standard_norm() -> float:
    # E|Z - Z'| with Z,Z' ~ N(0,1)
    return 2.0 / math.sqrt(math.pi)

@torch.no_grad()
def E_abs_x_minus_Z(x: torch.Tensor) -> torch.Tensor:
    x = x.to(torch.float64)
    # E|x - Z| = 2*phi(x) + x*(2*Phi(x)-1)
    return 2.0 * _phi(x) + x * (2.0 * _Phi(x) - 1.0)

@torch.no_grad()
def pairwise_abs_mean_u_stat(x: torch.Tensor) -> torch.Tensor:
    x = x.to(torch.float64)
    m = x.numel()
    if m < 2:
        return torch.tensor(float("nan"), dtype=torch.float64, device=x.device)
    xs, _ = torch.sort(x.view(-1))
    idx = torch.arange(1, m + 1, dtype=torch.float64, device=x.device)
    coef = 2.0 * idx - m - 1.0
    s = (coef * xs).sum()
    mean_pair_abs = (2.0 * s) / (m * (m - 1))
    return mean_pair_abs

@torch.no_grad()
def energy_distance_to_standard_normal(
    x: torch.Tensor,
    standardize: str = "meanstd"
) -> float:
    """
    Returns ρ̂ = ED( F_x , N(0,1) ) estimated from sample x.
    standardize ∈ {"meanstd","robust"}: how to standardize x before ED.
    """
    if standardize == "robust":
        xs = robust_standardize(x)
    elif standardize == "meanstd":
        xs = meanstd_standardize(x)
    else:
        raise ValueError("standardize must be 'robust' or 'meanstd'")

    if xs.numel() < 8:
        return float("nan")

    term_xz = E_abs_x_minus_Z(xs).mean()                         # tensor float64
    term_xx = pairwise_abs_mean_u_stat(xs)                       # tensor float64
    term_zz = torch.tensor(expected_abs_diff_standard_norm(),
                           dtype=torch.float64, device=xs.device)
    D = 2.0 * term_xz - term_xx - term_zz
    return float(D.item())

# -----------------------------
# Simulation helpers
# -----------------------------
@torch.no_grad()
def simulate_one_n(
    n: int,
    trials: int,
    symmetrize: bool,
    offdiag_mode: str,
    device: str,
    standardize: str
) -> np.ndarray:
    vals = []
    for _ in range(trials):
        C = torch.randn(n, n, device=device, dtype=torch.float64)
        C.fill_diagonal_(0.0)
        if symmetrize:
            C = 0.5 * (C + C.T)
            mode = "upper"
        else:
            mode = offdiag_mode
        x = offdiag_values(C, mode=mode)
        d = energy_distance_to_standard_normal(x, standardize=standardize)
        vals.append(d)
    return np.array(vals, dtype=np.float64)

@torch.no_grad()
def simulate_iid_normal(
    m: int,
    trials: int,
    device: str,
    standardize: str
) -> np.ndarray:
    """
    Baseline: draw i.i.d. N(0,1) samples of length m and compute ED to N(0,1)
    with the same standardization. This captures small-sample bias due to
    in-sample standardization.
    """
    vals = []
    for _ in range(trials):
        x = torch.randn(m, device=device, dtype=torch.float64)
        d = energy_distance_to_standard_normal(x, standardize=standardize)
        vals.append(d)
    return np.array(vals, dtype=np.float64)

@torch.no_grad()
def trials_for_n(
    n: int,
    target_total_offdiag: int,
    min_trials: int,
    symmetrize: bool,
    offdiag_mode: str
) -> int:
    if symmetrize:
        od_per_matrix = n * (n - 1) // 2
    else:
        od_per_matrix = n * (n - 1) if offdiag_mode == "all" else n * (n - 1) // 2
    return max(min_trials, math.ceil(target_total_offdiag / max(1, od_per_matrix)))

# -----------------------------
# Plotting routines
# -----------------------------
def plot_bias_curve(
    ns: List[int],
    data: Dict[int, np.ndarray],
    out_png: str,
    symmetrize: bool,
    offdiag_mode: str,
    standardize: str,
    baseline: Optional[Dict[int, np.ndarray]] = None
):
    means = np.array([data[n].mean() for n in ns])
    stds  = np.array([data[n].std(ddof=1) if len(data[n])>1 else 0.0 for n in ns])

    plt.figure(figsize=(10,6))
    # Raw ED curve
    plt.plot(ns, means, marker="o", label="Mean(ρ̂): off-diagonals")
    plt.fill_between(ns, means - stds, means + stds, alpha=0.15, label="±1 std (raw)")

    title = f"Empirical ρ̂ Bias Curve • symmetrize={symmetrize} • mode={offdiag_mode} • std={standardize}"
    if baseline is not None:
        b_means = np.array([baseline[n].mean() for n in ns])
        # Overlay baseline (i.i.d. Gaussian) mean
        plt.plot(ns, b_means, marker="s", linestyle="--", label="Mean(ρ̂): i.i.d. N(0,1) baseline")
        # Bias-corrected (difference of means)
        corr_means = means - b_means
        plt.plot(ns, corr_means, marker="^", label="Bias-corrected Mean(ρ̂)")

        title += " • with baseline"

    plt.axhline(0.0, linestyle="--", alpha=0.6)
    plt.title(title)
    plt.xlabel("Latent space dimension n"); plt.ylabel("ρ̂ (ED to N(0,1))")
    plt.grid(True, linestyle="--", alpha=0.4); plt.legend()
    plt.tight_layout(); plt.savefig(out_png, dpi=150); plt.close()

def plot_box_violin(ns: List[int], data: Dict[int, np.ndarray], out_png: str, standardize: str):
    ordered = [data[n] for n in ns]
    plt.figure(figsize=(10,6))
    _ = plt.violinplot(ordered, positions=range(len(ns)), showmeans=True, showextrema=True)
    plt.xticks(range(len(ns)), ns)
    plt.title(f"ρ̂ Distribution per n (violin) • std={standardize}")
    plt.xlabel("Latent space dimension n"); plt.ylabel("ρ̂ (ED to N(0,1))")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout(); plt.savefig(out_png, dpi=150); plt.close()

def plot_convergence(
    n_values: List[int],
    base_trials: int,
    symmetrize: bool,
    offdiag_mode: str,
    device: str,
    out_png: str,
    standardize: str
):
    plt.figure(figsize=(10,6))
    for n in n_values:
        checkpoints = [max(4, int(base_trials * t)) for t in [0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]]
        series = []
        for t in checkpoints:
            vals = simulate_one_n(n, t, symmetrize, offdiag_mode, device, standardize)
            series.append(vals.mean())
        plt.plot(checkpoints, series, marker="o", label=f"n={n}")
    plt.title(f"Convergence of Mean ρ̂ vs Trials • std={standardize}")
    plt.xlabel("Trials"); plt.ylabel("Mean ρ̂")
    plt.grid(True, linestyle="--", alpha=0.4); plt.legend()
    plt.tight_layout(); plt.savefig(out_png, dpi=150); plt.close()

def plot_ed_vs_offdiag(
    ns: List[int],
    data: Dict[int, np.ndarray],
    symmetrize: bool,
    out_png: str,
    standardize: str
):
    m_vals = [(n*(n-1)//2) if symmetrize else (n*(n-1)) for n in ns]
    means = [data[n].mean() for n in ns]
    plt.figure(figsize=(9,6))
    plt.plot(m_vals, means, marker="o")
    plt.title(f"Mean ρ̂ vs Off-diagonal count m • std={standardize}")
    plt.xlabel("m = number of off-diagonals"); plt.ylabel("Mean ρ̂")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout(); plt.savefig(out_png, dpi=150); plt.close()

def plot_qq(
    ns_subset: List[int],
    symmetrize: bool,
    offdiag_mode: str,
    device: str,
    out_dir: str,
    standardize: str
):
    for n in ns_subset:
        # One matrix per QQ plot
        C = torch.randn(n, n, device=device, dtype=torch.float64)
        C.fill_diagonal_(0.0)
        if symmetrize:
            C = 0.5 * (C + C.T)
            mode = "upper"
        else:
            mode = offdiag_mode
        x = offdiag_values(C, mode=mode)

        if standardize == "meanstd":
            xs = meanstd_standardize(x).cpu().numpy()
        else:
            xs = robust_standardize(x).cpu().numpy()

        xs = np.sort(xs)
        q = np.linspace(0.001, 0.999, len(xs))
        z = np.sqrt(2) * torch.erfinv(torch.tensor(2*q - 1, dtype=torch.float64)).numpy()
        plt.figure(figsize=(6,6))
        plt.scatter(z, xs, s=8)
        lim = float(max(abs(xs[0]), abs(xs[-1]), abs(z[0]), abs(z[-1])))
        plt.plot([-lim, lim], [-lim, lim], linestyle="--", alpha=0.6)
        plt.title(f"QQ-plot vs N(0,1) (n={n}, std={standardize})")
        plt.xlabel("Theoretical quantiles"); plt.ylabel("Empirical quantiles")
        plt.grid(True, linestyle="--", alpha=0.4)
        out_png = os.path.join(out_dir, f"qq_n{n}.png")
        plt.tight_layout(); plt.savefig(out_png, dpi=150); plt.close()

# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=str, default="./ed_diagnostics_out")
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--ns", type=str, default="16,24,32,48,64,96,128,192")
    ap.add_argument("--symmetrize", action="store_true",
                    help="Use 0.5*(C+C^T) and upper-tri off-diagonals")
    ap.add_argument("--offdiag_mode", type=str, default="all", choices=["all","upper"])
    ap.add_argument("--target_offdiag", type=int, default=120000)
    ap.add_argument("--min_trials", type=int, default=32)
    ap.add_argument("--trials_cap", type=int, default=2048)
    ap.add_argument("--convergence_ns", type=str, default="16,24,32")
    ap.add_argument("--convergence_trials", type=int, default=256)
    ap.add_argument("--standardize", type=str, default="robust", choices=["meanstd","robust"],
                    help="Standardization used before ED")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--with_baseline", action="store_true",
                    help="Also simulate i.i.d. N(0,1) baseline and plot bias-corrected curve")
    args = ap.parse_args()

    # Reproducibility
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    os.makedirs(args.outdir, exist_ok=True)
    ns = [int(s) for s in args.ns.split(",") if s.strip()]
    conv_ns = [int(s) for s in args.convergence_ns.split(",") if s.strip()]

    # Collect ED samples per n with trial scaling
    data: Dict[int, np.ndarray] = {}
    trials_map: Dict[int, int] = {}
    for n in ns:
        t = trials_for_n(n, args.target_offdiag, args.min_trials, args.symmetrize, args.offdiag_mode)
        t = min(t, args.trials_cap)
        vals = simulate_one_n(n, t, args.symmetrize, args.offdiag_mode, args.device, args.standardize)
        data[n] = vals
        trials_map[n] = t

    # Optional: baseline i.i.d. Gaussian with matched sample size m and trials
    baseline: Optional[Dict[int, np.ndarray]] = None
    if args.with_baseline:
        baseline = {}
        for n in ns:
            m = (n*(n-1)//2) if args.symmetrize else (n*(n-1) if args.offdiag_mode == "all" else n*(n-1)//2)
            baseline[n] = simulate_iid_normal(m, trials_map[n], args.device, args.standardize)

    # Save raw data for reproducibility
    np.savez(os.path.join(args.outdir, "ed_bias_raw.npz"), **{f"n{n}": data[n] for n in ns})
    summary = {
        int(n): {
            "mean": float(data[n].mean()),
            "std": float(data[n].std(ddof=1) if len(data[n])>1 else 0.0),
            "trials": int(len(data[n]))
        } for n in ns
    }

    if baseline is not None:
        np.savez(os.path.join(args.outdir, "ed_baseline_raw.npz"), **{f"n{n}": baseline[n] for n in ns})
        for n in ns:
            summary[int(n)]["baseline_mean"] = float(baseline[n].mean())
            summary[int(n)]["bias_corrected_mean"] = float(data[n].mean() - baseline[n].mean())

    with open(os.path.join(args.outdir, "ed_bias_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # 1) Bias curve (with optional baseline & bias-corrected overlay)
    plot_bias_curve(
        ns, data,
        os.path.join(args.outdir, "01_bias_curve.png"),
        symmetrize=args.symmetrize,
        offdiag_mode=args.offdiag_mode,
        standardize=args.standardize,
        baseline=baseline
    )

    # 2) Distribution per n
    plot_box_violin(ns, data, os.path.join(args.outdir, "02_violin_distributions.png"), args.standardize)

    # 3) Convergence vs trials for small n
    plot_convergence(
        conv_ns, args.convergence_trials, args.symmetrize, args.offdiag_mode,
        args.device, os.path.join(args.outdir, "03_convergence.png"), args.standardize
    )

    # 4) ρ̂ vs m
    plot_ed_vs_offdiag(ns, data, args.symmetrize, os.path.join(args.outdir, "04_rho_vs_m.png"), args.standardize)

    # 5) QQ plots for a subset
    plot_qq(conv_ns, args.symmetrize, args.offdiag_mode, args.device, args.outdir, args.standardize)

    print(f"Diagnostics complete. See outputs in {args.outdir}")

if __name__ == "__main__":
    main()

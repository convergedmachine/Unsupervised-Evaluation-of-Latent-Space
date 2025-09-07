#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
normality_simulation.py

Compare 1-D normality divergence metrics on synthetic data:
- Energy Distance to N(0,1)
- MMD^2 (RBF) vs. N(0,1) (with MC against Gaussian)
- Mardia univariate skew^2 + (excess-kurtosis)^2
- 1-D Wasserstein-2 distance vs. N(0,1)

Outputs:
- CSV/JSON table with metrics
- One bar chart per metric (PNG)
- One combined grouped bar chart with all methods (PNG)

Dependencies: torch, numpy, pandas, matplotlib
"""
import os, math, json, argparse
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional

import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt

torch.set_default_dtype(torch.float64)

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
def expected_abs_diff_standard_norm() -> float:
    # E|Z - Z'|, Z,Z' ~ N(0,1)
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

# -----------------------------
# Metrics
# -----------------------------
@torch.no_grad()
def energy_distance_to_standard_normal(
    x: torch.Tensor,
    standardize: str = "meanstd"
) -> float:
    """
    Returns ρ̂ = ED(F_x , N(0,1)) estimated from sample x.
    standardize ∈ {"meanstd","robust"}.
    """
    if standardize == "robust":
        xs = robust_standardize(x)
    elif standardize == "meanstd":
        xs = meanstd_standardize(x)
    else:
        raise ValueError("standardize must be 'robust' or 'meanstd'")

    if xs.numel() < 8:
        return float("nan")

    term_xz = E_abs_x_minus_Z(xs).mean()
    term_xx = pairwise_abs_mean_u_stat(xs)
    term_zz = torch.tensor(expected_abs_diff_standard_norm(),
                           dtype=torch.float64, device=xs.device)
    D = 2.0 * term_xz - term_xx - term_zz
    return float(D.item())

@torch.no_grad()
def _apply_standardize(x: torch.Tensor, standardize: str) -> torch.Tensor:
    if standardize == "robust":
        return robust_standardize(x)
    elif standardize == "meanstd":
        return meanstd_standardize(x)
    else:
        raise ValueError("standardize must be 'robust' or 'meanstd'")

@torch.no_grad()
def mmd_rbf_to_standard_normal(
    x: torch.Tensor,
    bandwidth: Optional[float] = None,
    standardize: str = "meanstd",
    z_mc: int = 512,
    rng: Optional[torch.Generator] = None,
    kxx_subsample: int = 600
) -> float:
    """
    Unbiased MMD^2 (RBF kernel) between empirical x and N(0,1).
    - bandwidth: if None, median heuristic (on a subsample if large)
    - z_mc: number of Gaussian draws for Monte-Carlo expectations
    - kxx_subsample: limit Kxx to at most this many points for speed
    Returns nonnegative float (MMD^2).
    """
    xs = _apply_standardize(x, standardize).view(-1)
    n = xs.numel()
    if n < 4:
        return float("nan")

    if bandwidth is None:
        if n > 1000:
            idx = torch.randperm(n, generator=rng)[:1000]
            xs_med = xs[idx]
        else:
            xs_med = xs
        diffs = xs_med.unsqueeze(0) - xs_med.unsqueeze(1)
        d2 = (diffs**2).triu(diagonal=1)
        med = torch.median(d2[d2 > 0]) if (d2 > 0).any() else torch.tensor(1.0, dtype=torch.float64)
        bandwidth = torch.sqrt(0.5 * med).item() if med > 0 else 1.0

    sigma2 = float(bandwidth) ** 2

    def k(a, b):
        return torch.exp(- (a.unsqueeze(1) - b.unsqueeze(0))**2 / (2.0 * sigma2))

    # Kxx (unbiased)
    if n > kxx_subsample:
        perm = torch.randperm(n, generator=rng)
        idx = perm[:kxx_subsample]
        xsub = xs[idx]
        m = xsub.numel()
        Kxx = k(xsub, xsub)
        sum_Kxx = (Kxx.sum() - torch.diagonal(Kxx).sum()) / (m * (m - 1))
    else:
        Kxx = k(xs, xs)
        sum_Kxx = (Kxx.sum() - torch.diagonal(Kxx).sum()) / (n * (n - 1))

    # Kzz via MC
    g = rng if rng is not None else torch.Generator(device=xs.device)
    Z = torch.randn(z_mc, dtype=torch.float64, generator=g, device=xs.device)
    z = Z.shape[0]
    Kzz = k(Z, Z)
    sum_Kzz = (Kzz.sum() - torch.diagonal(Kzz).sum()) / (z * (z - 1))

    # cross
    Kxz = k(xs, Z)
    sum_Kxz = Kxz.mean()

    mmd2 = float(sum_Kxx + sum_Kzz - 2.0 * sum_Kxz)
    return max(0.0, mmd2)

@torch.no_grad()
def mardia_univariate_distance(
    x: torch.Tensor,
    standardize: str = "meanstd"
) -> Dict[str, float]:
    """
    In 1-D, Mardia's measures reduce to classical skewness and kurtosis.
    Returns:
      - skew2   = (sample skewness)^2
      - excess2 = (sample kurtosis - 3)^2
      - combined = skew2 + excess2
    """
    xs = _apply_standardize(x, standardize).view(-1)
    n = xs.numel()
    if n < 8:
        return {"skew2": float("nan"), "excess2": float("nan"), "combined": float("nan")}

    mu = xs.mean()
    s = xs.std(unbiased=True).clamp(min=1e-12)
    z = (xs - mu) / s

    skew = (z**3).mean()
    kurt = (z**4).mean()
    skew2 = float(skew**2)
    excess2 = float((kurt - 3.0)**2)
    return {"skew2": skew2, "excess2": excess2, "combined": skew2 + excess2}

@torch.no_grad()
def wasserstein2_to_standard_normal(
    x: torch.Tensor,
    standardize: str = "meanstd"
) -> float:
    """
    1-D Wasserstein-2 distance W2(x, N(0,1)) using quantile matching.
    Returns W2 (not squared).
    """
    xs = _apply_standardize(x, standardize).view(-1)
    n = xs.numel()
    if n < 2:
        return float("nan")
    xs_sorted, _ = torch.sort(xs)

    i = torch.arange(1, n + 1, dtype=torch.float64, device=xs.device)
    p = (i - 0.5) / n
    q = math.sqrt(2.0) * torch.erfinv(2.0 * p - 1.0)  # Gaussian quantiles

    w2_sq = torch.mean((xs_sorted - q)**2)
    return float(torch.sqrt(w2_sq).item())

# -----------------------------
# Synthetic data generators
# -----------------------------
def sample_synthetic(n: int, kind: str, rng: torch.Generator) -> torch.Tensor:
    if kind == "Gaussian":
        return torch.randn(n, generator=rng, dtype=torch.float64)
    elif kind == "t(df=3)":
        # heavy-tailed
        df = 3.0
        Z = torch.randn(n, generator=rng, dtype=torch.float64)
        # Chi^2(df) ~ Gamma(df/2, 2), sample then normalize
        U = torch.distributions.Gamma(df/2.0, 2.0).sample((n,)).to(torch.float64)
        return Z / torch.sqrt(U / df)
    elif kind == "Laplace(var=1)":
        # Laplace with variance 1 → b = 1/sqrt(2)
        b = 1.0 / math.sqrt(2.0)
        U = torch.rand(n, generator=rng, dtype=torch.float64)
        return b * torch.sign(U - 0.5) * torch.log1p(-2.0 * torch.abs(U - 0.5))
    elif kind == "Gaussian Mixture":
        # 0.5*N(-2,1) + 0.5*N(2,1)
        comps = (torch.rand(n, generator=rng, dtype=torch.float64) < 0.5)
        Z = torch.randn(n, generator=rng, dtype=torch.float64)
        return torch.where(comps, Z + 2.0, Z - 2.0)
    elif kind == "Skewed (Exp-1)":
        # Exponential(1) - 1 (skewed right)
        E = torch.distributions.Exponential(1.0).sample((n,)).to(torch.float64)
        return E - 1.0
    else:
        raise ValueError(f"Unknown kind: {kind}")

# -----------------------------
# Config / Runner
# -----------------------------
@dataclass(frozen=True)
class SimConfig:
    n: int = 2000
    standardize: str = "meanstd"     # "meanstd" | "robust"
    seed: int = 12345
    outdir: str = "runs/normality_demo"
    # MMD controls
    mmd_z_mc: int = 512
    mmd_kxx_subsample: int = 600

def ensure_outdir(path: str):
    os.makedirs(path, exist_ok=True)

def save_config(cfg: SimConfig):
    ensure_outdir(cfg.outdir)
    with open(os.path.join(cfg.outdir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)

def compute_all_metrics(
    x: torch.Tensor,
    cfg: SimConfig,
    rng: torch.Generator
) -> Dict[str, float]:
    ed = energy_distance_to_standard_normal(x, standardize=cfg.standardize)
    mmd = mmd_rbf_to_standard_normal(
        x, bandwidth=None, standardize=cfg.standardize,
        z_mc=cfg.mmd_z_mc, rng=rng, kxx_subsample=cfg.mmd_kxx_subsample
    )
    mardia = mardia_univariate_distance(x, standardize=cfg.standardize)
    w2 = wasserstein2_to_standard_normal(x, standardize=cfg.standardize)

    return {
        "EnergyDistance": ed,
        "MMD_RBF": mmd,
        "Mardia_skew2": mardia["skew2"],
        "Mardia_excess2": mardia["excess2"],
        "Mardia_combined": mardia["combined"],
        "Wasserstein2": w2
    }

# -----------------------------
# Plotting
# -----------------------------
def plot_metric_bars(df: pd.DataFrame, metric: str, outdir: str):
    plt.figure()
    df[metric].plot(kind="bar")
    plt.ylabel(metric)
    plt.title(f"{metric} vs. distribution")
    plt.tight_layout()
    path = os.path.join(outdir, f"{metric}.png")
    plt.savefig(path)
    plt.close()

def plot_all_metrics(df: pd.DataFrame, outdir: str):
    """
    Grouped bar plot: distributions on x-axis; different methods as side-by-side bars.
    No explicit colors/styles set (matplotlib default palette).
    """
    metrics = ["EnergyDistance", "MMD_RBF", "Mardia_combined", "Wasserstein2"]
    ax = df[metrics].plot(kind="bar", figsize=(10, 6), width=0.8)
    ax.set_ylabel("Divergence Score")
    ax.set_title("Comparison of Normality Divergence Methods")
    ax.set_ylim(0.0, 0.1)
    ax.legend(title="Method")
    plt.tight_layout()
    path = os.path.join(outdir, "all_methods_comparison.png")
    plt.savefig(path)
    plt.close()

# -----------------------------
# Orchestration
# -----------------------------
def run(cfg: SimConfig):
    save_config(cfg)
    ensure_outdir(cfg.outdir)

    rng = torch.Generator()
    rng.manual_seed(cfg.seed)

    kinds = [
        "Gaussian",
        "t(df=3)",
        "Laplace(var=1)",
        "Gaussian Mixture",
        "Skewed (Exp-1)",
    ]

    rows: List[Dict[str, float]] = []
    for kind in kinds:
        x = sample_synthetic(cfg.n, kind, rng)
        metrics = compute_all_metrics(x, cfg, rng)
        metrics["Distribution"] = kind
        rows.append(metrics)

    df = pd.DataFrame(rows).set_index("Distribution")
    csv_path = os.path.join(cfg.outdir, "metrics_table.csv")
    df.to_csv(csv_path)
    with open(os.path.join(cfg.outdir, "metrics_table.json"), "w", encoding="utf-8") as f:
        json.dump(json.loads(df.to_json(orient="index")), f, indent=2)

    # Individual metric charts
    for metric in ["EnergyDistance", "MMD_RBF", "Mardia_combined", "Wasserstein2"]:
        plot_metric_bars(df, metric, cfg.outdir)

    # Combined grouped bar plot
    plot_all_metrics(df, cfg.outdir)

    print(f"[ok] Results saved to: {cfg.outdir}")
    print(f" - CSV: {csv_path}")
    for metric in ["EnergyDistance", "MMD_RBF", "Mardia_combined", "Wasserstein2"]:
        print(f" - Figure: {os.path.join(cfg.outdir, metric + '.png')}")
    print(f" - Combined: {os.path.join(cfg.outdir, 'all_methods_comparison.png')}")

# -----------------------------
# CLI
# -----------------------------
def parse_args() -> SimConfig:
    p = argparse.ArgumentParser(description="Compare 1-D normality metrics on synthetic distributions.")
    p.add_argument("--n", type=int, default=2000, help="Sample size per distribution.")
    p.add_argument("--standardize", type=str, default="meanstd", choices=["meanstd", "robust"], help="Standardization mode.")
    p.add_argument("--seed", type=int, default=12345, help="Master RNG seed.")
    p.add_argument("--outdir", type=str, default="runs/normality_demo", help="Output directory.")

    # MMD speeds/sensitivity
    p.add_argument("--mmd_z_mc", type=int, default=512, help="MC draws from N(0,1) for MMD.")
    p.add_argument("--mmd_kxx_subsample", type=int, default=600, help="Subsample size for Kxx computation.")

    args = p.parse_args()
    return SimConfig(
        n=args.n,
        standardize=args.standardize,
        seed=args.seed,
        outdir=args.outdir,
        mmd_z_mc=args.mmd_z_mc,
        mmd_kxx_subsample=args.mmd_kxx_subsample,
    )

if __name__ == "__main__":
    cfg = parse_args()
    run(cfg)

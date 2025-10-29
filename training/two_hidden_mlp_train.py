import os
import json
import math
import argparse
import numpy as np
from typing import Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
import torch.optim as optim
from torch.utils.data import DataLoader
from collections import defaultdict

# -----------------------------
# Dataset loader
# -----------------------------
def load_dataset(
    root: str = "data_folder",
    augment: bool = True,
    batch_size: int = 128,
    num_workers: int = 4,
    pin_memory: bool = True,
    drop_last: bool = False,
):
    """
    Returns:
      train_loader, val_loader, in_dim, out_dim, sizes_dict
    Data comes flattened to vectors for MLPs.
    """
    # CIFAR-100 stats
    mean = (0.5071, 0.4867, 0.4408)
    std  = (0.2675, 0.2565, 0.2761)

    # Transforms (flatten to 3072-dim vectors for MLP)
    if augment:
        train_transform = T.Compose([
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean, std),
            T.Lambda(lambda t: t.reshape(-1)),  # -> [3072]
        ])
    else:
        train_transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean, std),
            T.Lambda(lambda t: t.reshape(-1)),
        ])

    test_transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean, std),
        T.Lambda(lambda t: t.reshape(-1)),
    ])

    # Datasets (set download=True on first run if needed)
    trainset = torchvision.datasets.CIFAR100(
        root=root, train=True, download=False, transform=train_transform
    )
    testset = torchvision.datasets.CIFAR100(
        root=root, train=False, download=False, transform=test_transform
    )

    # DataLoaders
    common_dl_kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        persistent_workers=(num_workers > 0),
    )
    train_loader = DataLoader(trainset, shuffle=True,  **common_dl_kwargs)
    val_loader   = DataLoader(testset,   shuffle=False, **common_dl_kwargs)

    # MLP I/O dims
    in_dim = 3 * 32 * 32      # 3072
    out_dim = 100

    sizes = {"train": len(trainset), "val": len(testset)}
    return train_loader, val_loader, in_dim, out_dim, sizes


# -----------------------------
# 3-layer MLP (2 hidden) with coupling layers (bias-free linear)
# -----------------------------
class CoupledMLP3(nn.Module):
    """
    Two hidden layers. Bias-free square 'coupling' Linear(d,d, bias=False)
    inserted after each hidden activation.
      x -> fc1 -> BN -> tanh -> C1 -> fc2 -> BN -> tanh -> C2 -> fc_out
    """
    def __init__(self, in_dim: int, h1: int, h2: int, out_dim: int):
        super().__init__()
        # main layers
        self.fc1 = nn.Linear(in_dim, h1, bias=True)
        self.bn1 = nn.BatchNorm1d(h1)
        self.fc2 = nn.Linear(h1, h2, bias=True)
        self.bn2 = nn.BatchNorm1d(h2)
        self.fc_out = nn.Linear(h2, out_dim, bias=True)

        # coupling layers (no bias)
        self.C1 = nn.Linear(h1, h1, bias=False)
        self.C2 = nn.Linear(h2, h2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h1 = torch.tanh(self.bn1(self.fc1(x)))
        h1 = self.C1(h1)  # coupling layer #1

        h2 = torch.tanh(self.bn2(self.fc2(h1)))
        h2 = self.C2(h2)  # coupling layer #2

        logits = self.fc_out(h2)
        return logits

    def coupling_matrices(self):
        """
        Returns tuple of coupling matrices as (weight^T) to match h -> C h convention.
        """
        C1 = self.C1.weight.T.detach()
        C2 = self.C2.weight.T.detach()
        return (C1, C2)


# -----------------------------
# Core utilities (ED)
# -----------------------------
# -------------------------------------------------------------------------
# Standard Normal Density and CDF
# -------------------------------------------------------------------------
@torch.no_grad()
def _phi(x: torch.Tensor) -> torch.Tensor:
    """Standard normal PDF."""
    return torch.exp(-0.5 * x**2) / math.sqrt(2 * math.pi)

@torch.no_grad()
def _Phi(x: torch.Tensor) -> torch.Tensor:
    """Standard normal CDF."""
    return 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


# -------------------------------------------------------------------------
# Standardization Utilities
# -------------------------------------------------------------------------
@torch.no_grad()
def robust_standardize(x: torch.Tensor) -> torch.Tensor:
    """Robust standardization using median and MAD."""
    x = x.to(torch.float32)
    med = x.median()
    mad = (x - med).abs().median().clamp(min=1e-12)
    return (x - med) / (1.4826 * mad)

@torch.no_grad()
def meanstd_standardize(x: torch.Tensor) -> torch.Tensor:
    """Standardize using mean and standard deviation."""
    x = x.to(torch.float32)
    mu = x.mean()
    sd = x.std(unbiased=True).clamp(min=1e-12)
    return (x - mu) / sd


# -------------------------------------------------------------------------
# Off-diagonal Extraction
# -------------------------------------------------------------------------
@torch.no_grad()
def offdiag_values(C: torch.Tensor, mode: str = "all", zero_mean: bool = True) -> torch.Tensor:
    """
    Extract the off-diagonal elements from a square coupling matrix.

    Args:
        C (torch.Tensor): square matrix [n, n]
        mode (str): 'upper' (only upper triangle) or 'all'
        zero_mean (bool): subtract mean of off-diagonals (default=True)
    """
    n = C.shape[0]
    if mode == "upper":
        iu = torch.triu_indices(n, n, offset=1, device=C.device)
        vals = C[iu[0], iu[1]].to(torch.float32)
    elif mode == "all":
        mask = ~torch.eye(n, dtype=torch.bool, device=C.device)
        vals = C[mask].flatten().to(torch.float32)
    else:
        raise ValueError("mode must be 'upper' or 'all'")

    if zero_mean:
        vals = vals - vals.mean()

    return vals


# -------------------------------------------------------------------------
# Energy Distance Components
# -------------------------------------------------------------------------
@torch.no_grad()
def expected_abs_diff_standard_norm() -> float:
    """E|Z - Z'| for Z,Z' ~ N(0,1)."""
    return 2.0 / math.sqrt(math.pi)

@torch.no_grad()
def E_abs_x_minus_Z(x: torch.Tensor) -> torch.Tensor:
    """E|x - Z| where Z ~ N(0,1), computed elementwise."""
    return 2.0 * _phi(x) + x * (2.0 * _Phi(x) - 1.0)

@torch.no_grad()
def pairwise_abs_mean_u_stat(x: torch.Tensor) -> torch.Tensor:
    """Unbiased U-statistic for E|x - x'|."""
    m = x.numel()
    if m < 2:
        return torch.tensor(float("nan"), dtype=torch.float32)
    xs, _ = torch.sort(x.view(-1))
    idx = torch.arange(1, m + 1, dtype=torch.float32, device=x.device)
    coef = 2.0 * idx - m - 1.0
    s = (coef * xs).sum()
    mean_pair_abs = (2.0 * s) / (m * (m - 1))
    return mean_pair_abs


# -------------------------------------------------------------------------
# Energy Distance to N(0,1)
# -------------------------------------------------------------------------
@torch.no_grad()
def energy_distance_to_standard_normal(x: torch.Tensor, standardize: str = "robust") -> float:
    """Compute energy distance between empirical x and standard normal."""
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
    term_zz = expected_abs_diff_standard_norm()
    D = 2.0 * term_xz - term_xx - term_zz
    return float(D.item())


# -------------------------------------------------------------------------
# Main Evaluation: Coupling-Matrix Energy Distance
# -------------------------------------------------------------------------
@torch.no_grad()
def coupling_matrix_energy_distance(C: torch.Tensor, n_samples: int, mode: str = "upper") -> float:
    """
    Compute energy distance of Fisher-transformed off-diagonal couplings to N(0,1).

    Args:
        C (torch.Tensor): correlation or coupling matrix [d, d]
        n_samples (int): number of observations used to estimate C
        mode (str): 'upper' or 'all' off-diagonal extraction
    """
    # 1. Extract off-diagonal correlations
    r = offdiag_values(C, mode=mode, zero_mean=False)

    # 2. Fisher z-transform (stabilized correlation variance)
    z = torch.sqrt(torch.tensor(float(n_samples - 3))) * torch.atanh(r)

    # 3. Compare to N(0,1) via energy distance
    D = energy_distance_to_standard_normal(z, standardize="robust")

    return D

# -----------------------------
# Training utils
# -----------------------------
def accuracy(y_pred, y_true):
    return (y_pred.argmax(dim=1) == y_true).float().mean().item()

def train_model(
    train_loader,
    val_loader,
    model,
    lr: float = 1e-3,
    max_epochs: int = 200,
    device: str = None,
):
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = -float("inf")
    best_val_epoch = -1
    best_state_acc = None

    best_avg_ed = float("inf")
    best_ed_epoch = -1
    best_state_ed = None

    val_acc_history = []
    ed_history = defaultdict(list)

    has_couplings = hasattr(model, "coupling_matrices") and callable(getattr(model, "coupling_matrices"))

    for epoch in range(max_epochs):
        # ---- train
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()

        # ---- validate
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                out = model(xb)
                pred = out.argmax(dim=1)
                correct += (pred == yb).sum().item()
                total += yb.numel()
        val_acc = correct / max(1, total)
        val_acc_history.append(val_acc)

        # ---- ED tracking
        avg_ed = float("inf")
        if has_couplings:
            Cs = model.coupling_matrices()  # tuple of C_i as W^T
            ed_values = []
            for idx, C in enumerate(Cs, start=1):
                ed_val = coupling_matrix_energy_distance(C, n_samples=len(val_loader.dataset), mode="all")
                ed_history[f"C{idx}"].append(ed_val)
                if np.isfinite(ed_val):
                    ed_values.append(ed_val)

            # keep lists aligned if any key missing later
            max_k = len(Cs)
            for k in list(ed_history.keys()):
                if k not in [f"C{i}" for i in range(1, max_k + 1)] and len(ed_history[k]) < len(val_acc_history):
                    ed_history[k].append(float("nan"))

            if len(ed_values) > 0 and all(np.isfinite(v) for v in ed_values):
                avg_ed = float(np.mean(ed_values))

        # ---- best-by-acc
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_epoch = epoch
            best_state_acc = {k: v.detach().cpu() for k, v in model.state_dict().items()}

        # ---- best-by-ED
        if has_couplings and np.isfinite(avg_ed) and avg_ed < best_avg_ed:
            best_avg_ed = avg_ed
            best_ed_epoch = epoch
            best_state_ed = {k: v.detach().cpu() for k, v in model.state_dict().items()}

        # ---- log
        log = f"Epoch {epoch:03d}: val_acc={val_acc:.4f}"
        if has_couplings:
            parts = [f"{k}={ed_history[k][-1]:.4f}" for k in sorted(ed_history.keys())]
            if parts:
                log += " | ED(" + ") | ED(".join(parts) + ")"
            if np.isfinite(avg_ed):
                log += f" | avgED={avg_ed:.4f}"
        print(log)

    return {
        "final_model": model.cpu(),
        "best_state_acc": best_state_acc,
        "best_val_acc": best_val_acc,
        "best_val_epoch": best_val_epoch,
        "best_state_ed": best_state_ed,
        "best_avg_ed": best_avg_ed,
        "best_ed_epoch": best_ed_epoch,
        "val_hist": val_acc_history,
        "ed_hist": dict(ed_history),
    }


# -----------------------------
# Saving helpers
# -----------------------------
def _safe_makedirs(path: str):
    os.makedirs(path, exist_ok=True)

def _save_json(path: str, obj: Dict[str, Any]):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def _save_checkpoint(path: str, state_dict: Dict[str, torch.Tensor], meta: Dict[str, Any]):
    payload = {
        "state_dict": state_dict,
        "meta": meta,
    }
    torch.save(payload, path)


# -----------------------------
# Run wrapper (single configuration)
# -----------------------------
def run(
    dataset_name="cifar100",
    out_dir="two_hidden_mlp_records",
    h1=1024, h2=512, rep: int = 0,
    batch_size: int = 128,
    augment: bool = True,
    num_workers: int = 12,
    max_epochs: int = 500,
    lr: float = 1e-3,
):
    _safe_makedirs(out_dir)

    # Get loaders + dims
    train_loader, val_loader, in_dim, out_dim, sizes = load_dataset(
        batch_size=batch_size,
        augment=augment,
        num_workers=num_workers,
    )

    # Build 2-hidden model
    model = CoupledMLP3(in_dim, h1, h2, out_dim)

    # Train with loaders
    result = train_model(
        train_loader=train_loader,
        val_loader=val_loader,
        model=model,
        max_epochs=max_epochs,
        device=None,
        lr=lr,
    )

    # Save per-run histories (single file per run)
    hist_dir = os.path.join(out_dir, "histories", dataset_name, f"h1_{h1}_h2_{h2}")
    _safe_makedirs(hist_dir)
    hist_payload = {
        "dataset": dataset_name,
        "sizes": sizes,
        "h1": h1, "h2": h2,
        "rep": rep,
        "val_hist": result["val_hist"],
        "ed_hist": result["ed_hist"],
    }
    _save_json(os.path.join(hist_dir, f"run_{rep:02d}.json"), hist_payload)

    # Save/overwrite checkpoints deterministically (no grid bookkeeping)
    ckpt_dir = os.path.join(out_dir, "checkpoints", dataset_name, f"h1_{h1}_h2_{h2}")
    _safe_makedirs(ckpt_dir)

    if result["best_state_acc"] is not None:
        ckpt_path_acc = os.path.join(ckpt_dir, f"rep_{rep:02d}_best-acc.pt")
        meta_acc = {
            "criterion": "best-acc",
            "dataset": dataset_name,
            "h1": h1,
            "h2": h2,
            "best_val_acc": result["best_val_acc"],
            "best_val_epoch": result["best_val_epoch"],
            "rep": rep,
        }
        _save_checkpoint(ckpt_path_acc, result["best_state_acc"], meta_acc)
        print(f"[SAVE] {ckpt_path_acc} (val_acc={result['best_val_acc']:.4f} @ epoch {result['best_val_epoch']})")

    if result["best_state_ed"] is not None and np.isfinite(result["best_avg_ed"]):
        ckpt_path_ed = os.path.join(ckpt_dir, f"rep_{rep:02d}_best-ed.pt")
        meta_ed = {
            "criterion": "best-ed",
            "dataset": dataset_name,
            "h1": h1,
            "h2": h2,
            "best_avg_ed": result["best_avg_ed"],
            "best_ed_epoch": result["best_ed_epoch"],
            "avg_definition": "mean of available ED(Ci) at epoch",
            "rep": rep,
        }
        _save_checkpoint(ckpt_path_ed, result["best_state_ed"], meta_ed)
        print(f"[SAVE] {ckpt_path_ed} (avgED={result['best_avg_ed']:.6f} @ epoch {result['best_ed_epoch']})")

    return result


# -----------------------------
# CLI (single run; no loops)
# -----------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Single-run CoupledMLP3 trainer (no grid loops).")
    p.add_argument("--dataset", type=str, default="cifar100")
    p.add_argument("--out_dir", type=str, default="two_hidden_mlp_records")
    p.add_argument("--h1", type=int, default=1024)
    p.add_argument("--h2", type=int, default=1024)
    p.add_argument("--rep", type=int, default=0)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--augment", action="store_true", default=True)
    p.add_argument("--no-augment", dest="augment", action="store_false")
    p.add_argument("--num_workers", type=int, default=12)
    p.add_argument("--max_epochs", type=int, default=500)
    p.add_argument("--lr", type=float, default=1e-3)
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    run(
        dataset_name=args.dataset,
        out_dir=args.out_dir,
        h1=args.h1,
        h2=args.h2,
        rep=args.rep,
        batch_size=args.batch_size,
        augment=args.augment,
        num_workers=args.num_workers,
        max_epochs=args.max_epochs,
        lr=args.lr,
    )

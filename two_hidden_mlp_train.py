import os
import json
import math
import numpy as np
from typing import Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, random_split
from collections import defaultdict

# ---- Dataset loader ----
def load_dataset(
    name: str = "cifar100",
    root: str = "data_folder",
    augment: bool = True,
    batch_size: int = 128,
    num_workers: int = 4,
    val_split: float = 0.2,
    seed: int = 42,
    pin_memory: bool = True,
    drop_last: bool = False,
):
    """
    Returns:
      train_loader, val_loader, test_loader, in_dim, out_dim, sizes_dict
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
            T.Lambda(lambda t: t.view(-1)),  # -> [3072]
        ])
    else:
        train_transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean, std),
            T.Lambda(lambda t: t.view(-1)),
        ])

    test_transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean, std),
        T.Lambda(lambda t: t.view(-1)),
    ])

    # Datasets
    full_train = torchvision.datasets.CIFAR100(
        root=root, train=True, download=False, transform=train_transform
    )
    testset = torchvision.datasets.CIFAR100(
        root=root, train=False, download=False, transform=test_transform
    )

    # Split train -> train/val
    n_total = len(full_train)
    n_val = int(round(val_split * n_total))
    n_train = n_total - n_val
    g = torch.Generator().manual_seed(seed)
    trainset, valset = random_split(full_train, [n_train, n_val], generator=g)

    # DataLoaders
    common_dl_kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        persistent_workers=(num_workers > 0),
    )
    train_loader = DataLoader(trainset, shuffle=True,  **common_dl_kwargs)
    val_loader   = DataLoader(valset,   shuffle=False, **common_dl_kwargs)
    test_loader  = DataLoader(testset,  shuffle=False, **common_dl_kwargs)

    # MLP I/O dims
    in_dim = 3 * 32 * 32      # 3072
    out_dim = 100

    sizes = {"train": n_train, "val": n_val, "test": len(testset)}
    return train_loader, val_loader, test_loader, in_dim, out_dim, sizes


# -------- 3-layer MLP (2 hidden) with coupling layers (bias-free linear) --------
class CoupledMLP3(nn.Module):
    """
    Two hidden layers. Bias-free square 'coupling' Linear(d,d, bias=False)
    inserted after each hidden ReLU.
      x -> fc1 -> BN -> ReLU -> C1 -> fc2 -> BN -> ReLU -> C2 -> fc_out
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
        h1 = F.relu(self.bn1(self.fc1(x)))
        h1 = self.C1(h1)  # coupling layer #1

        h2 = F.relu(self.bn2(self.fc2(h1)))
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


# ---- ED helpers ----
def _phi(x: torch.Tensor) -> torch.Tensor:
    return torch.exp(-0.5 * x**2) / math.sqrt(2*math.pi)

def _Phi(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

@torch.no_grad()
def energy_distance_to_gaussian_from_C(
    C: torch.Tensor,
    robust: bool = True,
    max_pairs_exact: int = 50_000_000,
    subsample_pairs: int = 5_000_000,
    seed: int = 1337,
) -> float:
    """
    Energy distance between the empirical distribution of off-diagonal entries of C
    (after robust standardization) and N(0,1). Lower is better; 0 = perfect match.
    """
    C = C.detach().float()
    if C.ndim != 2 or C.shape[0] != C.shape[1]:
        raise ValueError("C must be square (d x d).")
    d = C.shape[0]
    if d < 2:
        return float('nan')

    mask = ~torch.eye(d, dtype=torch.bool, device=C.device)
    x = C[mask]  # n = d*(d-1)
    n = x.numel()
    if n < 8:
        return float('nan')

    if robust:
        med = x.median()
        mad = (x - med).abs().median().clamp(min=1e-12)
        scale = 1.4826 * mad
        z = (x - med) / scale
    else:
        mu = x.mean()
        sd = x.std(unbiased=False).clamp(min=1e-12)
        z = (x - mu) / sd

    # E|Z - G| where G ~ N(0,1)
    EzG = 2.0 * _phi(z) + z * (2.0 * _Phi(z) - 1.0)
    A = 2.0 * EzG.mean()

    z = z.to(torch.float64)
    pair_budget = n * (n - 1)
    if pair_budget <= max_pairs_exact:
        B_sum = 0.0
        count = 0
        chunk = min(n, 8192)
        for start in range(0, n, chunk):
            z1 = z[start:start+chunk].unsqueeze(1)
            diff = (z1 - z.unsqueeze(0)).abs()
            B_sum += diff.sum().item()
            count += diff.numel()
        # remove diagonal pairs
        count -= n
        B = B_sum / count
    else:
        g = torch.Generator(device=C.device).manual_seed(seed)
        m = min(subsample_pairs, pair_budget)
        i = torch.randint(0, n, (m,), generator=g, device=C.device)
        j = torch.randint(0, n-1, (m,), generator=g, device=C.device)
        j = j + (j >= i).to(j.dtype)
        B = (z[i] - z[j]).abs().mean().item()

    Cconst = 2.0 / math.sqrt(math.pi)
    ED = float((A - B - Cconst).item() if isinstance(A, torch.Tensor) else (A - B - Cconst))
    return ED


# ---- Training utils ----
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

        # ---- ED tracking (dynamic number of couplings)
        avg_ed = float("inf")
        if has_couplings:
            Cs = model.coupling_matrices()  # tuple of C_i as W^T
            ed_values = []
            for idx, C in enumerate(Cs, start=1):
                ed_val = energy_distance_to_gaussian_from_C(C.cpu())
                ed_history[f"C{idx}"].append(ed_val)
                if np.isfinite(ed_val):
                    ed_values.append(ed_val)
            # maintain alignment for any missing later keys (if any)
            max_k = len(Cs)
            for k in list(ed_history.keys()):
                if k not in [f"C{i}" for i in range(1, max_k+1)] and len(ed_history[k]) < len(val_acc_history):
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
            parts = []
            for k in sorted(ed_history.keys()):
                parts.append(f"{k}={ed_history[k][-1]:.4f}")
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


# ---- Saving helpers ----
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


# ---- Run wrapper ----
def run(
    dataset_name="cifar100",
    out_dir="runs",
    h1=1024, h2=512, rep: int = 0,
    batch_size: int = 128,
    augment: bool = True,
    num_workers: int = 12,
    max_epochs: int = 200,
):
    _safe_makedirs(out_dir)

    # Get loaders + dims
    train_loader, val_loader, test_loader, in_dim, out_dim, sizes = load_dataset(
        name=dataset_name,
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
    )

    # Save per-run histories
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

    return result


# ---- Grid launcher with best-model tracking per (dataset, h1, h2) ----
if __name__ == "__main__":
    out_dir = "two_hidden_mlp_records"  # keep same root
    dataset_name = "cifar100"
    # trackers for best across repetitions
    # keys are (dataset, h1, h2)
    best_by_acc: Dict[Tuple[str,int,int], Dict[str, Any]] = {}
    best_by_ed:  Dict[Tuple[str,int,int], Dict[str, Any]] = {}

    for h1 in [1536, 3072, 4608, 6144]:
        for h2 in [1536, 3072, 4608, 6144]:
            for rep in range(1):
                res = run(dataset_name, out_dir=out_dir, h1=h1, h2=h2, rep=rep)

                key = (dataset_name, h1, h2)

                # ----- best by validation accuracy -----
                cur_best_acc = best_by_acc.get(key, {"best_val_acc": -float("inf")})["best_val_acc"]
                if res["best_val_acc"] > cur_best_acc and res["best_state_acc"] is not None:
                    best_by_acc[key] = {
                        "best_val_acc": res["best_val_acc"],
                        "best_val_epoch": res["best_val_epoch"],
                        "state_dict": res["best_state_acc"],
                    }
                    # Save/overwrite checkpoint
                    ckpt_dir = os.path.join(out_dir, "checkpoints", dataset_name)
                    _safe_makedirs(ckpt_dir)
                    ckpt_path = os.path.join(ckpt_dir, f"h1_{h1}_h2_{h2}_best-acc.pt")
                    meta = {
                        "criterion": "best-acc",
                        "dataset": dataset_name,
                        "h1": h1,
                        "h2": h2,
                        "best_val_acc": res["best_val_acc"],
                        "best_val_epoch": res["best_val_epoch"],
                    }
                    _save_checkpoint(ckpt_path, res["best_state_acc"], meta)
                    print(f"[SAVE] {ckpt_path} (val_acc={res['best_val_acc']:.4f} @ epoch {res['best_val_epoch']})")

                # ----- best by average ED -----
                cur_best_ed = best_by_ed.get(key, {"best_avg_ed": float("inf")})["best_avg_ed"]
                if np.isfinite(res["best_avg_ed"]) and res["best_avg_ed"] < cur_best_ed and res["best_state_ed"] is not None:
                    best_by_ed[key] = {
                        "best_avg_ed": res["best_avg_ed"],
                        "best_ed_epoch": res["best_ed_epoch"],
                        "state_dict": res["best_state_ed"],
                    }
                    # Save/overwrite checkpoint
                    ckpt_dir = os.path.join(out_dir, "checkpoints", dataset_name)
                    _safe_makedirs(ckpt_dir)
                    ckpt_path = os.path.join(ckpt_dir, f"h1_{h1}_h2_{h2}_best-ed.pt")
                    meta = {
                        "criterion": "best-ed",
                        "dataset": dataset_name,
                        "h1": h1,
                        "h2": h2,
                        "best_avg_ed": res["best_avg_ed"],
                        "best_ed_epoch": res["best_ed_epoch"],
                        "avg_definition": "mean of available ED(Ci) at epoch",
                    }
                    _save_checkpoint(ckpt_path, res["best_state_ed"], meta)
                    print(f"[SAVE] {ckpt_path} (avgED={res['best_avg_ed']:.6f} @ epoch {res['best_ed_epoch']})")

    print("Done.")

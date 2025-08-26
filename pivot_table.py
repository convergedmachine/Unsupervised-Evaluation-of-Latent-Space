import os
import json
import math
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Any, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

H_LIST = [256, 512, 1024, 2048, 4096]
datasets = ["mnist_basic", "mnist_rotated", "mnist_background_random", "mnist_background_images"]

# ---- Dataset loader (your provided function) ----
def load_dataset(name:str):
    from datasets.larochelle_etal_2007.dataset import (
        MNIST_Basic, MNIST_BackgroundImages, MNIST_BackgroundRandom,
        MNIST_Rotated
    )
    dmap = {
        "mnist_basic": MNIST_Basic,
        "mnist_rotated": MNIST_Rotated,
        "mnist_background_random": MNIST_BackgroundRandom,
        "mnist_background_images": MNIST_BackgroundImages,
    }
    D = dmap[name]()
    X = D.latent_structure_task()
    y = D._labels.copy()
    return X, y

# -------- 4-layer MLP with coupling layers (bias-free linear) --------
class CoupledMLP4(nn.Module):
    """
    Four-layer MLP (in -> h1 -> h2 -> h3 -> out) with BatchNorm+ReLU and
    bias-free Linear couplings C1, C2, C3 after each hidden.
    """
    def __init__(self, in_dim: int, h1: int, h2: int, h3: int, out_dim: int):
        super().__init__()
        # main layers
        self.fc1 = nn.Linear(in_dim, h1, bias=True)
        self.bn1 = nn.BatchNorm1d(h1)
        self.fc2 = nn.Linear(h1, h2, bias=True)
        self.bn2 = nn.BatchNorm1d(h2)
        self.fc3 = nn.Linear(h2, h3, bias=True)
        self.bn3 = nn.BatchNorm1d(h3)
        self.fc4 = nn.Linear(h3, out_dim, bias=True)
        # coupling layers (no bias)
        self.C1 = nn.Linear(h1, h1, bias=False)
        self.C2 = nn.Linear(h2, h2, bias=False)
        self.C3 = nn.Linear(h3, h3, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h1 = F.relu(self.bn1(self.fc1(x))); h1 = self.C1(h1)
        h2 = F.relu(self.bn2(self.fc2(h1))); h2 = self.C2(h2)
        h3 = F.relu(self.bn3(self.fc3(h2))); h3 = self.C3(h3)
        logits = self.fc4(h3)
        return logits

    def coupling_matrices(self):
        # return transposed weights to match h -> C h convention
        return (self.C1.weight.T.detach(),
                self.C2.weight.T.detach(),
                self.C3.weight.T.detach())

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

# ---------- helpers ----------
def _parse_hs_from_fname(fname: str) -> Tuple[int, int, int]:
    """
    Expect patterns like:
      h1_<H1>_h2_<H2>_h3_<H3>_best-acc.pt
      h1_<H1>_h2_<H2>_h3_<H3>_best-ed.pt
    Robust parsing: split by underscores and search tokens.
    """
    parts = fname.replace(".pt","").split("_")
    # e.g., ['h1','1024','h2','512','h3','256','best-acc']
    try:
        i1 = parts.index("h1"); h1 = int(parts[i1+1])
        i2 = parts.index("h2"); h2 = int(parts[i2+1])
        i3 = parts.index("h3"); h3 = int(parts[i3+1])
        return h1, h2, h3
    except Exception as e:
        raise ValueError(f"Cannot parse h1,h2,h3 from filename: {fname}") from e

def _load_best_checkpoints(root_ckpt_dir: str) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """
    Returns:
      best_ckpts[dataset]['best-acc'] = dict(path, h1, h2, h3, meta)
      best_ckpts[dataset]['best-ed']  = dict(path, h1, h2, h3, meta)
    If multiple candidates exist, we pick by meta if available; else first.
    """
    best_ckpts: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for ds in datasets:
        ds_dir = os.path.join(root_ckpt_dir, ds)
        if not os.path.isdir(ds_dir):
            continue
        acc_list = [f for f in os.listdir(ds_dir) if f.endswith("_best-acc.pt")]
        ed_list  = [f for f in os.listdir(ds_dir) if f.endswith("_best-ed.pt")]

        def _pick_best(file_list: List[str], criterion: str):
            best = None
            best_score = None
            for fname in file_list:
                fpath = os.path.join(ds_dir, fname)
                try:
                    ckpt = torch.load(fpath, weights_only=False, map_location="cpu")
                    meta = ckpt.get("meta", {})
                    if criterion == "best-acc":
                        score = float(meta.get("best_val_acc", float("-inf")))
                    else:
                        # lower ED is better → maximize negative ED
                        score = -float(meta.get("best_avg_ed", float("inf")))
                    if best is None or score > best_score:
                        h1, h2, h3 = _parse_hs_from_fname(fname)
                        best = {"path": fpath, "h1": h1, "h2": h2, "h3": h3, "meta": meta}
                        best_score = score
                except Exception:
                    # If reading meta fails, still consider file (fallback)
                    h1, h2, h3 = _parse_hs_from_fname(fname)
                    if best is None:
                        best = {"path": fpath, "h1": h1, "h2": h2, "h3": h3, "meta": {}}
                        best_score = -1e9
            return best

        acc_best = _pick_best(acc_list, "best-acc") if acc_list else None
        ed_best  = _pick_best(ed_list,  "best-ed")  if ed_list  else None
        if acc_best or ed_best:
            best_ckpts.setdefault(ds, {})
            if acc_best: best_ckpts[ds]["best-acc"] = acc_best
            if ed_best:  best_ckpts[ds]["best-ed"]  = ed_best
    return best_ckpts

@torch.no_grad()
def _evaluate_on_dataset(model: nn.Module, dataset: str) -> float:
    X, y = load_dataset(dataset)
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)
    logits = model(X)
    acc = (logits.argmax(dim=1) == y).float().mean().item()
    return acc

def _rebuild_model_for_ckpt(ckpt_payload: Dict[str, Any], in_dim: int, out_dim: int,
                            h1: int, h2: int, h3: int) -> nn.Module:
    model = CoupledMLP4(in_dim, h1, h2, h3, out_dim)
    state = ckpt_payload["state_dict"] if "state_dict" in ckpt_payload else ckpt_payload
    model.load_state_dict(state)
    model.eval()
    return model

def _model_dims_for_dataset(dataset: str) -> Tuple[int, int]:
    X, y = load_dataset(dataset)
    in_dim = X.shape[1]
    out_dim = len(np.unique(y))
    return in_dim, out_dim

def _model_avg_ed(model: nn.Module) -> float:
    Cs = model.coupling_matrices()
    eds = []
    for C in Cs:
        if C is not None:
            eds.append(energy_distance_to_gaussian_from_C(C))
    return float(np.nanmean(eds)) if eds else float("nan")

# ---------- main cross-eval ----------
def cross_evaluate(root_ckpt_dir="three_hidden_mlp_records/checkpoints") -> pd.DataFrame:
    """
    For each source dataset, load its best-acc and best-ed model,
    then evaluate on every target dataset. Return a tidy DataFrame:
      [Source, ModelType, h1, h2, h3, Target, Acc, AvgED]
    Note: AvgED is model-intrinsic (same across targets); repeated per row for convenience.
    """
    best_ckpts = _load_best_checkpoints(root_ckpt_dir)
    rows = []

    for source_ds in datasets:
        if source_ds not in best_ckpts:
            continue
        for model_type in ["best-acc", "best-ed"]:
            if model_type not in best_ckpts[source_ds]:
                continue

            entry = best_ckpts[source_ds][model_type]
            path, h1, h2, h3 = entry["path"], entry["h1"], entry["h2"], entry["h3"]

            # Build model using source dataset dims (in/out)
            in_dim, out_dim = _model_dims_for_dataset(source_ds)
            ckpt_payload = torch.load(path, weights_only=False, map_location="cpu")
            model = _rebuild_model_for_ckpt(ckpt_payload, in_dim, out_dim, h1, h2, h3)
            avg_ed = _model_avg_ed(model)

            # Evaluate on every target
            for target_ds in datasets:
                acc = _evaluate_on_dataset(model, target_ds)
                rows.append({
                    "Source": source_ds,
                    "ModelType": model_type,   # 'best-acc' or 'best-ed'
                    "h1": h1,
                    "h2": h2,
                    "h3": h3,
                    "Target": target_ds,
                    "Acc": acc,
                    "AvgED": avg_ed,          # same across targets for a fixed model
                })

    df = pd.DataFrame(rows)
    return df

if __name__ == "__main__":
    df = cross_evaluate("three_hidden_mlp_records/checkpoints")
    # Pretty print long form
    if not df.empty:
        print(df.sort_values(["Source", "ModelType", "Target"]).to_string(index=False))
    else:
        print("[WARN] No checkpoints found / DataFrame is empty.")

    # Desired orders
    row_order = ["mnist_basic", "mnist_rotated",
                 "mnist_background_random", "mnist_background_images"]
    col_order = row_order                         # same order for Source
    model_order = ["best-acc", "best-ed"]

    if not df.empty:
        # Set categorical dtypes
        df["Target"]    = pd.Categorical(df["Target"], categories=row_order, ordered=True)
        df["Source"]    = pd.Categorical(df["Source"], categories=col_order, ordered=True)
        df["ModelType"] = pd.Categorical(df["ModelType"], categories=model_order, ordered=True)

        # Pivot (Target rows; Source/Model columns)
        pivot = df.pivot_table(index="Target",
                               columns=["Source", "ModelType"],
                               values="Acc",
                               aggfunc="mean")

        # Sort by categorical order on both axes
        pivot = pivot.sort_index(axis=0).sort_index(axis=1)

        # (Optional) hard reindex columns to ensure exact grid even if some combos are missing)
        full_cols = pd.MultiIndex.from_product([col_order, model_order], names=["Source", "ModelType"])
        pivot = pivot.reindex(columns=full_cols)

        print("\n=== Accuracy pivot (Target rows; Source/Model columns) ===")
        print(pivot.to_string())

        # Optionally: include h3 in a separate analysis, e.g., groupby
        # g = df.groupby(["Source","ModelType","h3"])["Acc"].mean()
        # print("\n=== Mean Acc by Source/Model/h3 ===\n", g)

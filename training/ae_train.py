import argparse, os, math, json, random
from dataclasses import dataclass
from typing import Optional, List
import numpy as np
import torch
import torch.nn as nn
import torchvision
from sklearn.model_selection import KFold
import optuna
from optuna.samplers import RandomSampler, TPESampler

# ---------------------------
# Utilities
# ---------------------------
def set_seed(seed:int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def _make_activation(name: str) -> nn.Module:
    return nn.Tanh() if name == 'tanh' else nn.Sigmoid()

def _init_first_layer(w: torch.Tensor, dist: str, scale_heur: str, n_in: int, scale_mult: Optional[float]):
    if scale_heur == 'Glorot':
        nn.init.xavier_uniform_(w) if dist == 'uniform' else nn.init.xavier_normal_(w)
    elif scale_heur == 'old':
        s = (scale_mult if scale_mult is not None else 1.0) / math.sqrt(max(1.0, n_in))
        if dist == 'uniform':
            nn.init.uniform_(w, -s, s)
        else:
            nn.init.normal_(w, 0.0, s)
    else:
        raise ValueError(scale_heur)

# ---------------------------
# 7-Layer Autoencoder with coupling layers after every hidden block
# ---------------------------
class Autoencoder7LayerCL(nn.Module):
    """
    Symmetric AE with 7 hidden layers total:
    Encoder:  x -> fc1(h1) -> act -> cl1 -> fc2(h2) -> act -> cl2 -> fc3(h3) -> act -> cl3 -> fc4(z) -> act -> cl4
    Decoder:  -> fc5(h3) -> act -> cl5 -> fc6(h2) -> act -> cl6 -> fc7(h1) -> act -> cl7 -> fc_out(D) -> linear

    Coupling layers are square (bias=False) and used in forward pass for tracking.
    """
    def __init__(self, n_in:int, h1:int, h2:int, h3:int, z:int,
                 activation:str, dist1:str, scale_heur1:str, scale_mult1: Optional[float]):
        super().__init__()
        self.act = _make_activation(activation)

        # Encoder
        self.fc1 = nn.Linear(n_in, h1, bias=True)
        self.bn1 = nn.BatchNorm1d(h1)
        self.cl1 = nn.Linear(h1, h1, bias=False)

        self.fc2 = nn.Linear(h1, h2, bias=True)
        self.bn2 = nn.BatchNorm1d(h2)
        self.cl2 = nn.Linear(h2, h2, bias=False)

        self.fc3 = nn.Linear(h2, h3, bias=True)
        self.bn3 = nn.BatchNorm1d(h3)
        self.cl3 = nn.Linear(h3, h3, bias=False)

        self.fc4 = nn.Linear(h3, z,  bias=True)
        self.cl4 = nn.Linear(z,  z,  bias=False)

        # Decoder
        self.fc5 = nn.Linear(z,  h3, bias=True)
        self.bn5 = nn.BatchNorm1d(h3)
        self.cl5 = nn.Linear(h3, h3, bias=False)

        self.fc6 = nn.Linear(h3, h2, bias=True)
        self.bn6 = nn.BatchNorm1d(h2)
        self.cl6 = nn.Linear(h2, h2, bias=False)

        self.fc7 = nn.Linear(h2, h1, bias=True)
        self.bn7 = nn.BatchNorm1d(h1)
        self.cl7 = nn.Linear(h1, h1, bias=False)

        self.fc_out = nn.Linear(h1, n_in, bias=True)  # linear output

        # Init first layer as requested; defaults elsewhere
        _init_first_layer(self.fc1.weight, dist1, scale_heur1, n_in, scale_mult1)
        nn.init.zeros_(self.fc1.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h1 = self.act(self.bn1(self.fc1(x))); h1 = self.cl1(h1)
        h2 = self.act(self.bn2(self.fc2(h1))); h2 = self.cl2(h2)
        h3 = self.act(self.bn3(self.fc3(h2))); h3 = self.cl3(h3)
        z  = self.fc4(h3); z  = self.cl4(z)
        d3 = self.act(self.bn5(self.fc5(z)));  d3 = self.cl5(d3)
        d2 = self.act(self.bn6(self.fc6(d3))); d2 = self.cl6(d2)
        d1 = self.act(self.bn7(self.fc7(d2))); d1 = self.cl7(d1)
        xhat = self.fc_out(d1)
        return xhat

# -----------------------------
# Core utilities
# -----------------------------
@torch.no_grad()
def _phi(x: torch.Tensor) -> torch.Tensor:
    return torch.exp(-0.5 * x**2) / math.sqrt(2 * math.pi)

@torch.no_grad()
def _Phi(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

@torch.no_grad()
def robust_standardize(x: torch.Tensor) -> torch.Tensor:
    x = x.to(torch.float32)
    med = x.median()
    mad = (x - med).abs().median().clamp(min=1e-12)
    return (x - med) / (1.4826 * mad)

@torch.no_grad()
def meanstd_standardize(x: torch.Tensor) -> torch.Tensor:
    x = x.to(torch.float32)
    mu = x.mean()
    sd = x.std(unbiased=True).clamp(min=1e-12)
    return (x - mu) / sd

@torch.no_grad()
def offdiag_values(C: torch.Tensor, mode: str = "all") -> torch.Tensor:
    n = C.shape[0]
    if mode == "upper":
        iu = torch.triu_indices(n, n, offset=1, device=C.device)
        return C[iu[0], iu[1]].to(torch.float32)
    elif mode == "all":
        mask = ~torch.eye(n, dtype=torch.bool, device=C.device)
        return C[mask].flatten().to(torch.float32)
    else:
        raise ValueError("mode must be 'upper' or 'all'")

@torch.no_grad()
def expected_abs_diff_standard_norm() -> float:
    return 2.0 / math.sqrt(math.pi)

@torch.no_grad()
def E_abs_x_minus_Z(x: torch.Tensor) -> torch.Tensor:
    return 2.0 * _phi(x) + x * (2.0 * _Phi(x) - 1.0)

@torch.no_grad()
def pairwise_abs_mean_u_stat(x: torch.Tensor) -> torch.Tensor:
    m = x.numel()
    if m < 2:
        return torch.tensor(float("nan"), dtype=torch.float32)
    xs, _ = torch.sort(x.view(-1))
    idx = torch.arange(1, m + 1, dtype=torch.float32, device=x.device)
    coef = 2.0 * idx - m - 1.0
    s = (coef * xs).sum()
    mean_pair_abs = (2.0 * s) / (m * (m - 1))
    return mean_pair_abs

@torch.no_grad()
def energy_distance_to_standard_normal(x: torch.Tensor, standardize: str = "robust") -> float:
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

# ---------------------------
# Training (reconstruction loss only)
# ---------------------------
def train_eval_once_ae(model: nn.Module,
                       Xtr: np.ndarray,
                       Xva: np.ndarray,
                       device: str,
                       trial: optuna.trial.Trial,
                       max_epochs: int = 200,
                       step_base: int = 0) -> float:
    model = model.to(device)
    lr = trial.suggest_float('lr', 1e-6, 1.0, log=True)
    lr_anneal_start = trial.suggest_int('lr_anneal_start', 100, 10000, log=True)
    bs = trial.suggest_categorical('batch_size', [20, 100, 256, 512])
    l2_choice = trial.suggest_categorical('l2_penalty_choice', ['zero', 'nz'])
    l2_penalty = 0.0 if l2_choice == 'zero' else trial.suggest_float('l2_penalty_nz', 1e-8, 1e-2, log=True)

    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.0, weight_decay=l2_penalty)
    loss_fn = nn.MSELoss(reduction='mean')

    Xtr_t = torch.tensor(Xtr, dtype=torch.float32, device=device)
    Xva_t = torch.tensor(Xva, dtype=torch.float32, device=device)

    n = Xtr_t.shape[0]
    global_step = step_base

    def annealed_lr(base_lr: float, t: int, t0: int) -> float:
        return base_lr if t < t0 else base_lr * (t0 / float(t + 1))

    for epoch in range(1, max_epochs + 1):
        idx = torch.randperm(n, device=device)
        model.train()
        for start in range(0, n, bs):
            end = min(n, start + bs)
            xb = Xtr_t[idx[start:end]]
            cur_lr = annealed_lr(lr, global_step, lr_anneal_start)
            for pg in opt.param_groups:
                pg['lr'] = cur_lr
            opt.zero_grad(set_to_none=True)
            xhat = model(xb)
            loss = loss_fn(xhat, xb)
            loss.backward()
            opt.step()
            global_step += 1

    model.eval()
    with torch.no_grad():
        xhat = model(Xva_t)
        val_loss = loss_fn(xhat, Xva_t).item()
    return float(val_loss)

# ---------------------------
# Objective: minimize reconstruction error + log ED metrics
# ---------------------------
def objective_ae(trial, X: np.ndarray, n_splits:int=5, max_epochs:int=200, device:str='cpu'):
    seed = trial.suggest_categorical('iseed', [5,6,7,8])
    set_seed(seed)

    # Compute mean and std per feature (along batch dimension, axis=0)
    mean = np.mean(X, axis=0)  # Shape: (num_features,)
    std = np.std(X, axis=0)   # Shape: (num_features,)

    # Avoid division by zero by setting std=0 to std=1
    std = np.where(std == 0, 1.0, std)

    # Standardize: (X - mean) / std
    Xp = (X - mean) / std
    
    n_in = Xp.shape[1]

    # Hidden sizes (7 hidden: h1,h2,h3,z,h3,h2,h1)
    h1 = trial.suggest_int('nhid1', 128, 1024, step=16)
    h2 = trial.suggest_int('nhid2', 128, 1024, step=16)
    h3 = trial.suggest_int('nhid3', 128, 1024, step=16)
    z  = trial.suggest_int('ncode',  128, 1024, step=16)

    squash = trial.suggest_categorical('squash', ['tanh','logistic'])
    dist1 = trial.suggest_categorical('dist1', ['uniform','normal'])
    scale_heur1 = trial.suggest_categorical('scale_heur1', ['old','Glorot'])
    scale_mult1 = trial.suggest_float('scale_mult1', 0.2, 2.0) if scale_heur1 == 'old' else None

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    losses: List[float] = []

    # Track ED of every coupling layer across folds
    ed_lists = {k: [] for k in ['cl1','cl2','cl3','cl4','cl5','cl6','cl7']}

    for fold_idx, (tr_idx, va_idx) in enumerate(kf.split(Xp)):
        Xtr, Xva = Xp[tr_idx], Xp[va_idx]

        model = Autoencoder7LayerCL(
            n_in=n_in, h1=h1, h2=h2, h3=h3, z=z,
            activation=squash, dist1=dist1, scale_heur1=scale_heur1, scale_mult1=scale_mult1
        )

        val_loss = train_eval_once_ae(
            model, Xtr, Xva, device, trial,
            max_epochs=max_epochs, step_base=fold_idx * max_epochs
        )
        losses.append(val_loss)

        # Energy distance diagnostics
        for name in ed_lists.keys():
            W = getattr(model, name).weight
            ed_lists[name].append(energy_distance_to_standard_normal(W))

    mean_loss = float(np.mean(losses))
    trial.set_user_attr('val_recon_loss_mean', mean_loss)
    trial.set_user_attr('val_recon_loss_std', float(np.std(losses)))
    # Log per-layer ED means + overall mean
    ed_means = {k: float(np.nanmean(v)) if len(v) else float('nan') for k, v in ed_lists.items()}
    for k, v in ed_means.items():
        trial.set_user_attr(f'ed_{k}_mean', v)
    trial.set_user_attr('ed_overall_mean', float(np.nanmean(list(ed_means.values()))))

    return mean_loss

# ---------------------------
# Dataset loading (Larochelle + Torchvision)
# ---------------------------
def load_dataset(name: str, root: str = "data_folder"):
    import numpy as np
    from datasets.larochelle_etal_2007.dataset import (
        MNIST_Basic, MNIST_BackgroundImages, MNIST_BackgroundRandom,
        MNIST_Rotated, MNIST_RotatedBackgroundImages,
        Rectangles, RectanglesImages, Convex
    )
    larochelle_map = {
        "mnist_basic": MNIST_Basic,
        "mnist_background_images": MNIST_BackgroundImages,
        "mnist_background_random": MNIST_BackgroundRandom,
        "mnist_rotated": MNIST_Rotated,
        "mnist_rotated_background_images": MNIST_RotatedBackgroundImages,
        "rectangles": Rectangles,
        "rectangles_images": RectanglesImages,
        "convex": Convex,
    }

    if name in larochelle_map:
        D = larochelle_map[name]()
        X = D.latent_structure_task().astype(np.float32)
        y = D._labels.copy().astype(np.int64)
        if X.ndim > 2:
            X = X.reshape(X.shape[0], -1)
        return X, y

    if name == "cifar10":
        tr = torchvision.datasets.CIFAR10(root=root, train=True, download=True)
        te = torchvision.datasets.CIFAR10(root=root, train=False, download=True)
        X = np.concatenate([tr.data, te.data], axis=0)
        y = np.array(tr.targets + te.targets, dtype=np.int64)
        return X.reshape(X.shape[0], -1).astype(np.float32), y
    if name == "cifar100":
        tr = torchvision.datasets.CIFAR100(root=root, train=True, download=True)
        te = torchvision.datasets.CIFAR100(root=root, train=False, download=True)
        X = np.concatenate([tr.data, te.data], axis=0)
        y = np.array(tr.targets + te.targets, dtype=np.int64)
        return X.reshape(X.shape[0], -1).astype(np.float32), y
    if name == "fashion_mnist":
        tr = torchvision.datasets.FashionMNIST(root=root, train=True, download=True)
        te = torchvision.datasets.FashionMNIST(root=root, train=False, download=True)
        X = np.concatenate([tr.data.numpy(), te.data.numpy()], axis=0).astype(np.float32)
        y = np.concatenate([tr.targets.numpy(), te.targets.numpy()], axis=0).astype(np.int64)
        return X.reshape(X.shape[0], -1), y

    raise ValueError(f"Unknown dataset: {name}")

# ---------------------------
# Reporting
# ---------------------------
def save_trials(study: optuna.Study, out_dir: str, dataset_name:str):
    df = study.trials_dataframe()
    csv_path = os.path.join(out_dir, f"{dataset_name}_trials.csv")
    df.to_csv(csv_path, index=False)
    return csv_path

def save_best(study: optuna.Study, out_dir: str, dataset_name:str):
    best_json = os.path.join(out_dir, f"{dataset_name}_best.json")
    bt = study.best_trial
    with open(best_json, "w") as f:
        json.dump({
            "best_value": study.best_value,           # mean CV recon MSE
            "best_params": study.best_params,
            # handy diagnostics
            "val_recon_loss_mean": bt.user_attrs.get('val_recon_loss_mean', None),
            "val_recon_loss_std": bt.user_attrs.get('val_recon_loss_std', None),
            "ed_overall_mean": bt.user_attrs.get('ed_overall_mean', None),
            **{k: bt.user_attrs.get(k) for k in [
                'ed_cl1_mean','ed_cl2_mean','ed_cl3_mean','ed_cl4_mean','ed_cl5_mean','ed_cl6_mean','ed_cl7_mean'
            ]},
        }, f, indent=2)
    return best_json

# ---------------------------
# Main
# ---------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dataset",
        type=str,
        default="rectangles_images",
        choices=[
            "mnist_basic","mnist_background_images","mnist_background_random",
            "mnist_rotated","mnist_rotated_background_images",
            "rectangles","rectangles_images","convex",
            "fashion_mnist","cifar10","cifar100",
        ],
    )
    ap.add_argument("--optimization", type=str, default="random", choices=["random","parzen"])
    ap.add_argument("--trials", type=int, default=96)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", type=str, default="runs/ae_experiment")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--cv", type=int, default=3, help="Stratified K-folds")
    ap.add_argument("--cl", action='store_true', help="Use CL model (default: False)")
    ap.add_argument("--epochs", type=int, default=200)    
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    set_seed(args.seed)

    X, _ = load_dataset(args.dataset)

    sampler = RandomSampler(seed=args.seed) if args.optimization == "random" else TPESampler(
        multivariate=True, group=True, n_startup_trials=min(20, args.trials//5), seed=1234
    )

    study = optuna.create_study(direction='minimize', sampler=sampler)
    study.optimize(
        lambda t: objective_ae(t, X, n_splits=args.cv, max_epochs=args.epochs, device=args.device),
        n_trials=args.trials
    )

    csv_path = save_trials(study, args.out, args.dataset)
    best_json = save_best(study, args.out, args.dataset)
    print(f"Trials saved to {csv_path}")
    print(f"Best trial saved to {best_json}")

if __name__ == "__main__":
    main()

import argparse, os, math, json, random
from dataclasses import dataclass
from typing import Optional, List, Tuple
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import optuna
from optuna.samplers import RandomSampler, TPESampler

# ---------------------------
# Utilities
# ---------------------------
def set_seed(seed:int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

@dataclass
class PrepResult:
    X: np.ndarray
    preproc: str
    info: dict

def preprocess(trial, X: np.ndarray) -> PrepResult:
    choice = trial.suggest_categorical('preproc', ['raw','colnorm','pca'])
    info = {}
    Xp = X
    if choice == 'colnorm':
        thresh = trial.suggest_float('colnorm_thresh', 1e-9, 1e-3, log=True)
        std = X.std(axis=0, ddof=0)
        mask = std > thresh
        if mask.sum() == 0:
            mask = np.ones_like(std, dtype=bool)
        scaler = StandardScaler(with_mean=True, with_std=True)
        Xp = scaler.fit_transform(X[:, mask])
        info.update({'thresh': float(thresh), 'kept': int(mask.sum())})
    elif choice == 'pca':
        energy = trial.suggest_float('pca_energy', 0.5, 1.0)
        scaler = StandardScaler(with_mean=True, with_std=True)
        Xs = scaler.fit_transform(X)
        pca = PCA(n_components=energy, svd_solver='full')
        Xp = pca.fit_transform(Xs)
        info.update({'energy': float(energy), 'n_components': int(getattr(pca, "n_components_", Xp.shape[1]))})
    else:
        pass
    return PrepResult(X=Xp, preproc=choice, info=info)

# ---------------------------
# Models (3-layer variants)
# ---------------------------
def _make_activation(name: str) -> nn.Module:
    if name == 'tanh':
        return nn.Tanh()
    elif name == 'logistic':
        return nn.Sigmoid()
    raise ValueError(name)

def _init_first_layer(w: torch.Tensor, dist: str, scale_heur: str, n_in: int, scale_mult: Optional[float]):
    if scale_heur == 'Glorot':
        nn.init.xavier_uniform_(w) if dist == 'uniform' else nn.init.xavier_normal_(w)
    elif scale_heur == 'old':
        fan_in = n_in
        s = (scale_mult if scale_mult is not None else 1.0) / math.sqrt(max(1.0, fan_in))
        if dist == 'uniform':
            nn.init.uniform_(w, -s, s)
        else:
            nn.init.normal_(w, 0.0, s)
    else:
        raise ValueError(scale_heur)

class ThreeLayerMLP(nn.Module):
    """
    x -> fc1 -> act -> fc2 -> act -> fc3 -> act -> fc_out
    (no coupling layers)
    """
    def __init__(self, n_in:int, h1:int, h2:int, h3:int, n_out:int,
                 activation:str, dist1:str, scale_heur1:str, scale_mult1: Optional[float]):
        super().__init__()
        self.fc1 = nn.Linear(n_in, h1, bias=True)
        self.fc2 = nn.Linear(h1, h2, bias=True)
        self.fc3 = nn.Linear(h2, h3, bias=True)
        self.fc_out = nn.Linear(h3, n_out, bias=True)
        self.act = _make_activation(activation)

        # init first layer per requested heuristic/distribution
        _init_first_layer(self.fc1.weight, dist1, scale_heur1, n_in, scale_mult1)
        nn.init.zeros_(self.fc1.bias)

        # other layers: default PyTorch init is fine
        nn.init.zeros_(self.fc_out.weight); nn.init.zeros_(self.fc_out.bias)  # zero-init final layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h1 = self.act(self.fc1(x))
        h2 = self.act(self.fc2(h1))
        h3 = self.act(self.fc3(h2))
        return self.fc_out(h3)

class ThreeLayerMLPcl(nn.Module):
    """
    x -> fc1 -> act -> cl1 -> fc2 -> act -> cl2 -> fc3 -> act -> cl3 -> fc_out
    """
    def __init__(self, n_in:int, h1:int, h2:int, h3:int, n_out:int,
                 activation:str, dist1:str, scale_heur1:str, scale_mult1: Optional[float]):
        super().__init__()
        self.fc1 = nn.Linear(n_in, h1, bias=True)
        self.cl1 = nn.Linear(h1, h1, bias=False)

        self.fc2 = nn.Linear(h1, h2, bias=True)
        self.cl2 = nn.Linear(h2, h2, bias=False)

        self.fc3 = nn.Linear(h2, h3, bias=True)
        self.cl3 = nn.Linear(h3, h3, bias=False)

        self.fc_out = nn.Linear(h3, n_out, bias=True)

        self.act = _make_activation(activation)

        # init first layer per requested heuristic/distribution
        _init_first_layer(self.fc1.weight, dist1, scale_heur1, n_in, scale_mult1)
        nn.init.zeros_(self.fc1.bias)

        # zero-init final layer (zero_softmax analogue)
        nn.init.zeros_(self.fc_out.weight); nn.init.zeros_(self.fc_out.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h1 = self.act(self.fc1(x))
        h1c = self.cl1(h1)

        h2 = self.act(self.fc2(h1c))
        h2c = self.cl2(h2)

        h3 = self.act(self.fc3(h2c))
        h3c = self.cl3(h3)

        return self.fc_out(h3c)

# ---------------------------
# Train / Eval (single run)
# ---------------------------
def train_eval_once(model: nn.Module,
                    Xtr: np.ndarray, ytr: np.ndarray,
                    Xva: np.ndarray, yva: np.ndarray,
                    device: str,
                    trial: Optional[optuna.trial.Trial],
                    max_epochs: int = 200,
                    step_base: int = 0) -> float:
    model = model.to(device)
    # --- Hyperparams (NIPS'11 style) ---
    lr = trial.suggest_float('lr', 1e-6, 1.0, log=True)
    lr_anneal_start = trial.suggest_int('lr_anneal_start', 100, 10000, log=True)
    bs = trial.suggest_categorical('batch_size', [20, 100])
    l2_choice = trial.suggest_categorical('l2_penalty_choice', ['zero', 'nz'])
    l2_penalty = 0.0 if l2_choice == 'zero' else trial.suggest_float('l2_penalty_nz', 1e-8, 1e-2, log=True)

    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.0, weight_decay=l2_penalty)

    Xtr_t = torch.tensor(Xtr, dtype=torch.float32, device=device)
    ytr_t = torch.tensor(ytr, dtype=torch.long, device=device)
    Xva_t = torch.tensor(Xva, dtype=torch.float32, device=device)
    yva_t = torch.tensor(yva, dtype=torch.long, device=device)

    n = Xtr_t.shape[0]
    global_step = step_base

    def annealed_lr(base_lr: float, t: int, t0: int) -> float:
        if t < t0:
            return base_lr
        return base_lr * (t0 / float(t + 1))

    for epoch in range(1, max_epochs + 1):
        idx = torch.randperm(n, device=device)
        model.train()
        for start in range(0, n, bs):
            end = min(n, start + bs)
            batch_idx = idx[start:end]
            xb = Xtr_t[batch_idx]
            yb = ytr_t[batch_idx]
            cur_lr = annealed_lr(lr, global_step, lr_anneal_start)
            for pg in opt.param_groups:
                pg['lr'] = cur_lr
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            loss.backward()
            opt.step()
            global_step += 1

    model.eval()
    with torch.no_grad():
        logits = model(Xva_t)
        pred = torch.argmax(logits, dim=1)
        acc = (pred == yva_t).float().mean().item()
    return float(acc)

# ---------------------------
# Energy distance helper
# ---------------------------
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
    C = C.detach().float()
    if C.ndim != 2 or C.shape[0] != C.shape[1]:
        raise ValueError("C must be square (d x d).")
    d = C.shape[0]
    if d < 2:
        return float('nan')
    mask = ~torch.eye(d, dtype=torch.bool, device=C.device)
    x = C[mask]
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

# ---------------------------
# Objective for Optuna
# ---------------------------
def objective(trial, X: np.ndarray, y: np.ndarray, n_splits:int=3, max_epochs:int=200, use_cl:bool=False, device:str='cpu'):
    seed = trial.suggest_categorical('iseed', [5,6,7,8])
    set_seed(seed)

    prep = preprocess(trial, X)
    Xp = prep.X
    n_in = Xp.shape[1]
    n_classes = len(np.unique(y))

    # Hidden sizes for three layers
    h1 = trial.suggest_int('nhid1', 16, 1024, step=16)
    h2 = trial.suggest_int('nhid2', 16, 1024, step=16)
    h3 = trial.suggest_int('nhid3', 16, 1024, step=16)

    squash = trial.suggest_categorical('squash', ['tanh','logistic'])
    dist1 = trial.suggest_categorical('dist1', ['uniform','normal'])
    scale_heur1 = trial.suggest_categorical('scale_heur1', ['old','Glorot'])
    scale_mult1 = None
    if scale_heur1 == 'old':
        scale_mult1 = trial.suggest_float('scale_mult1', 0.2, 2.0)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    accs: List[float] = []
    ed_c1_list: List[float] = []
    ed_c2_list: List[float] = []
    ed_c3_list: List[float] = []

    for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(Xp, y)):
        Xtr, Xva = Xp[tr_idx], Xp[va_idx]
        ytr, yva = y[tr_idx], y[va_idx]

        if use_cl:
            model = ThreeLayerMLPcl(n_in, h1, h2, h3, n_classes, squash, dist1, scale_heur1, scale_mult1)
        else:
            model = ThreeLayerMLP(n_in, h1, h2, h3, n_classes, squash, dist1, scale_heur1, scale_mult1)

        acc = train_eval_once(
            model, Xtr, ytr, Xva, yva, device, trial,
            max_epochs=max_epochs,
            step_base=fold_idx * max_epochs
        )
        accs.append(acc)

        if use_cl:
            ed1 = energy_distance_to_gaussian_from_C(model.cl1.weight)
            ed2 = energy_distance_to_gaussian_from_C(model.cl2.weight)
            ed3 = energy_distance_to_gaussian_from_C(model.cl3.weight)
            ed_c1_list.append(float(ed1))
            ed_c2_list.append(float(ed2))
            ed_c3_list.append(float(ed3))

    if use_cl and len(ed_c1_list) > 0:
        avg_ed = float(np.mean([np.mean(ed_c1_list), np.mean(ed_c2_list), np.mean(ed_c3_list)]))
        trial.set_user_attr('avg_fit_criteria', avg_ed)
        trial.set_user_attr('ed_cl1_mean', float(np.mean(ed_c1_list)))
        trial.set_user_attr('ed_cl2_mean', float(np.mean(ed_c2_list)))
        trial.set_user_attr('ed_cl3_mean', float(np.mean(ed_c3_list)))

    return float(np.mean(accs))

# ---------------------------
# Dataset loading (Larochelle + Torchvision)
# ---------------------------
def load_dataset(name: str, root: str = "data_folder"):
    """
    Returns:
        X (np.ndarray, float32, shape [N, D]), y (np.ndarray, int64, shape [N])
    Notes:
      - Torchvision datasets are concatenated (train+test), flattened, and left unnormalized.
        Your preprocessing pipeline (colnorm / PCA) will handle scaling.
      - Larochelle datasets use latent_structure_task() and labels from the dataset object.
    """
    import numpy as np

    # Larochelle et al. 2007
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

    # Torchvision datasets
    import torchvision

    if name == "cifar10":
        trainset = torchvision.datasets.CIFAR10(root=root, train=True, download=True)
        testset  = torchvision.datasets.CIFAR10(root=root, train=False, download=True)
        X = np.concatenate([trainset.data, testset.data], axis=0)
        y = np.array(trainset.targets + testset.targets, dtype=np.int64)
        X = X.reshape(X.shape[0], -1).astype(np.float32)
        return X, y

    if name == "cifar100":
        trainset = torchvision.datasets.CIFAR100(root=root, train=True, download=True)
        testset  = torchvision.datasets.CIFAR100(root=root, train=False, download=True)
        X = np.concatenate([trainset.data, testset.data], axis=0)
        y = np.array(trainset.targets + testset.targets, dtype=np.int64)
        X = X.reshape(X.shape[0], -1).astype(np.float32)
        return X, y

    if name == "fashion_mnist":
        trainset = torchvision.datasets.FashionMNIST(root=root, train=True, download=True)
        testset  = torchvision.datasets.FashionMNIST(root=root, train=False, download=True)
        X = np.concatenate([trainset.data.numpy(), testset.data.numpy()], axis=0)
        y = np.concatenate([trainset.targets.numpy(), testset.targets.numpy()], axis=0).astype(np.int64)
        X = X.reshape(X.shape[0], -1).astype(np.float32)
        return X, y

    raise ValueError(f"Unknown dataset: {name}")

# ---------------------------
# Reporting helpers
# ---------------------------
def save_trials(study: optuna.Study, out_dir: str, dataset_name:str):
    df = study.trials_dataframe()
    csv_path = os.path.join(out_dir, f"{dataset_name}_trials.csv")
    df.to_csv(csv_path, index=False)
    return csv_path

def save_best(study: optuna.Study, out_dir: str, dataset_name:str):
    best_json = os.path.join(out_dir, f"{dataset_name}_best.json")
    with open(best_json, "w") as f:
        json.dump({"best_value": study.best_value, "best_params": study.best_params}, f, indent=2)
    return best_json

def save_trials_cl(study: optuna.Study, out_dir: str, dataset_name: str):
    df = study.trials_dataframe()
    # include per-trial avg_fit_criteria (if present)
    df = df.assign(
        avg_fit_criteria=[t.user_attrs.get('avg_fit_criteria', None) for t in study.trials],
        ed_cl1_mean=[t.user_attrs.get('ed_cl1_mean', None) for t in study.trials],
        ed_cl2_mean=[t.user_attrs.get('ed_cl2_mean', None) for t in study.trials],
        ed_cl3_mean=[t.user_attrs.get('ed_cl3_mean', None) for t in study.trials],
    )
    csv_path = os.path.join(out_dir, f"{dataset_name}_trials.csv")
    df.to_csv(csv_path, index=False)
    return csv_path

def save_best_cl(study: optuna.Study, out_dir: str, dataset_name: str):
    best_json = os.path.join(out_dir, f"{dataset_name}_best.json")
    best_trial = study.best_trial
    with open(best_json, "w") as f:
        json.dump({
            "best_value": study.best_value,
            "best_params": study.best_params,
            "avg_fit_criteria": best_trial.user_attrs.get('avg_fit_criteria', None),
            "ed_cl1_mean": best_trial.user_attrs.get('ed_cl1_mean', None),
            "ed_cl2_mean": best_trial.user_attrs.get('ed_cl2_mean', None),
            "ed_cl3_mean": best_trial.user_attrs.get('ed_cl3_mean', None),
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
            # Larochelle
            "mnist_basic",
            "mnist_background_images",
            "mnist_background_random",
            "mnist_rotated",
            "mnist_rotated_background_images",
            "rectangles",
            "rectangles_images",
            "convex",
            # Torchvision
            "fashion_mnist",
            "cifar10",
            "cifar100",
        ],
    )
    ap.add_argument("--optimization", type=str, default="random",
                    choices=["random", "parzen"],)
    ap.add_argument("--trials", type=int, default=128)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", type=str, default="runs/experiment")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--cv", type=int, default=5, help="Stratified K-folds")
    ap.add_argument("--cl", action='store_true', help="Use CL model with three coupling layers (default: False)")
    ap.add_argument("--epochs", type=int, default=200)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    set_seed(args.seed)

    X, y = load_dataset(args.dataset)
    if args.optimization == "random":
        sampler = RandomSampler(seed=args.seed)
    elif args.optimization == "parzen":
        sampler = TPESampler(
            multivariate=True,
            group=True,
            n_startup_trials=min(20, args.trials // 5),
            seed=1234,
        )
    else:
        raise ValueError(f"Unknown optimization: {args.optimization}")

    study = optuna.create_study(direction='maximize', sampler=sampler)
    study.optimize(
        lambda t: objective(t, X, y, n_splits=args.cv, max_epochs=args.epochs, use_cl=args.cl, device=args.device),
        n_trials=args.trials
    )

    # Save + data exports
    if args.cl:
        csv_path = save_trials_cl(study, args.out, args.dataset)
        best_json = save_best_cl(study, args.out, args.dataset)
    else:
        csv_path = save_trials(study, args.out, args.dataset)
        best_json = save_best(study, args.out, args.dataset)

    print(f"Trials saved to {csv_path}")
    print(f"Best trial saved to {best_json}")

if __name__ == "__main__":
    main()

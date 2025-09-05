import argparse, os, math, json, random
from dataclasses import dataclass
from typing import Optional, List
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
# Model
# ---------------------------
class OneHiddenMLP(nn.Module):
    def __init__(self, n_in:int, n_hid:int, n_out:int, activation:str,
                 dist:str, scale_heur:str, scale_mult: Optional[float]):
        super().__init__()
        self.fc1 = nn.Linear(n_in, n_hid, bias=True)
        self.bn1 = nn.BatchNorm1d(n_hid)
        self.fc2 = nn.Linear(n_hid, n_out, bias=True)
        if activation == 'tanh':
            self.act = nn.Tanh()
        elif activation == 'logistic':
            self.act = nn.Sigmoid()
        else:
            raise ValueError(activation)
        # zero-init final layer (zero_softmax analogue)
        nn.init.zeros_(self.fc2.weight); nn.init.zeros_(self.fc2.bias)
        # initialize first layer per requested heuristic/distribution
        if scale_heur == 'Glorot':
            nn.init.xavier_uniform_(self.fc1.weight) if dist == 'uniform' else nn.init.xavier_normal_(self.fc1.weight)
        elif scale_heur == 'old':
            fan_in = n_in
            s = (scale_mult if scale_mult is not None else 1.0) / math.sqrt(max(1.0, fan_in))
            if dist == 'uniform':
                nn.init.uniform_(self.fc1.weight, -s, s)
            else:
                nn.init.normal_(self.fc1.weight, 0.0, s)
        else:
            raise ValueError(scale_heur)
        nn.init.zeros_(self.fc1.bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.bn1(self.fc1(x)))
        return self.fc2(h)
        
class OneHiddenMLPcl(nn.Module):
    def __init__(self, n_in:int, n_hid:int, n_out:int, activation:str,
                 dist:str, scale_heur:str, scale_mult: Optional[float]):
        super().__init__()
        self.fc1 = nn.Linear(n_in, n_hid, bias=True)
        self.cl = nn.Linear(n_hid, n_hid, bias=False)
        self.fc2 = nn.Linear(n_hid, n_out, bias=True)
        if activation == 'tanh':
            self.act = nn.Tanh()
        elif activation == 'logistic':
            self.act = nn.Sigmoid()
        else:
            raise ValueError(activation)
        # zero-init final layer (zero_softmax analogue)
        nn.init.zeros_(self.fc2.weight); nn.init.zeros_(self.fc2.bias)
        # initialize first layer per requested heuristic/distribution
        if scale_heur == 'Glorot':
            nn.init.xavier_uniform_(self.fc1.weight) if dist == 'uniform' else nn.init.xavier_normal_(self.fc1.weight)
        elif scale_heur == 'old':
            fan_in = n_in
            s = (scale_mult if scale_mult is not None else 1.0) / math.sqrt(max(1.0, fan_in))
            if dist == 'uniform':
                nn.init.uniform_(self.fc1.weight, -s, s)
            else:
                nn.init.normal_(self.fc1.weight, 0.0, s)
        else:
            raise ValueError(scale_heur)
        nn.init.zeros_(self.fc1.bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.fc1(x))
        c = self.cl(h)
        return self.fc2(c)
        
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
    global_step = step_base # to support fold-wise offset
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
def energy_distance_to_gaussian(C: torch.Tensor, mu: float = 0.0, sigma: float = 1.0, num_samples: int = None) -> torch.Tensor:
    """
    Compute the energy distance between the off-diagonal elements of a coupling matrix C
    and a Gaussian distribution with mean mu and standard deviation sigma.
    
    Args:
        C (torch.Tensor): Square coupling matrix.
        mu (float): Mean of the target Gaussian distribution (default: 0.0).
        sigma (float): Standard deviation of the target Gaussian distribution (default: 1.0).
        num_samples (int, optional): Number of samples to draw from the Gaussian.
            If None, set equal to the number of off-diagonal elements.
    
    Returns:
        torch.Tensor: Energy distance (scalar).
    """
    # Ensure C is a square matrix
    if C.shape[0] != C.shape[1]:
        raise ValueError("Input matrix C must be square")
    
    # Extract off-diagonal elements
    n = C.shape[0]
    off_diag_mask = ~torch.eye(n, dtype=torch.bool, device=C.device)
    X = C[off_diag_mask].flatten()
    n_samples = X.shape[0]
    
    # If num_samples not specified, match the number of off-diagonal elements
    if num_samples is None:
        num_samples = n_samples
    
    # Sample from Gaussian distribution
    Y = torch.normal(mean=mu, std=sigma, size=(num_samples,), device=C.device)
    
    # Compute pairwise distances
    # Term 1: 2 * E[|X - Y|]
    XY_diff = torch.abs(X.unsqueeze(1) - Y.unsqueeze(0))  # Shape: (n_samples, num_samples)
    term1 = 2 * XY_diff.mean()
    
    # Term 2: E[|X - X'|]
    XX_diff = torch.abs(X.unsqueeze(1) - X.unsqueeze(0))  # Shape: (n_samples, n_samples)
    term2 = XX_diff.mean()
    
    # Term 3: E[|Y - Y'|]
    YY_diff = torch.abs(Y.unsqueeze(1) - Y.unsqueeze(0))  # Shape: (num_samples, num_samples)
    term3 = YY_diff.mean()
    
    # Energy distance
    energy_dist = term1 - term2 - term3
    
    return energy_dist.item()
    
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
    n_hid = trial.suggest_int('nhid1', 16, 1024, step=16)
    squash = trial.suggest_categorical('squash', ['tanh','logistic'])
    dist1 = trial.suggest_categorical('dist1', ['uniform','normal'])
    scale_heur1 = trial.suggest_categorical('scale_heur1', ['old','Glorot'])
    scale_mult1 = None
    if scale_heur1 == 'old':
        scale_mult1 = trial.suggest_float('scale_mult1', 0.2, 2.0)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    accs = []
    fit_criterias = []
    for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(Xp, y)):
        Xtr, Xva = Xp[tr_idx], Xp[va_idx]
        ytr, yva = y[tr_idx], y[va_idx]
        if use_cl:
            model = OneHiddenMLPcl(n_in, n_hid, n_classes, squash, dist1, scale_heur1, scale_mult1)
        else:
            model = OneHiddenMLP(n_in, n_hid, n_classes, squash, dist1, scale_heur1, scale_mult1)
        acc = train_eval_once(
            model, Xtr, ytr, Xva, yva, device, trial,
            max_epochs=max_epochs,
            step_base=fold_idx * max_epochs
        )
        accs.append(acc)
        if use_cl:
            # record max off-diagonal (absolute optional: uncomment next line to use abs)
            cl_w = model.cl.weight
            fit_criteria = energy_distance_to_gaussian(cl_w)
            fit_criterias.append(float(fit_criteria))
    if use_cl and len(fit_criterias) > 0:
        trial.set_user_attr('avg_fit_criteria', float(np.mean(fit_criterias)))
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
        # Ensure 2D (N, D)
        if X.ndim > 2:
            X = X.reshape(X.shape[0], -1)
        return X, y

    # Torchvision datasets
    import torchvision
    import torch

    if name == "cifar10":
        trainset = torchvision.datasets.CIFAR10(root=root, train=True, download=True)
        testset  = torchvision.datasets.CIFAR10(root=root, train=False, download=True)
        X = np.concatenate([trainset.data, testset.data], axis=0)            # (N, 32, 32, 3), uint8
        y = np.array(trainset.targets + testset.targets, dtype=np.int64)
        X = X.reshape(X.shape[0], -1).astype(np.float32)                      # (N, 3072)
        return X, y

    if name == "cifar100":
        trainset = torchvision.datasets.CIFAR100(root=root, train=True, download=True)
        testset  = torchvision.datasets.CIFAR100(root=root, train=False, download=True)
        X = np.concatenate([trainset.data, testset.data], axis=0)            # (N, 32, 32, 3), uint8
        y = np.array(trainset.targets + testset.targets, dtype=np.int64)
        X = X.reshape(X.shape[0], -1).astype(np.float32)                      # (N, 3072)
        return X, y

    if name == "fashion_mnist":
        trainset = torchvision.datasets.FashionMNIST(root=root, train=True, download=True)
        testset  = torchvision.datasets.FashionMNIST(root=root, train=False, download=True)
        # .data is a torch.Tensor (N, 28, 28), uint8
        X = np.concatenate([trainset.data.numpy(), testset.data.numpy()], axis=0)
        y = np.concatenate([trainset.targets.numpy(), testset.targets.numpy()], axis=0).astype(np.int64)
        X = X.reshape(X.shape[0], -1).astype(np.float32)                      # (N, 784)
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
    df = df.assign(avg_fit_criteria=[t.user_attrs.get('avg_fit_criteria', None) for t in study.trials])
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
            "avg_fit_criteria": best_trial.user_attrs.get('avg_fit_criteria', None)
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
    ap.add_argument("--cl", action='store_true', help="Use CL model (default: False)")
    ap.add_argument("--epochs", type=int, default=200)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    set_seed(args.seed)
    X, y = load_dataset(args.dataset)
    total_trials = args.trials
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
    # Save + data exports (no plots)
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
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
from optuna.samplers import GridSampler, RandomSampler, TPESampler

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
                    max_epochs: int = 50,
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
    global_step = step_base  # to support fold-wise offset

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

def _phi(x: torch.Tensor) -> torch.Tensor:
    # standard normal pdf
    return torch.exp(-0.5 * x**2) / math.sqrt(2*math.pi)

def _Phi(x: torch.Tensor) -> torch.Tensor:
    # standard normal cdf via erf
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

    Args
    ----
    C : (d,d) tensor, coupling matrix
    robust : if True, center by median and scale by 1.4826*MAD; else mean/std
    max_pairs_exact : compute the U-statistic exactly if n*(n-1) <= this budget
    subsample_pairs : if exact is too big, randomly sample this many ordered pairs
    seed : RNG for subsampling

    Returns
    -------
    float : energy distance score (>= 0 in population; small-sample estimates can be slightly <0)
    """
    C = C.detach().cpu().float()
    if C.ndim != 2 or C.shape[0] != C.shape[1]:
        raise ValueError("C must be square (d x d).")
    d = C.shape[0]
    if d < 2:
        return float('nan')

    # 1) collect off-diagonal entries (use both upper & lower; DO NOT abs())
    mask = ~torch.eye(d, dtype=torch.bool)
    x = C[mask]                  # shape [n], where n = d*(d-1)
    n = x.numel()
    if n < 8:
        return float('nan')

    # 2) robust (or classical) standardization
    if robust:
        med = x.median()
        mad = (x - med).abs().median().clamp(min=1e-12)
        scale = 1.4826 * mad
        z = (x - med) / scale
    else:
        mu = x.mean()
        sd = x.std(unbiased=False).clamp(min=1e-12)
        z = (x - mu) / sd

    # 3) Components of energy distance to N(0,1)
    # 3a) A = 2 * E|X - G| = (2/n) * sum_i E|z_i - G|
    #     with closed form: E|G - a| = 2*phi(a) + a*(2*Phi(a)-1)
    EzG = 2.0 * _phi(z) + z * (2.0 * _Phi(z) - 1.0)
    A = 2.0 * EzG.mean()

    # 3b) B = E|X - X'| estimated as U-statistic
    # exact if feasible, else subsample ordered pairs (i != j)
    # Use float64 for stability if very large
    z = z.to(torch.float64)

    pair_budget = n * (n - 1)
    if pair_budget <= max_pairs_exact:
        # exact: compute mean |z_i - z_j| over i != j using trick with all-pairs L1
        # memory-friendly chunking if needed
        B_sum = 0.0
        count = 0
        # choose chunk size by memory; 8k is usually safe
        chunk = min(n, 8192)
        for start in range(0, n, chunk):
            z1 = z[start:start+chunk].unsqueeze(1)        # [c,1]
            # broadcast against all z
            diff = (z1 - z.unsqueeze(0)).abs()            # [c,n]
            # remove diagonal terms when start block overlaps
            if start == 0:
                # we'll remove diagonals after full loop to keep code simple
                pass
            B_sum += diff.sum().item()
            count += diff.numel()
        # subtract diagonal contributions (|z_i - z_i| = 0 anyway) and exclude them in count
        count -= n  # exclude i==j
        B = B_sum / count
    else:
        # subsample ordered pairs
        g = torch.Generator(device='cpu').manual_seed(seed)
        # number of pairs to draw cannot exceed n*(n-1)
        m = min(subsample_pairs, pair_budget)
        i = torch.randint(0, n, (m,), generator=g)
        j = torch.randint(0, n-1, (m,), generator=g)
        # ensure i != j by mapping j>=i -> j+1
        j = j + (j >= i)
        B = (z[i] - z[j]).abs().mean().item()

    # 3c) C = E|G - G'| = 2 / sqrt(pi)
    Cconst = 2.0 / math.sqrt(math.pi)

    ED = float((A - B - Cconst).item() if isinstance(A, torch.Tensor) else (A - B - Cconst))
    return ED

# ---------------------------
# Objective for Optuna
# ---------------------------
def objective(trial, X: np.ndarray, y: np.ndarray, n_splits:int=3, max_epochs:int=50, use_cl:bool=False, device:str='cpu'):
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
            fit_criteria = energy_distance_to_gaussian_from_C(cl_w)
            fit_criterias.append(float(fit_criteria))

    if use_cl and len(fit_criterias) > 0:
        trial.set_user_attr('avg_fit_criteria', float(np.mean(fit_criterias)))
    return float(np.mean(accs))


# ---------------------------
# Dataset loading (from dataset.py) with safe fallback
# ---------------------------
def load_dataset(name:str):
    from datasets.larochelle_etal_2007.dataset import (
        MNIST_Basic, MNIST_BackgroundImages, MNIST_BackgroundRandom,
        MNIST_Rotated, MNIST_RotatedBackgroundImages,
        Rectangles, RectanglesImages, Convex
    )
    dmap = {
        "mnist_basic": MNIST_Basic,
        "mnist_background_images": MNIST_BackgroundImages,
        "mnist_background_random": MNIST_BackgroundRandom,
        "mnist_rotated": MNIST_Rotated,
        "mnist_rotated_background_images": MNIST_RotatedBackgroundImages,
        "rectangles": Rectangles,
        "rectangles_images": RectanglesImages,
        "convex": Convex,
    }
    D = dmap[name]()
    X = D.latent_structure_task()
    y = D._labels.copy()
    return X, y


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
# Grid space (NEW) — tweak freely
# ---------------------------
def make_search_space():
    # Keep this modest to avoid huge Cartesian products.
    return {
        # preprocessing
        "preproc": ["raw", "colnorm", "pca"],
        "colnorm_thresh": [1e-9, 1e-6, 1e-3],
        "pca_energy": [0.70, 0.85, 0.95, 0.99],

        # seeds / model structure
        "iseed": [5, 6, 7, 8],
        "nhid1": [64, 128, 256, 512],
        "squash": ["tanh", "logistic"],
        "dist1": ["uniform", "normal"],
        "scale_heur1": ["old", "Glorot"],
        "scale_mult1": [0.5, 1.0, 1.5, 2.0],   # only used when scale_heur1 == 'old'

        # SGD + anneal + batches + L2
        "lr": [1e-3, 3e-3, 1e-2, 3e-2],
        "lr_anneal_start": [100, 1000, 5000],
        "batch_size": [20, 100],
        "l2_penalty_choice": ["zero", "nz"],
        "l2_penalty_nz": [1e-6, 1e-4, 1e-2],
    }


# ---------------------------
# Main
# ---------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, default="rectangles_images",
                    choices=["mnist_basic","mnist_background_images","mnist_background_random",
                             "mnist_rotated","mnist_rotated_background_images",
                             "rectangles","rectangles_images","convex"])
    ap.add_argument("--optimization", type=str, default="grid",
                    choices=["grid", "random", "parzen"],)
    ap.add_argument("--trials", type=int, default=80)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--out", type=str, default="runs/experiment")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--cv", type=int, default=3, help="Stratified K-folds")
    ap.add_argument("--cl", action='store_true', help="Use CL model (default: False)")
    ap.add_argument("--epochs", type=int, default=50)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    set_seed(args.seed)

    X, y = load_dataset(args.dataset)

    # default number of trials (overridden for grid to avoid exceeding cartesian space)
    total_trials = args.trials

    if args.optimization == "grid":
        search_space = make_search_space()
        # compute the cartesian size to cap n_trials
        total_trials = 1
        for _, v in search_space.items():
            total_trials *= len(v)
        sampler = GridSampler(search_space)
    elif args.optimization == "random":
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
        n_trials=total_trials if args.optimization == "grid" else args.trials
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

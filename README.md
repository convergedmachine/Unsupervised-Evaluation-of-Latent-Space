````markdown
# Representation Learning Experiments

This repository contains training and evaluation pipelines for **Autoencoders (AE)** and **Multi-Layer Perceptrons (MLPs)** with up to three hidden layers, alongside tools for visualization and analysis.  
The focus is on **representation quality diagnostics** using *energy distance (ED)* metrics derived from coupling matrices, as well as Optuna-driven hyperparameter optimization.

---

## 📂 Repository Structure

- **Training Scripts**
  - `ae_train.py` — 7-layer Autoencoder with coupling layers. Trains to minimize reconstruction error and logs ED metrics:contentReference[oaicite:6]{index=6}.
  - `one_hidden_mlp_train.py` — 1-hidden-layer MLP training with hyperparameter search:contentReference[oaicite:7]{index=7}.
  - `two_hidden_mlp_train.py` — 2-hidden-layer MLP training with coupling diagnostics:contentReference[oaicite:8]{index=8}.
  - `three_hidden_mlp_train.py` — 3-hidden-layer MLP training (with and without coupling layers):contentReference[oaicite:9]{index=9}.

- **Visualization Tools**
  - `visualize_accuracy_vs_param.py` — Plot accuracy vs. hyperparameters across multiple datasets:contentReference[oaicite:10]{index=10}.
  - `visualize_two_hidden_mlp.py` — Grid visualizations of validation accuracy and energy distance trajectories for 2-hidden MLPs:contentReference[oaicite:11]{index=11}.
  - `vizs_helpers.py` — Helper functions for plotting efficiency curves, ARD panels, and parameter sweeps:contentReference[oaicite:12]{index=12}.

- **Analysis Utilities**
  - `pivot_table.py` — Collects and organizes Optuna trial results into dictionaries for comparison of search strategies (Parzen vs. Random), and generates accuracy–parameter plots:contentReference[oaicite:13]{index=13}.

---

## 🔑 Key Features

- **Autoencoder with Coupling Layers**  
  - Symmetric 7-layer AE (`ae_train.py`)  
  - Tracks *reconstruction error* and *energy distance to Gaussian noise* from each coupling layer.  
  - Diagnostics for disentanglement via off-diagonal coupling statistics:contentReference[oaicite:14]{index=14}.

- **Multi-Layer Perceptrons (MLPs)**  
  - Configurations: 1, 2, and 3 hidden layers.  
  - Supports coupling layers (`cl`) for redundancy diagnostics.  
  - Training with SGD or AdamW and learning rate annealing:contentReference[oaicite:15]{index=15}:contentReference[oaicite:16]{index=16}.

- **Hyperparameter Optimization**  
  - Implemented with **Optuna** (Random and TPE samplers).  
  - Search spaces follow NIPS’11 hyperparameter settings (hidden units, activation, initialization, learning rate, batch size, weight decay, etc.):contentReference[oaicite:17]{index=17}.

- **Visualization & Diagnostics**  
  - Accuracy vs. parameter scatter plots.  
  - Efficiency curves (best-of-K trials).  
  - ARD relevance boxplots for hyperparameter importance.  
  - Time-series grids of validation accuracy and ED for each hidden-size configuration:contentReference[oaicite:18]{index=18}:contentReference[oaicite:19]{index=19}.

---

## 🚀 Getting Started

### 1. Install Dependencies
```bash
pip install torch torchvision optuna scikit-learn matplotlib pandas
````

### 2. Run Training

Example: Train a 2-hidden MLP on CIFAR-100:

```bash
python two_hidden_mlp_train.py --dataset cifar100 --epochs 200 --device cuda
```

Train Autoencoder with 7 hidden layers:

```bash
python ae_train.py --dataset mnist_basic --epochs 200 --device cuda
```

### 3. Run Visualizations

Accuracy vs parameter plots:

```bash
python visualize_accuracy_vs_param.py
```

Grid plots for 2-hidden MLP trajectories:

```bash
python visualize_two_hidden_mlp.py
```

Pivot-table analysis:

```bash
python pivot_table.py
```

---

## 📊 Datasets

Supported datasets include:

* **Larochelle et al. (2007) suite**: MNIST variants (basic, rotated, background), Rectangles, Convex.
* **Torchvision datasets**: CIFAR-10, CIFAR-100, Fashion-MNIST.

---

## 📈 Example Outputs

* Validation accuracy vs ED trajectories across hidden sizes.
* Efficiency curves comparing Parzen vs Random search.
* Hyperparameter relevance boxplots (ARD).
* Scatter plots of accuracy vs parameter metrics.

---

## 🔬 Research Context

This codebase underpins experiments on **unsupervised evaluation of representation quality**, particularly:

* Coupling-layer based redundancy diagnostics.
* Energy-distance to Gaussian as an unsupervised measure of representation disentanglement.
* Hyperparameter search efficiency analysis.

---

## 📄 License

MIT License.
See `LICENSE` for details.

```

---

Would you like me to also add **example figures** (like mock plots/screenshots placeholders in the README) so it’s ready for GitHub presentation, or should I keep it purely text-based?
```


import torch
from torch.utils.data import Dataset, DataLoader
from datasets.larochelle_etal_2007.dataset import (
    MNIST_Basic,
    MNIST_BackgroundImages,
    MNIST_BackgroundRandom,
    MNIST_Rotated,
    MNIST_RotatedBackgroundImages,
    Convex,
    Rectangles,
    RectanglesImages
)

class TorchDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i], self.y[i]

class VectorXV(object):
    def __init__(self, dataset):
        self.dataset = dataset

    def to_torch_datasets(self, download=True):
        ds = self.dataset
        ds.fetch(download_if_missing=download)
        ds.build_meta()
        n_train = ds.descr['n_train']
        n_valid = ds.descr['n_valid']
        n_test = ds.descr['n_test']

        start, end = 0, n_train
        x_train = ds._inputs[start:end].reshape((end-start), -1)
        y_train = ds._labels[start:end]

        start, end = n_train, n_train + n_valid
        x_valid = ds._inputs[start:end].reshape((end-start), -1)
        y_valid = ds._labels[start:end]

        start, end = n_train + n_valid, n_train + n_valid + n_test
        x_test = ds._inputs[start:end].reshape((end-start), -1)
        y_test = ds._labels[start:end]

        return (
            TorchDataset(x_train, y_train),
            TorchDataset(x_valid, y_valid),
            TorchDataset(x_test, y_test)
        )

# Example usage:
if __name__ == "__main__":
    from torch.utils.data import DataLoader
    ds = Convex()
    vx = VectorXV(ds)
    train_set, val_set, test_set = vx.to_torch_datasets()

    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=64)
    test_loader = DataLoader(test_set, batch_size=64)

    for xb, yb in train_loader:
        print(xb.shape, yb.shape)
        break

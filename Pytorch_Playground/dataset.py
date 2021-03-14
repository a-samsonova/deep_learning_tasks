"""
Генерация синтетических данных
"""
from sklearn import datasets
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

class Circles(Dataset):
    def __init__(self, n_samples, shuffle, noise=0.1, random_state=0, factor=0.8):
        self.X, self.y = datasets.make_circles(n_samples=n_samples, shuffle=shuffle,
                                               noise=noise, random_state=random_state, factor=factor)
        sc = StandardScaler()
        self.X = sc.fit_transform(self.X)
        self.X, self.y = self.X.astype(np.float32), self.y.astype(np.int)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]), torch.from_numpy(np.array(self.y[idx]))

    def plot_data(self):
        plt.figure(figsize=(8, 8))
        plt.scatter(self.X[:, 0], self.X[:, 1], c=self.y)
        plt.show()

if __name__ == "__main__":
    circles = Circles(n_samples=5000, shuffle=True, factor=0.5)
    circles.plot_data()
    train_dataset = Circles(n_samples=50, shuffle=True, random_state=1)
    test_dataset = Circles(n_samples=10, shuffle=True, random_state=2)

    train_dataloader = DataLoader(train_dataset, batch_size=5, shuffle=True)

    for i, (x, y) in enumerate(train_dataloader):
        print("Batch: ", i)
        print(x, y)

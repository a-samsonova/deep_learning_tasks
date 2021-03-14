from torch.nn import CrossEntropyLoss, L1Loss, MSELoss, MultiLabelMarginLoss
from torch.optim import Adam, SGD, RMSprop, LBFGS
import torch
from torch.utils.data import DataLoader
import logging
import os
from dataset import Circles
from nn import SimpleMLP
from datetime import datetime
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt

class Trainer():
    def __init__(self, model, lr, optimizer=None, criterion=None):
        self.model = model

        self.available_losses = ['CrossEntropyLoss', 'L1Loss', 'MSELoss', 'MultiLabelMarginLoss']
        self.criterion = self.get_criterion(criterion=criterion)

        self.available_optimizers = ['Adam', 'SGD', 'RMSprop', 'LBFGS']
        self.optimizer = self.get_optimizer(optimizer=optimizer, lr=lr)

        cuda = torch.cuda.is_available()
        self.device = torch.device("cuda:0" if cuda else "cpu")

    def get_criterion(self, criterion):
        if criterion == "CrossEntropyLoss":
            return CrossEntropyLoss()
        if criterion == "L1Loss":
            return L1Loss()
        if criterion == "MSELoss":
            return MSELoss()
        if criterion == "MultiLabelMarginLoss":
            return MultiLabelMarginLoss()

    def get_optimizer(self, optimizer, lr):
        if optimizer == "Adam":
            return Adam(self.model.parameters(), lr)
        if optimizer == "SGD":
            return SGD(self.model.parameters(), lr)
        if optimizer == "RMSprop":
            return RMSprop(self.model.parameters(), lr)
        if optimizer == "LBFGS":
            return LBFGS(self.model.parameters(), lr)

    def log(self, path):
        logging.basicConfig(format='%(asctime)s %(message)s',
                            filename=os.path.join(path, str(datetime.now())+"std.log"),
                            filemode='w')
        logger = logging.getLogger()
        return logger

    def fit(self, train_dataloader, n_epochs):
        writer = SummaryWriter()

        loop = tqdm(range(n_epochs))
        for epoch in loop:
            epoch_loss = 0
            for i, (x_batch, y_batch) in enumerate(train_dataloader):
                self.optimizer.zero_grad()
                output = self.model(x_batch)
                loss = self.criterion(output, y_batch.type(torch.LongTensor))
                writer.add_scalar("Loss/train", loss, epoch)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()

            loop.set_description(f"Epoch [{epoch}/{n_epochs}]")
            loop.set_postfix(loss=epoch_loss/len(train_dataloader))

        writer.flush()
        writer.close()

    def predict(self, test_dataloader):
        writer = SummaryWriter()
        all_outputs = torch.tensor([], dtype=torch.long)
        self.model.eval()
        total_acc = 0
        cnt = 0
        with torch.no_grad():
            for i, (x_batch, y_batch) in enumerate(test_dataloader):
                output_batch = self.model(x_batch)
                _, predicted = torch.max(output_batch.data, 1)
                all_outputs = torch.cat((all_outputs, predicted), 0)
                acc = accuracy_score(predicted, y_batch)
                total_acc += acc
                writer.add_scalar("Accuracy/Batch", acc, i)
                cnt += 1
            print("Average Accuracy: ", total_acc/cnt)

        writer.flush()
        writer.close()
        return all_outputs

    def predict_proba(self, test_dataloader):
        all_outputs = torch.tensor([], dtype=torch.float32)
        self.model.eval()
        with torch.no_grad():
            for i, (x_batch, y_batch) in enumerate(test_dataloader):
                output_batch = self.model(x_batch)
                all_outputs = torch.cat((all_outputs, output_batch), 0)
        return all_outputs

    def predict_proba_tensor(self, T):
        self.model.eval()
        with torch.no_grad():
            output = self.model(T)
        return output

if __name__ == "__main__":
    layers_list_example = [(2, 5), (5, 3), (3, 6), (6, 2)]
    model = SimpleMLP(layers_list_example)

    trainer = Trainer(model, optimizer="Adam", criterion="CrossEntropyLoss", lr=0.01)
    print(trainer.device)

    train_dataset = Circles(n_samples=5000, shuffle=True, random_state=1, factor=0.5)
    test_dataset = Circles(n_samples=1000, shuffle=True, random_state=2, factor=0.5)

    train_dataloader = DataLoader(train_dataset, batch_size=50, shuffle=True)
    test_dataloader = DataLoader(train_dataset, batch_size=50, shuffle=False)

    trainer.fit(train_dataloader, n_epochs=100)
    res = trainer.predict(test_dataloader)

    # Tenserboard
    # https://tensorboard.dev/experiment/AdYd1TgeTlaLWXx6I8JUbA/#scalars

    # Scatter Plot{"
    h = 0.02
    X = test_dataset.X
    y = test_dataset.y

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 0].min() - 1, X[:, 0].max() + 1

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    grid_tensor = torch.FloatTensor(np.c_[xx.ravel(), yy.ravel()])

    Z = torch.autograd.Variable(grid_tensor)
    Z = Z.data.numpy()
    Z = np.argmax(Z, axis=1)
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(10, 8))

    plt.contourf(xx, yy, Z, cmap=plt.cm.rainbow, alpha=0.3)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.rainbow)

    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title("Scatter plot", fontsize=15)
    plt.xlabel("x", fontsize=13)
    plt.ylabel("y", fontsize=13)
    plt.show()
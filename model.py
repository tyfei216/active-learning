'''
Description: 

Author: Tianyi Fei
Date: 1969-12-31 19:00:00
LastEditors: Tianyi Fei
LastEditTime: 2022-04-26 11:11:07
'''
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
import pickle
from torch.utils.data import DataLoader
from activeselect import pick
from sklearn.metrics import mean_squared_error
from torch.utils.data import random_split
import numpy as np

EPOCH = 700
ITERS = 5000


def getdataset(x, y1, y2=None):
    if y2 is None:
        y = y1
    else:
        y = np.transpose(np.stack((y1, y2)))
    ds = RegressionDataset(x, y)
    d = DataLoader(ds, batch_size=20, shuffle=True)
    return d


class RegressionDataset(Dataset):
    def __init__(self, x, y):
        # print(x.shape)
        # print(y.shape)
        self.x = torch.tensor(x).float()
        self.y = torch.tensor(y).float()
        self.length = x.shape[0]

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return self.x[index], self.y[index]


def train(model, dataset):
    cri = nn.MSELoss()
    model.train()
    # opt = optim.SGD(model.parameters(), lr=0.001, momentum=0.99)
    opt = optim.Adam(model.parameters(), lr=0.001)
    iters = 0
    while True:
        for _, j in enumerate(dataset):
            iters += 1
            x, y = j
            pre = model(x)
            # print(x, pre)
            # exit()
            # print(pre.shape, y.shape)
            loss = cri(pre, y)
            # if len(x) < 2:
            #     print(x, y, pre)
            #     exit()

            opt.zero_grad()
            loss.backward()
            opt.step()
        if iters > ITERS:
            break
    # print(iters)
    return model


def test(model, dataset):
    pres = []
    ys = []
    with torch.no_grad():
        for _, j in enumerate(dataset):
            x, y = j
            pre = model(x)
            pre = pre.numpy()
            pres.append(pre)
            ys.append(y.numpy())
    # print(mean_squared_error(np.concatenate(pres), np.concatenate(ys)))
    # print(np.concatenate(pres).shape, np.concatenate(ys).shape)
    res = mean_squared_error(np.concatenate(pres), np.concatenate(ys))
    return res


class TwoLayer(nn.Module):
    def __init__(self, in_dim, out_dim) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_dim, in_dim * 2)
        self.fc2 = nn.Linear(in_dim * 2, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, out_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        # x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x.squeeze(1)


if __name__ == "__main__":

    with open("./cache/mydataset.pkl", "rb") as f:
        data = pickle.load(f)
    import pandas as pd
    df = pd.read_csv("./cache/Breast_TCGA_with_fs.csv",
                     header=None,
                     index_col=None)

    print(df.shape)
    df = df.drop(0)
    print(df.head())
    df = df[list(range(11))]
    df = df.dropna()
    for i in df[10]:
        print(i)
    data = df[10]
    # Ds = RegressionDataset(data["x"],
    #                        np.transpose(np.stack((data["y7"], data["y8"]))))

    Ds = RegressionDataset(data["x"], data["y6"])

    trainset, testset = random_split(Ds, [600, len(Ds) - 600])
    dl = DataLoader(trainset, batch_size=40, shuffle=True)

    model = TwoLayer(9, 1)
    model = train(model, dl)
    dl = DataLoader(testset, batch_size=40, shuffle=True)
    res = test(model, dl)
    print(res)
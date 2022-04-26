'''
Description: 

Author: Tianyi Fei
Date: 1969-12-31 19:00:00
LastEditors: Tianyi Fei
LastEditTime: 2022-04-25 22:59:04
'''
import activeselect
import pandas as pd
import dataset
import random
from tqdm import tqdm
from sklearn.model_selection import KFold
from sklearn.linear_model import Lasso
import numpy as np
from sklearn.metrics import mean_squared_error
import pickle
from sklearn.linear_model import LinearRegression
import argparse
import os
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine
from sklearn.metrics import pairwise_distances
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from scipy.stats import pearsonr
import model
import torch

NUM_SIMU = 1
NUM_SAM = 40
BATCH_SIZE = 10


def feature_selection(x, y):
    _, a = x.shape
    ret = np.zeros(a)
    # print(x.shape, y.shape)
    for i in range(a):
        ret[i], _ = pearsonr(x[:, i], y)
    ret = np.abs(ret)
    ret /= np.mean(ret)
    return np.diag(ret)


def simulatetwo(df, seed, method, feature_select=True):
    valres_active = np.zeros((NUM_SIMU, 3, NUM_SAM))
    testres_active = np.zeros((NUM_SIMU, 3, NUM_SAM))
    for i in range(NUM_SIMU):
        random.seed(i + seed)
        np.random.seed(seed=i + seed)
        # 5-fold cross validation
        kf = KFold(n_splits=3, shuffle=True)
        cnt = -1
        print(i)
        for trainidx, testidx in kf.split(df):
            cnt += 1
            trainds = df.iloc[trainidx]
            trainds = dataset.BaseDataset(trainds, ini=31)
            testds = df.iloc[testidx]
            for j in tqdm(range(NUM_SAM)):
                d = trainds.getKnown()
                y1 = d["y1"].to_numpy()
                y2 = d["y2"].to_numpy()
                x = d.drop(["y1", "y2"], axis=1).to_numpy()
                # if len(d) % 100 == 0:
                #     print(len(d))
                ds = model.getdataset(x, y1, y2)
                m = model.TwoLayer(9, 2)
                m = model.train(m, ds)
                # gm = Lasso().fit(x, y)
                test_x = testds.drop(["y1", "y2"], axis=1).to_numpy()
                test_y1 = testds["y1"].to_numpy()
                test_y2 = testds["y2"].to_numpy()
                ds = model.getdataset(test_x, test_y1, test_y2)

                testres_active[i, cnt, j] = model.test(m, ds)
                du = trainds.getUnknown()
                dux = du.drop(["y1", "y2"], axis=1).to_numpy()
                du_y1 = du["y1"].to_numpy()
                du_y2 = du["y2"].to_numpy()
                ds = model.getdataset(dux, du_y1, du_y2)

                valres_active[i, cnt, j] = model.test(m, ds)
                if method == 'random':
                    new_points = random.sample(range(len(du)), k=BATCH_SIZE)
                    for new_point in new_points:
                        trainds.addKnown(du.index[new_point])

                elif method == "active":
                    sim = cos[du.index, :]
                    sim = sim[:, d.index]
                    simSelf = cos[du.index, :]
                    simSelf = simSelf[:, du.index]

                    new_points = activeselect.pick(dux,
                                                   x,
                                                   y,
                                                   d["pre"],
                                                   1,
                                                   sim=sim,
                                                   selfSim=simSelf)
                    for new_point in new_points:
                        trainds.addKnown(du.index[new_point])

                elif method == "gaussian":
                    predict, var = gm.predict(dux, return_std=True)
                    # t = np.argsort(var)
                    # for new_point in t[-1:]:
                    #     trainds.addKnown(du.index[new_point])
                    t = np.argmax(var)
                    trainds.addKnown(du.index[t])

                elif method == "gaussianb":
                    kernel = RBF()
                    gm1 = GaussianProcessRegressor(kernel=kernel).fit(x, y1)
                    gm2 = GaussianProcessRegressor(kernel=kernel).fit(x, y2)
                    for _ in range(BATCH_SIZE):
                        predict1, var1 = gm1.predict(dux, return_std=True)
                        predict2, var2 = gm2.predict(dux, return_std=True)
                        var = var1 + var2
                        new_point = np.argmax(var)
                        trainds.addKnown(du.index[new_point])
                        x = np.concatenate((x, dux[new_point].reshape(1, -1)))
                        y1 = np.append(y1, predict1[new_point])
                        y2 = np.append(y2, predict2[new_point])
                        gm1 = GaussianProcessRegressor(kernel=kernel).fit(
                            x, y1)
                        gm2 = GaussianProcessRegressor(kernel=kernel).fit(
                            x, y2)

                else:
                    raise NotImplementedError

    return testres_active, valres_active


def simulate(df, seed, method, feature_select=True, deep=False):
    valres_active = np.zeros((NUM_SIMU, 3, NUM_SAM))
    testres_active = np.zeros((NUM_SIMU, 3, NUM_SAM))
    # x = df.drop("y", axis=1).to_numpy()
    # norms = np.linalg.norm(x, axis=1)
    # cos = x @ x.T
    # dis = norms.reshape(-1, 1) @ norms.reshape(1, -1)
    # cos = cos / dis
    # cos += 1
    # cos /= 2
    # print(cos.shape, type(cos))

    for i in range(NUM_SIMU):
        random.seed(i + seed)
        np.random.seed(seed=i + seed)
        # 5-fold cross validation
        kf = KFold(n_splits=3, shuffle=True)
        cnt = -1
        print(i)
        for trainidx, testidx in kf.split(df):
            cnt += 1
            trainds = df.iloc[trainidx]
            trainds = dataset.BaseDataset(trainds, ini=31)
            testds = df.iloc[testidx]
            for j in tqdm(range(NUM_SAM)):
                d = trainds.getKnown()
                y = d["y"].to_numpy()
                x = d.drop("y", axis=1).to_numpy()
                ds = model.getdataset(x, y)
                if feature_select:
                    M = feature_selection(x, y)
                    x = x @ M
                # if len(d) % 100 == 0:
                #     print(len(d))
                if deep:
                    ds = model.getdataset(x, y)
                    m = model.TwoLayer(9, 1)

                    m = model.train(m, ds)

                else:
                    kernel = RBF(length_scale=2,
                                 length_scale_bounds=(1e-8, 1e8))
                    # print(x, y)
                    gm = GaussianProcessRegressor(kernel=kernel).fit(x, y)
                # gm = Lasso().fit(x, y)
                # d["pre"] = gm.predict(x)
                test_x = testds.drop("y", axis=1).to_numpy()
                if feature_select:
                    test_x = test_x @ M
                if deep:

                    test_y = testds["y"].to_numpy()
                    # print(np.max(test_y), np.min(test_y))
                    # print(np.max(test_x), np.min(test_x))
                    # for t in test_x:
                    #     print(t)
                    ds = model.getdataset(test_x, test_y)
                if deep:
                    # pass
                    testres_active[i, cnt, j] = model.test(m, ds)
                else:
                    predict_y = gm.predict(test_x)
                    testres_active[i, cnt, j] = mean_squared_error(
                        testds["y"], predict_y)

                du = trainds.getUnknown()
                dux = du.drop("y", axis=1).to_numpy()
                if feature_select:
                    dux = dux @ M
                if deep:
                    du_y = du["y"].to_numpy()
                    ds = model.getdataset(dux, du_y)
                if deep:
                    valres_active[i, cnt, j] = model.test(m, ds)
                else:
                    predict = gm.predict(dux)
                    valres_active[i, cnt,
                                  j] = mean_squared_error(du["y"], predict)
                if method == 'random':
                    new_points = random.sample(range(len(du)), k=BATCH_SIZE)
                    for new_point in new_points:
                        trainds.addKnown(du.index[new_point])

                elif method == "active":
                    sim = cos[du.index, :]
                    sim = sim[:, d.index]
                    simSelf = cos[du.index, :]
                    simSelf = simSelf[:, du.index]

                    new_points = activeselect.pick(dux,
                                                   x,
                                                   y,
                                                   d["pre"],
                                                   1,
                                                   sim=sim,
                                                   selfSim=simSelf)
                    for new_point in new_points:
                        trainds.addKnown(du.index[new_point])

                elif method == "gaussian":
                    predict, var = gm.predict(dux, return_std=True)
                    # t = np.argsort(var)
                    # for new_point in t[-1:]:
                    #     trainds.addKnown(du.index[new_point])
                    t = np.argmax(var)
                    trainds.addKnown(du.index[t])

                elif method == "gaussianb":
                    kernel = RBF(length_scale=2,
                                 length_scale_bounds=(1e-8, 1e8))
                    gm = GaussianProcessRegressor(kernel=kernel).fit(x, y)
                    for _ in range(BATCH_SIZE):
                        predict, var = gm.predict(dux, return_std=True)
                        new_point = np.argmax(var)
                        trainds.addKnown(du.index[new_point])
                        x = np.concatenate((x, dux[new_point].reshape(1, -1)))
                        y = np.append(y, predict[new_point])
                        gm = GaussianProcessRegressor(kernel=kernel).fit(x, y)

                else:
                    raise NotImplementedError

    return testres_active, valres_active


def Args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="./results/")
    parser.add_argument("--feature", type=str, default="achilles")
    parser.add_argument("--dataset", type=str, default="zhuang")
    parser.add_argument("--method", type=str, default="random")
    parser.add_argument("--baselearner", type=str, default="Linear")
    parser.add_argument("--savename", type=str, default=None)
    parser.add_argument("-f", action="store_true")
    parser.add_argument("-d", action="store_true")

    args = parser.parse_args()
    if args.savename is None:
        args.savename = args.dataset
    return args


if __name__ == "__main__":
    args = Args()
    if args.dataset == "zhuang":
        data = dataset.get_zhuang()
    elif args.dataset == "il2":
        data = dataset.get_il2()
    elif args.dataset == "ifng":
        data = dataset.get_ifng()
    elif args.dataset == "my1":
        data = dataset.get_my()["y1"]
    elif args.dataset == "my2":
        data = dataset.get_my()["y2"]
    elif args.dataset == "my3":
        data = dataset.get_my()["y3"]
    elif args.dataset == "my4":
        data = dataset.get_my()["y4"]
    elif args.dataset == "my6":
        data = dataset.get_my()["y6"]
    elif args.dataset == "two":
        data = (dataset.get_my()["y4"], dataset.get_my()["y5"])
    elif args.dataset == "two2":
        data = (dataset.get_my()["y7"], dataset.get_my()["y8"])
    elif args.dataset == "breast":
        df = pd.read_csv("./cache/Breast_TCGA_with_fs.csv",
                         header=None,
                         index_col=None)
        # print(df.shape)
        df = df.drop(0)
        df = df[list(range(11))]
        df = df.dropna()
        print(df.shape)
        data = np.array(df[10])
        # for i in data:
        #     print(i)

    else:
        print("dataset not available")
        exit()

    if args.feature == "achilles":
        feature = dataset.get_achilles()
    elif args.feature == "string":
        feature = dataset.get_string()
    elif args.feature == "my":
        feature = dataset.get_my()["x"]
    elif args.feature == "breast":
        df = pd.read_csv("./cache/Breast_TCGA_with_fs.csv",
                         header=None,
                         index_col=None)
        df = df.drop(0)
        df = df[list(range(11))]
        df = df.dropna()
        print(df.shape)
        feature = df[list(range(9))].values
    else:
        print("feature set not available")
        exit()

    # col = np.intersect1d(data.index, feature.columns)
    # x = feature[col].values
    # x = x.T
    # y = data[col]
    df = pd.DataFrame(feature)
    if "two" in args.dataset:
        df["y1"] = data[0]
        df["y2"] = data[1]
    else:
        df["y"] = data
    df.index = list(range(len(df)))
    print(df.shape)

    # norms = np.linalg.norm(x, axis=1)
    # cos = x @ x.T
    # dis = norms.reshape(-1, 1) @ norms.reshape(1, -1)
    # cos = cos / dis

    # exit()
    if "two" in args.dataset:
        res1, res2 = simulatetwo(df, 100, args.method, feature_select=args.f)
    else:
        res1, res2 = simulate(df,
                              100,
                              args.method,
                              feature_select=args.f,
                              deep=args.d)

    with open(os.path.join(args.checkpoint, args.savename + ".pkl"),
              "wb") as f:
        pickle.dump([res1, res2], f)

    plt.plot(np.mean(res1[:, :].reshape(3, -1), axis=0), label="val")
    plt.plot(np.mean(res2[:, :].reshape(3, -1), axis=0), label="test")
    plt.legend()
    plt.show()

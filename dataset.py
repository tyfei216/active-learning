'''
Description:

Author: Tianyi Fei
Date: 1969-12-31 19:00:00
LastEditors: Tianyi Fei
LastEditTime: 2022-03-11 16:18:14
'''
import numpy as np
import pandas as pd
from sklearn import preprocessing
from scipy.stats import zscore
import h5py
import random


def get_achilles():
    df = pd.read_csv("./cache/achilles.csv")
    genes_achilles = list(df.columns)
    strip_achilles = [x[:x.find(" ")] for x in genes_achilles]
    df.columns = strip_achilles
    df_ac = df.drop("DepMap_I", axis=1)
    df_ac.index = ["achilles" + str(i) for i in df_ac.index]
    df_ac = df_ac.dropna()
    df_ac = df_ac.apply(zscore)
    return df_ac


def get_ccle():
    df = pd.read_csv("./cache/ccle_protein_quantification.csv.gz")
    genes_ccle = df["Gene_Symbol"]
    genes_ccle = list(genes_ccle)
    df = df.drop("Gene_Symbol", axis=1)
    df.index = genes_ccle
    df_ccle = df.T
    df_ccle = df_ccle.iloc[5:]
    cols = [str(i) for i in df_ccle.columns]
    df_ccle.columns = cols
    df_ccle.index = ["ccle_" + str(i) for i in range(len(df_ccle))]
    # df_ccle = df_ccle.dropna()
    # for i in np.unique(df_ccle.columns):
    #     if (np.std(df_ccle[i]) < 1.0).any():
    #         df_ccle = df_ccle.drop(i, axis=1)
    # df_ccle = df_ccle.apply(zscore)
    return df_ccle


def get_string():
    f = open("./cache/string_human_genes.txt", "r")
    string_genes = []
    for i in f.readlines():
        string_genes.append(i.strip())
    f.close()
    values = np.genfromtxt("./cache/string_human_mashup_vectors_d800.txt",
                           delimiter="\t")
    df_string = pd.DataFrame(values.T, columns=string_genes)
    df_string.index = ["string_" + str(i) for i in df_string.index]
    df_string = df_string.dropna()
    df_string = df_string.apply(zscore)
    return df_string


def get_merged():
    df_ac = get_achilles()
    # df_ccle = get_ccle()
    df_string = get_string()
    return pd.concat([df_ac, df_string], join="inner", axis=0)


def get_ifng():
    f = h5py.File("./cache/schmidt_2021_ifng.h5", "r")
    s = pd.Series(f["covariates"][...], index=f["rownames"][...])
    return s


def get_zhuang():
    f = h5py.File("./cache/zhuang_2019.h5", "r")
    s = pd.Series(f["covariates"][...], index=f["rownames"][...])
    return s


def get_il2():
    f = f = h5py.File("./cache/schmidt_2021_il2.h5", "r")
    s = pd.Series(f["covariates"][...], index=f["rownames"][...])
    return s


class BaseDataset():
    def __init__(self, feature, target, ini=5) -> None:
        self.predicts = target.index
        df = pd.concat([feature, target], join="inner")
        df = df.T
        self.df = df
        self.length = len(self.df)
        self.known = ini
        self.mask = np.zeros(len(self.df), dtype=bool)
        self.knownlist = list(random.sample(range(self.length), k=ini))

        # assure the training set has different labels
        while len(np.unique(df.iloc[self.knownlist]["y"])) == 1:
            self.knownlist = list(random.sample(range(self.length), k=ini))
        self.mask[self.knownlist] = True
        self.index = list(df.index)

    def __len__(self):
        return len(self.df)

    # output the dataset for the given indexes
    def getList(self, l):
        df = self.df.iloc[l]
        df = df.copy()
        return df

    # return the known list
    def getKnown(self):
        df = self.df[self.mask]
        df = df.copy()
        return df

    # add a sample to the known list
    def addKnown(self, index):
        idx = self.index.index(index)
        if idx in self.knownlist:
            print("already known!")
            return
        self.knownlist.append(idx)
        self.mask[idx] = True

    # return all samples to the unknown list
    def getUnknown(self):
        df = self.df[~self.mask]
        df = df.copy()
        return df

    # return all samples
    def getWhole(self):
        df = self.df.copy()
        return df


class BaseDataset():
    def __init__(self, feature, target):
        self.predicts = target.index
        fulldata = pd.concat([feature, target], join="inner")
        self.fulldata = fulldata.T

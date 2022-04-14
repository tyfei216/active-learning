'''
Description: 

Author: Tianyi Fei
Date: 1969-12-31 19:00:00
LastEditors: Tianyi Fei
LastEditTime: 2022-04-12 22:08:41
'''
from sklearn.metrics import mean_squared_error
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

import dataset
import numpy as np

data = dataset.get_zhuang()
# feature = dataset.get_string()
feature = dataset.get_achilles()

col = np.intersect1d(data.index, feature.columns)

x = feature[col].values
x = x.T
y = data[col]
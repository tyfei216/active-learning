'''
Description: 

Author: Tianyi Fei
Date: 2022-04-11 11:22:07
LastEditors: Tianyi Fei
LastEditTime: 2022-04-11 22:04:51
'''
from scipy.spatial import distance_matrix
from scipy.spatial.distance import cosine
from sklearn.metrics import pairwise_distances
import numpy as np
import numpy.linalg


def pick(x,
         xknown,
         yknown,
         pre,
         num,
         sim=None,
         selfSim=None,
         alpha=1.0,
         beta=1.0):
    if sim is None:
        sim = pairwise_distances(x, xknown, metric=cosine)
    if selfSim is None:
        selfSim = pairwise_distances(x, x, metric=cosine)
    variance = np.abs(yknown - pre)
    variance /= np.mean(variance)
    score = np.dot(sim, variance)
    score -= np.sum(sim, axis=1) * alpha
    picked = []
    for i in range(num):
        t = np.argmax(score)
        picked.append(t)
        score[t] -= 1e100
        score -= selfSim[t] * beta
    return picked

'''
Description: 

Author: Tianyi Fei
Date: 1969-12-31 19:00:00
LastEditors: Tianyi Fei
LastEditTime: 2022-04-14 10:29:35
'''
import numpy as np
import pickle
import matplotlib.pyplot as plt

BLOCK_SIZE = 3
cov = np.zeros((BLOCK_SIZE * 3, BLOCK_SIZE * 3))
for i in range(BLOCK_SIZE):
    for j in range(BLOCK_SIZE):
        cov[i][j] = 0.2
        cov[i][i] = 1
        cov[i + BLOCK_SIZE][j + BLOCK_SIZE] = 0.2
        cov[i + BLOCK_SIZE][i + BLOCK_SIZE] = 1
        cov[i + BLOCK_SIZE * 2][j + BLOCK_SIZE * 2] = 0.2
        cov[i + BLOCK_SIZE * 2][i + BLOCK_SIZE * 2] = 1

x = np.random.multivariate_normal(np.zeros(BLOCK_SIZE * 3), cov, size=10000)
a1 = np.random.uniform(-5, 5, (BLOCK_SIZE * 3))
a1[BLOCK_SIZE * 2:] = 0
n1 = np.random.normal(scale=0.1, size=(10000))
y1 = 1 / (1 + np.exp(-x @ a1)) + n1
print(n1, y1)

a2 = np.random.uniform(-5, 5, (BLOCK_SIZE * 3))
a2[BLOCK_SIZE * 2:] = 0
a2[:BLOCK_SIZE] = a1[:BLOCK_SIZE]
n2 = np.random.normal(scale=0.15, size=(10000))
y2 = 1 / (1 + np.exp(-x @ a2)) + n2

a3 = np.random.uniform(-5, 5, (BLOCK_SIZE * 3))
a3[:BLOCK_SIZE] = 0
n3 = np.random.normal(scale=0.1, size=(10000))
y3 = 1 / (1 + np.exp(-x @ a3)) + n3

a4_1 = np.random.uniform(-5, 5, (BLOCK_SIZE * 3))
a4_2 = np.random.uniform(-5, 5, (BLOCK_SIZE * 3))
a4_3 = np.random.uniform(-5, 5, (BLOCK_SIZE * 3))
a4_4 = np.random.uniform(-5, 5, (BLOCK_SIZE * 3))
a4_5 = np.random.uniform(-5, 5, (BLOCK_SIZE * 3))
a4_6 = np.random.uniform(-5, 5, (BLOCK_SIZE * 3))
a4_1[:BLOCK_SIZE] = 0
a4_2[:BLOCK_SIZE] = 0
a4_3[:BLOCK_SIZE] = 0

n4 = np.random.normal(scale=0.1, size=(10000))
y4 = 1 / (1 + np.exp(-x @ a4_1)) + 1.5 / (1 + np.exp(-x @ a4_2)) + n4

n5 = np.random.normal(scale=0.1, size=(10000))
y5 = 1 / (1 + np.exp(-x @ a4_1)) + 1.5 / (1 + np.exp(-x @ a4_3)) + n5

n6 = np.random.normal(scale=0.1, size=(10000))
y6 = 1 / (1 + np.exp(-x @ a4_1)) + 1.5 / (1 + np.exp(-x @ a4_2)) + 1.5 / (
    1 + np.exp(-x @ a4_3)) + 1.5 / (1 + np.exp(-x @ a4_4)) + 1.5 / (
        1 + np.exp(-x @ a4_5)) + 1.5 / (1 + np.exp(-x @ a4_6)) + n5

n7 = np.random.normal(scale=0.1, size=(10000))
y7 = 1 / (1 + np.exp(-x @ a4_1)) + 1.5 / (1 + np.exp(-x @ a4_2)) + n4

n8 = np.random.normal(scale=0.1, size=(10000))
y8 = 1 / (1 + np.exp(-x @ a4_4)) + 1.5 / (1 + np.exp(-x @ a4_3)) + n5

with open("./cache/mydataset.pkl", "wb") as f:
    pickle.dump(
        {
            "x": x,
            "y1": y1,
            "y2": y2,
            "y3": y3,
            "a1": a1,
            "a2": a2,
            "a3": a3,
            "n1": n1,
            "n2": n2,
            "n3": n3,
            "n4": n4,
            "n5": n5,
            "y4": y4,
            "y5": y5,
            "y6": y6,
            "y7": y7,
            "y8": y8,
            "a4_1": a4_1,
            "a4_2": a4_2,
            "a4_3": a4_3,
            "a4_4": a4_4,
            "a4_5": a4_5,
            "a4_6": a4_6,
        }, f)

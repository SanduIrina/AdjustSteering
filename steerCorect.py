# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn; seaborn.set_style("whitegrid")

import matplotlib
matplotlib.use('GTKAgg')

import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
import pandas as pd
import pickle
import math

# def gradientDescent(X, y, theta, alpha, iterations):
#     m = len(y)
#     cost = np.zeros(iterations)
#     thetaMod = theta.copy()
#     thetaHist = np.zeros(iterations)
#
#     for i in range(iterations):
#         thetaMod = thetaMod - np.dot(X.T, (np.dot(X, thetaMod) - y)) * alpha / m
#         thetaHist[i] = thetaMod[1]
#         cost[i] = computeCost(X, y, thetaMod)
#
#     return thetaMod, thetaHist, cost
#
# data = np.loadtxt('data/ex1data2.txt', delimiter=',')
# x, y = data[:,:2], data[:,2]
# X = np.hstack((np.ones((X.shape[0],1)), X))
# theta = np.zeros(3)
#
# gradient, thetaHist, cost = gradientDescent(X, y, theta, 0.03, 500)
# paramsNorm = (np.array([1650, 3]) - mean) / sigma
# params = np.hstack((np.array([1]), paramsNorm))
# predict = np.dot(gradient, params)

# Training
# Load CSV and columns
# df = pd.read_csv("Steering.csv")
#
# Y = df['real'] # the real steering command
# X = df['given'] # the intended command
#
# # average difference:
#
# M = abs(X-Y)
#
# X = X.values.reshape(len(X), 1)
# Y = Y.values.reshape(len(Y), 1)
#
# # Split the data into training/testing sets
# X_train = X[:-17]
# X_test = X[-17:]
# # print(X_train)
#
#
# # Split the targets into training/testing sets
# Y_train = Y[:-17]
# Y_test = Y[-17:]
# # print(Y_train)
#
# # Plot outputs
# plt.scatter(X_test, Y_test, color='black')
# plt.title('Test Data')
# plt.xlabel('Size')
# plt.ylabel('Price')
# plt.xticks(())
# plt.yticks(())
#
# regr = linear_model.LinearRegression()
# regr.fit(X_train, Y_train)
#
filename = "saved_model.pkl"
# with open(filename, 'wb') as file:
#     pickle.dump(regr, file)
#
# # plt.plot(X_test, regr.predict(X_test), color='red', linewidth=3)
# # plt.show()
#
# # to make an individual prediction
# # print( str(round(regr.predict(5000))) )

#load model from file
with open(filename, 'rb') as file:
    pickle_model = pickle.load(file)

# to know what command z you should give if you want an actual output of c
z = np.array([[3.2]])
c = pickle_model.predict(z)

startDif = 0.01
target = 3.2
start = 4 * z
stop = 6 * z
evolution = []
i = 0
x = []
while(stop > start):
    mid = start + (stop - start) / 2
    print(mid)
    c = pickle_model.predict(mid)
    evolution.append(c[0][0])
    if(c >= target - 0.001 and c <= target + 0.001):
        print("found")
        break
    if (target < c):
        stop = mid
    else:
        start = mid
    x.append(i)
    i = i + 1
x.append(i)
print(c)
print(mid)
baseline = np.ones(len(evolution)) * target
print(evolution)
print(baseline)
print(x)
plt.plot(x, evolution, 'r')
plt.plot(x, baseline, 'b')
plt.show()
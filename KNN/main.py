import numpy as np
import pandas as pd
import matplotlib as npl
import matplotlib.pyplot as plt
from knn_C import *
from knn_R import *

# 设置画图时可以显示中文黑体
npl.rcParams["font.family"] = "SimHei"
npl.rcParams["axes.unicode_minus"] = False

# 读取数据，一个用于回归算法，一个用于分类算法
dataC = pd.read_csv("Iris.csv", header=0)
dataR = pd.read_csv("Iris.csv", header=0)

# 分类算法的数据处理
dataC["Species"] = dataC["Species"].map({"Iris-setosa": 0, "Iris-versicolor": 1, "Iris-virginica": 2})
dataC.drop("Id", axis=1, inplace=True)
dataC.drop_duplicates(inplace=True)
# 回归算法的数据处理
dataR.drop(["Id", "Species"], axis=1, inplace=True)
dataR.drop_duplicates(inplace=True)


def dataProcess_C(process_C=dataC):
    c0 = dataC[dataC["Species"] == 0]
    c1 = dataC[dataC["Species"] == 1]
    c2 = dataC[dataC["Species"] == 2]

    c0 = c0.sample(len(c0), random_state=0)
    c1 = c1.sample(len(c1), random_state=0)
    c2 = c2.sample(len(c2), random_state=0)

    train_X = pd.concat([c0.iloc[:40, :-1], c1.iloc[:40, :-1], c2.iloc[:40, :-1]], axis=0)
    train_y = pd.concat([c0.iloc[:40, -1], c1.iloc[:40, -1], c2.iloc[:40, -1]], axis=0)

    test_X = pd.concat([c0.iloc[40:, :-1], c1.iloc[40:, :-1], c2.iloc[40:, :-1]], axis=0)
    test_y = pd.concat([c0.iloc[40:, -1], c1.iloc[40:, -1], c2.iloc[40:, -1]], axis=0)

    return train_X, train_y, test_X, test_y, c0, c1, c2


trainC_X, trainC_y, testC_X, testC_y, c0, c1, c2 = dataProcess_C(dataC)


def dataProcess_R(process_R=dataR):
    r = dataR.sample(len(dataR), random_state=0)

    train_X = r.iloc[:120, :-1]
    train_y = r.iloc[:120, -1]

    test_X = r.iloc[120:, :-1]
    test_y = r.iloc[120:, -1]

    return train_X, train_y, test_X, test_y


trainR_X, trainR_y, testR_X, testR_y = dataProcess_R(dataR)

knnC = KNN_C(9)
knnC.fit(trainC_X, trainC_y)
result_C = knnC.predict(testC_X)
print(result_C)
print(testC_y.ravel())


knnR = KNN_R(3)
knnR.fit(trainR_X, trainR_y)
result_R = knnR.predict(testR_X)
print(result_R)
print(testR_y.ravel())


def drawPicture_R(r, test):
    plt.figure(figsize=(10, 10))
    plt.plot(result_R, "ro-")
    plt.plot(testR_y.ravel(), "go--")
    plt.title("KNN回归预测")
    plt.xlabel("节点序号")
    plt.ylabel("花瓣宽度")
    plt.show()


def drawPicture_C(Iris_setosa, Iris_versicolor, Iris_virginica, r, test):
    plt.scatter(x=c0["SepalLengthCm"][:40], y=c0["PetalLengthCm"][:40], color="r", label="Iris-versicolor")
    plt.scatter(x=c1["SepalLengthCm"][:40], y=c1["PetalLengthCm"][:40], color="g", label="Iris-setosa")
    plt.scatter(x=c2["SepalLengthCm"][:40], y=c2["PetalLengthCm"][:40], color="b", label="Iris-virginica")

    right = testC_X[result_C == testC_y]
    wrong = testC_X[result_C != testC_y]
    plt.scatter(x=right["SepalLengthCm"], y=right["PetalLengthCm"], color="c", marker=">", label="right")
    plt.scatter(x=wrong["SepalLengthCm"], y=wrong["PetalLengthCm"], color="m", marker="<", label="wrong")
    plt.xlabel("花蕊长度")
    plt.ylabel("花瓣长度")
    plt.title("KNN分类结果显示")
    # 打印自动选择最佳位置显示
    plt.legend(loc="best")
    plt.show()


drawPicture_R(result_R, testR_y)
drawPicture_C(c0, c1, c2, result_C, testC_y)
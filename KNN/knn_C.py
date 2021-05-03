import numpy as np


# KNN分类算法，检测所属类别
class KNN_C:
    def __init__(self, k):
        """
        :param k: 取前k个距离最短的值，也就是邻居的个数
        """
        self.k = k

    def fit(self, X, y):
        """
        训练方法
        :param X: 类数据类型，其形状为[样本数量， 样本特征数量]
        :param y: 类数组类型，其形状为[样本数量]
        :return:
        """
        self.X = np.asarray(X)
        self.y = np.asarray(y)

    def predict(self, X):
        """
        预测方法
        :param X: 类数组类型，其形状为[样本数量， 样本特征数量]
        :return: 返回预测好的结果，类数组类型
        """
        X = np.asarray(X)
        # 存放每一次迭代所求出的最短距离
        result = []
        # 以每一行作为一个数据来计算
        for x in X:
            # 计算测试集中的每一行和训练集的所有行的距离，但是其以列为单位计算,dis为列表类型
            # 将一行和所有行所计算的数都放到dis列表中
            dis = np.sqrt(np.sum((x - self.X) ** 2, axis=1))
            # 将dis中的数据从小到大排序，但是排序的是原始数据的索引
            # 例如，[5, 1, 2, 9, 6] 排序后index中为[1, 2, 0, 4, 3]
            #       0  1  2  3  4                1  2  5  6  9
            index = dis.argsort()
            # 截取index列表中前k个数，存放的为其索引
            index = index[:self.k]
            # 将索引传入到训练集y中，根据"Specise"计算类别出现的次数，传给count列表
            # 如k为5，有类别[0, 1, 2, 2, 2],则count中为[1,1,3],0，1，2分别出现的次数
            count = np.bincount(self.y[index])
            # 将count中最大的那个数的索引对应的行的类别，加入到result列表中，如上，则把类别为2的加入到result,
            result.append(count.argmax())
            # 一次迭代，直达将测试集全部行数计算完退出
        # 返回类数组类型result列表
        return np.asarray(result)

    def predict_w(self, X):
        """
        考虑权重的预测方法，只有一句语句不同，其余的释义与上一致
        :param X: 类数组类型，其形状为[样本数量， 样本特征数量]
        :return: 返回预测好的结果，类数组类型
        """
        X = np.asarray(X)

        result = []
        for x in X:
            dis = np.sqrt(np.sum((x - self.X) ** 2, axis=1))
            index = dis.argsort()
            index = index[:self.k]
            # 没有考虑权重时，是直接计算其类别出现的次数作为分类依据
            # 考虑权重是根据其距离的倒数来计算了，而不是直接算次数了，如[0，1，2，2，2]，这四个类别之前计算的距离分别为[2，4，3，5，2]
            # 其距离的倒数为[1/2， 1/4， 1/3， 1/5， 1/2]， 则count中为[1/2, 1/4, 1/3+1/5+1/2]
            count = np.bincount(self.y[index], weights=1 / dis[index])
            # 将count中最大的那个数的索引对应的行的类别，加入到result列表中，如上，则把类别为2的加入到result,
            result.append(count.argmax())
        return np.asarray(result)

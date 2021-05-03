import numpy as np


# KNN回归算法，预测数据
class KNN_R:
    def __init__(self, k):
        """
        :param k: 取前k个距离，即邻居的个数
        """
        self.k = k

    def fit(self, X, y):
        """
        训练函数，传入训练数据
        :param X: 类数据类型，其形状为[样本数量， 样本特征数量]
        :param y: 类数组类型，其形状为[样本数量]
        :return:
        """

        self.X = np.asarray(X)
        self.y = np.asarray(y)

    def predict(self, X):
        """
        测试函数，传入测试函数
        :param X: 类数据类型，其形状为[样本数量， 样本特征数量]
        :return: 返回预测好的结果，类数组类型
        """
        X = np.asarray(X)

        result = []
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
            # 将索引传入训练集y中，求它们的平均值
            # 例如k为3，则求y中索引为[1,2,0]的3行数据平均值作为预测结果，加入到result列表中
            result.append(np.mean(self.y[index]))
            # 一次迭代，直达将测试集全部行数计算完退出
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
            # 考虑权重后不再计算其平均值作为预测结果，权重计算如下（假定k为3）
            # 先将其前3个样本的距离倒数，加0.001是防止距离为0，即重合点
            # 然后每一个样本的权重为自己的距离倒数除以总距离倒数之和
            # 如[2,5,4]为y中索引，其计算距离为[1,2,3], 倒数为[1, 1/2, 1/3]
            # 每一个索引对应的权重为[1 / (11/6), 1/2 / (11/6), 1/3 / (11/6)]
            # 则再将索引传入y，将其对应乘以其权重再相加，则作为预测结果
            s = np.sum(1 / dis[index] + 0.001)
            weight = (1 / dis[index] + 0.001) / s
            result.append(np.sum(self.y[index] * weight))
            # 一次迭代，直达将测试集全部行数计算完退出
        return np.asarray(result)
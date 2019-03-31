import numpy as np

from sklearn.datasets import load_iris  # 导入数据集iris

iris = load_iris()  # 载入数据集


def make_one_hot(i, k):
    one = np.zeros(k)
    # one[i - 1] = 1
    np.put(one, i - 1, 1)
    return one


def cross_entropy(p1, p2):
    res = np.nan_to_num(-p1 * np.log(p2) - (1 - p1) * np.log(1 - p2))
    loss = np.average(np.sum(res, axis=1))
    return loss


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class Logistic(object):
    def __init__(self):
        """
        :param k: class num
        :param j: sample count
        """
        self.n, self.d = iris.data.shape
        self.k = 3  # 总分类数
        self.X = iris.data  # .reshape(-1, 1, self.d)  # (n,4)
        Y = np.array([make_one_hot(i, self.k) for i in iris.target])
        self.Y = Y  # .reshape((-1, 1, self.k))  # (n,1)
        self.W = np.random.random((self.k, self.d))  # (n,4)

    def train(self):
        learn_rate = 0.001
        epoch_num = 1 * 10000
        for num in range(epoch_num):
            O = np.dot(self.W, self.X.T).T  # (3,4)*(1,4).T=(3,1)
            y = np.exp(O) / np.sum(np.exp(O), axis=1).reshape(-1, 1)  # 本行的和，归一化
            dw = np.dot((self.Y - y).T, self.X)
            self.W += learn_rate * dw
            print("loss : {:04f}".format(cross_entropy(self.Y, y)))
        print(y)


class Datahelper(object):
    pass


if __name__ == "__main__":
    Logistic().train()

#  逻辑斯蒂回归 多类问题

王胜广 1801220015

## 1. 鸢尾花数据集介绍
在Sklearn机器学习包中，集成了各种各样的数据集，这里用的是鸢尾花卉（Iris）数据集。鸢尾花有三个亚属，分别是山鸢尾（Iris-setosa）、变色鸢尾（Iris-versicolor）和维吉尼亚鸢尾（Iris-virginica）。

该数据集一共包含4个特征变量，1个类别变量。共有150个样本，iris是鸢尾植物，这里存储了其萼片和花瓣的长宽，共4个属性，鸢尾植物分三类。

[[ 5.1  3.5  1.4  0.2]    
 [ 4.9  3.   1.4  0.2]    
 [ 4.7  3.2  1.3  0.2]    
 [ 4.6  3.1  1.5  0.2]    
 ....    
 [ 6.7  3.   5.2  2.3]    
 [ 6.3  2.5  5.   1.9]    
 [ 6.5  3.   5.2  2. ]    
 [ 6.2  3.4  5.4  2.3]    
 [ 5.9  3.   5.1  1.8]]    

target是一个数组，存储了data中每条记录属于哪一类鸢尾植物，所以数组的长度是150，数组元素的值因为共有3类鸢尾植物，所以不同值只有3个。种类为山鸢尾、杂色鸢尾、维吉尼亚鸢尾。
print len(iris.target)      #150个样本 每个样本4个特征
[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2
 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
 2 2]


### 2. 代码实现 


```python
import numpy as np

from sklearn.datasets import load_iris  # 导入数据集iris


def make_one_hot(i, k):
    one = np.zeros(k)
    np.put(one, i - 1, 1)
    return one


def cross_entropy(y_true, y_pred):
    res = np.nan_to_num(-y_true * np.log(y_pred) - (1 - y_true) * np.log(1 - y_pred))
    loss = np.average(np.sum(res, axis=1))
    return loss


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def softmax(y):
    _y = np.exp(y) / np.sum(np.exp(y), axis=1).reshape(-1, 1)
    return _y


iris = load_iris()  # 载入数据集
n, d = iris.data.shape  # 样本数目，样本维度
k = 3  # 类别数

X = iris.data
Y = np.array([make_one_hot(i, k) for i in iris.target])
W = np.random.random((k, d))

learning_rate = 0.001
epoch_num = 1 * 10000
for _ in range(epoch_num):
    _Y = np.dot(W, X.T).T  # 前向传播
    _Y = softmax(_Y)  # 激活函数
    dw = np.dot((Y - _Y).T, X)  # 梯度下降
    W += learning_rate * dw  # 反向传播
    print("loss : {:04f}".format(cross_entropy(Y, _Y)))
print(_Y)
```
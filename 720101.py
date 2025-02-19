import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.linear_model import LogisticRegression
import scipy.io
import pandas as pd


# 加载数据
data_train = scipy.io.loadmat("C:/Users/WYY/Downloads/data_train (1).mat")['data_train']
label_train = scipy.io.loadmat("C:/Users/WYY/Downloads/label_train (1).mat")['label_train']
data_test = scipy.io.loadmat("C:/Users/WYY/Downloads/data_test (2).mat")['data_test']


class RBFNetwork:
    def __init__(self, num_centers, gamma):
        self.num_centers = num_centers  # RBF 单元的数量
        self.gamma = gamma  # RBF 核的参数
        self.centers = None  # RBF 中心
        self.logistic_reg = None  # 用逻辑回归作为输出层

    def fit(self, X, y):
        # 使用 KMeans 选择 RBF 单元的中心
        kmeans = KMeans(n_clusters=self.num_centers, random_state=42, n_init=10)
        kmeans.fit(X)
        self.centers = kmeans.cluster_centers_

        # 计算 RBF 特征
        RBF_features = rbf_kernel(X, self.centers, gamma=self.gamma)

        # 训练逻辑回归分类器
        self.logistic_reg = LogisticRegression()
        self.logistic_reg.fit(RBF_features, y.ravel())

    def predict(self, X):
        # 计算 RBF 特征
        RBF_features = rbf_kernel(X, self.centers, gamma=self.gamma)

        # 进行预测
        return self.logistic_reg.predict(RBF_features)


# 设定 RBF 网络参数
num_centers = 20  # 选择 20 个 RBF 单元
gamma = 1.0 / (2 * np.std(data_train) ** 2)  # 计算 RBF 核的 gamma

# 训练 RBF 网络
rbf_net = RBFNetwork(num_centers=num_centers, gamma=gamma)
rbf_net.fit(data_train, label_train)

# 预测测试数据的类别
predicted_labels = rbf_net.predict(data_test)

# 显示预测结果
predictions_df = pd.DataFrame(predicted_labels, columns=["Predicted Label"])
print(predictions_df)
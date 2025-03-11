import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
import scipy.io

from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE  # 用于平衡数据
import scipy.io
import pandas as pd
import matplotlib.pyplot as plt

# 加载数据
data_train = scipy.io.loadmat("C:/Users/WYY/Downloads/data_train (1).mat")['data_train']
label_train = scipy.io.loadmat("C:/Users/WYY/Downloads/label_train (1).mat")['label_train']
data_test = scipy.io.loadmat("C:/Users/WYY/Downloads/data_test (2).mat")['data_test']

# 计算类别比例
unique, counts = np.unique(label_train, return_counts=True)
print(f"原始类别分布: {dict(zip(unique, counts))}")

# 数据标准化
scaler = StandardScaler()
data_train = scaler.fit_transform(data_train)
data_test = scaler.transform(data_test)

# 过采样少数类（让 -1 类别增加到与 1 类别相等）
smote = SMOTE(sampling_strategy='auto', random_state=42)
data_train, label_train = smote.fit_resample(data_train, label_train)

# 重新计算类别分布
unique, counts = np.unique(label_train, return_counts=True)
print(f"过采样后的类别分布: {dict(zip(unique, counts))}")

# 画出数据分布情况（可选）
plt.hist(label_train, bins=2, rwidth=0.8)
plt.xticks([-1, 1])
plt.title("Balanced Class Distribution")
plt.show()

class RBFNetwork:
    def __init__(self, num_centers, gamma):
        self.num_centers = num_centers
        self.gamma = gamma
        self.centers = None
        self.classifier = None  # SVM 分类器

    def fit(self, X, y):
        # 选择 RBF 单元中心
        kmeans = KMeans(n_clusters=self.num_centers, random_state=42, n_init=10)
        kmeans.fit(X)
        self.centers = kmeans.cluster_centers_

        # 计算 RBF 特征
        RBF_features = rbf_kernel(X, self.centers, gamma=self.gamma)

        # 检查 RBF 特征是否全一样
        print(f"RBF 特征最大值: {np.max(RBF_features)}, 最小值: {np.min(RBF_features)}")

        # 训练 SVM 分类器
        self.classifier = SVC(kernel='linear', class_weight='balanced', C=1.0)
        self.classifier.fit(RBF_features, y)

    def predict(self, X):
        RBF_features = rbf_kernel(X, self.centers, gamma=self.gamma)
        return self.classifier.predict(RBF_features)

# 交叉验证优化参数
kf = KFold(n_splits=5, shuffle=True, random_state=42)
best_accuracy = 0
best_num_centers = 10
best_gamma = 1.0

for num_centers in [5, 10, 15, 20, 25]:  # 试验不同的 RBF 单元数
    for gamma in np.logspace(-3, 1, 5):  # 试验不同的 gamma 值
        accuracies = []
        for train_idx, val_idx in kf.split(data_train):
            X_train, X_val = data_train[train_idx], data_train[val_idx]
            y_train, y_val = label_train[train_idx], label_train[val_idx]

            model = RBFNetwork(num_centers=num_centers, gamma=gamma)
            model.fit(X_train, y_train)
            val_predictions = model.predict(X_val)

            accuracy = np.mean(val_predictions == y_val)
            accuracies.append(accuracy)

        avg_accuracy = np.mean(accuracies)
        if avg_accuracy > best_accuracy:
            best_accuracy = avg_accuracy
            best_num_centers = num_centers
            best_gamma = gamma

print(f"最佳参数: num_centers={best_num_centers}, gamma={best_gamma}, 交叉验证准确率={best_accuracy:.4f}")

# 训练最终模型
rbf_net = RBFNetwork(num_centers=best_num_centers, gamma=best_gamma)
rbf_net.fit(data_train, label_train)

# 预测测试数据
predicted_labels = rbf_net.predict(data_test)

# 输出并保存预测结果
predictions_df = pd.DataFrame(predicted_labels, columns=["Predicted Label"])
print(predictions_df)
predictions_df.to_csv("optimized_predictions.csv", index=False)
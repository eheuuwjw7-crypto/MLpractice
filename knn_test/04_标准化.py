"""
标准化（Standardization）

公式：
    z = (x - μ) / σ

公式解释：
    x  : 原始数据
    μ  : 这一列特征的平均值（mean）
    σ  : 这一列特征的标准差（standard deviation）
    z  : 标准化后的结果

标准化后的特点：
    1. 每一列特征的均值接近 0
    2. 每一列特征的标准差接近 1

为什么要做标准化？
    KNN、SVM、逻辑回归、梯度下降类模型通常对特征尺度比较敏感。
    如果某个特征数值范围特别大，就可能在计算距离或优化时占据主导地位。
    所以常常需要先把不同量纲的特征拉回到可比较的范围。
"""

from sklearn.preprocessing import StandardScaler

x_train = [
    [3000, 3, 80],
    [4500, 5, 82],
    [5000, 7, 78],
    [8000, 9, 90],
    [9000, 10, 95],
    [12000, 12, 98],
]

transfer = StandardScaler()
x_train_std = transfer.fit_transform(x_train)

print("原始数据：")
print(x_train)

print("\n每列特征的均值：")
print(transfer.mean_)

print("\n每列特征的标准差：")
print(transfer.scale_)

print("\n标准化之后的数据：")
print(x_train_std)

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, root_mean_squared_error

from sklearn.linear_model import Ridge, RidgeCV

import pandas as pd
import numpy as np
from sklearn.utils._repr_html import estimator

# 1，加载数据
data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep=r"\s+", skiprows=22, header=None)
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
target = raw_df.values[1::2, 2]

# 2，数据预处理
x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=42)

# 3，特征工程
transfer = StandardScaler()
x_train = transfer.fit_transform(x_train)
x_test = transfer.transform(x_test)

# 4，模型训练
estimator = SGDRegressor(fit_intercept=True,learning_rate='constant',eta0=0.01)
# 训练
estimator.fit(x_train, y_train)
print(f'权重：{estimator.coef_}')
print(f'偏置：{estimator.intercept_}')

# 5，模型预测
y_pred = estimator.predict(x_test)
print(f'预测结果为：{y_pred}')

# 6，模型评估
print(f'均方误差为：{mean_squared_error(y_test, y_pred)}')        # MSE 公式：1/n * sum((y_true - y_pred)**2)
print(f'均方根误差：{root_mean_squared_error(y_test, y_pred)}')   # RMSE 公式：sqrt(1/n * sum((y_true - y_pred)**2))
print(f'平均绝对误差：{mean_absolute_error(y_test, y_pred)}')      # MAE 公式：1/n * sum(abs(y_true - y_pred))
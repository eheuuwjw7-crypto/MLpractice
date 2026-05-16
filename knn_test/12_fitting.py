"""
本文件通过 5 个小例子，演示模型复杂度与拟合效果之间的关系。

1. 欠拟合（underfitting）
含义：
模型太简单，无法学到数据中真正的规律。

常见表现：
- 训练集误差较大
- 测试集误差通常也较大
- 图像上看，模型曲线明显不能贴近数据趋势

常见原因：
- 模型过于简单，例如本应是曲线关系，却只用直线拟合
- 特征设计不够，无法表达真实规律
- 训练不充分

常见解决方法：
- 换更复杂一些的模型
- 增加更有表达能力的特征
- 延长训练时间或调整参数

2. 过拟合（overfitting）
含义：
模型太复杂，不仅学到了真实规律，还把训练数据中的噪声也学进去了。

常见表现：
- 训练集误差很小
- 测试集误差反而变大
- 图像上看，模型曲线扭来扭去，过度贴合样本点

常见原因：
- 模型复杂度过高，例如加入太多高次项
- 特征太多，而数据量太少
- 数据中噪声较大
- 缺少约束，模型参数可以随意变得很大

常见解决方法：
- 降低模型复杂度
- 增加训练数据
- 做特征筛选
- 使用正则化（regularization）

3. 正则化（regularization）
含义：
在原来的损失函数基础上，额外加入一个“惩罚项”，限制模型参数不要过大。

直观理解：
如果模型参数过大，往往说明模型为了贴合训练数据，做了过于激进的弯折。
正则化的作用，就是告诉模型：
“你可以拟合数据，但不要把参数放得太夸张。”

4. L1 正则化和 L2 正则化
L1 正则化：
- 典型模型是 Lasso
- 惩罚的是参数绝对值之和
- 特点是会把一部分不重要的参数直接压缩到 0
- 适合理解为“自动做特征选择”

L2 正则化：
- 典型模型是 Ridge
- 惩罚的是参数平方和
- 特点是通常不会把参数直接变成 0，而是让参数整体变小、更平滑
- 常用于缓解过拟合和多重共线性问题

一个简单记忆方式：
- L1：更容易产生稀疏解，有些系数会直接归零
- L2：更倾向于让所有系数都缩小，但不一定归零

本文件中的安排：
- dm01：故意让模型太简单，展示欠拟合
- dm02：让模型复杂度与数据规律更匹配
- dm03：故意加入很多高次项，展示过拟合
- dm04：用 L1 正则化缓解过拟合
- dm05：用 L2 正则化缓解过拟合
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression,Lasso,Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, root_mean_squared_error
from sklearn.utils._repr_html import estimator


def dm01_underfitting():
    # 固定随机种子，让你每次运行时拿到同一批随机数据，便于学习和复现结果。
    np.random.seed(42)

    # 构造一维特征 x。
    # 这里的 x 是在 [-3, 3] 区间内随机采样的 100 个点。
    x = np.random.uniform(-3, 3, size=100)

    # 构造标签 y。
    # 真正的数据规律是一个二次函数：0.5*x^2 + x + 2
    # 后面的 np.random.normal(...) 是人为加入的噪声，用来模拟真实世界中“不那么干净”的数据。
    y = 0.5*x**2 + x + 2 + np.random.normal(0,1,size=100)

    # 创建线性回归模型。
    # 注意：这里的“线性”指的是参数线性，不代表只能拟合直线。
    estimator = LinearRegression()

    # sklearn 要求特征 X 必须是二维结构，形状通常是：
    # (样本数, 特征数)
    # 当前 x 原本是一维数组，reshape(-1, 1) 后表示：
    # 100 个样本，每个样本只有 1 个特征。
    X = x.reshape(-1,1)

    # 用训练数据拟合模型。
    estimator.fit(X,y)

    # 预测结果 y_pred 是模型给出的“拟合直线”上的 y 值。
    y_pred = estimator.predict(X)

    # 均方误差 MSE：
    # 真实值与预测值差得越大，MSE 越大。
    # 这里因为真实规律其实是曲线，而模型只是一条直线，所以误差会偏大。
    my_ret = mean_squared_error(y,y_pred)
    print(f'均方误差：{my_ret}')

    # 散点图：原始样本点
    plt.scatter(x,y)

    # 直接用原始 x 连线时，x 的顺序是随机的，画出来会比较乱。
    # 这个示例主要是想突出“欠拟合”，所以仍然保留原写法。
    plt.plot(x,y_pred,color='r')
    plt.show()

def dm02_justfitting():
    # 同样先固定随机种子，确保三段示例使用一致的数据生成方式。
    np.random.seed(42)
    x = np.random.uniform(-3, 3, size=100)
    y = 0.5 * x ** 2 + x + 2 + np.random.normal(0, 1, size=100)

    estimator = LinearRegression()

    X = x.reshape(-1, 1)

    # 关键点：
    # 原始特征只有 x，本质上只能表达“一次关系”。
    # 这里额外加入 x^2，模型虽然仍是线性回归，
    # 但它现在学习的是：y = w1*x + w2*x^2 + b
    # 这样就具备了拟合二次曲线的能力。
    X_2 = np.hstack([X,X**2])
    estimator.fit(X_2,y)

    y_pred = estimator.predict(X_2)

    # 由于真实数据本来就接近二次关系，所以这里通常会比上一段拟合得更合理。
    my_ret = mean_squared_error(y,y_pred)
    print(f'均方误差：{my_ret}')

    plt.scatter(x,y)

    # 这里先对 x 排序，再按同样顺序取出预测值，
    # 这样画出的红线是平滑的，不会来回折返。
    plt.plot(np.sort(x),y_pred[np.argsort(x)],color='r')
    plt.show()

def dm03_overfitting():
    # 为了方便比较，这里仍然使用同样的数据生成方式。
    np.random.seed(42)
    x = np.random.uniform(-3, 3, size=100)
    y = 0.5 * x ** 2 + x + 2 + np.random.normal(0, 1, size=100)

    estimator = LinearRegression()

    X = x.reshape(-1, 1)

    # 这里人为加入了很多高次项，从 x 一直到 x^10。
    # 模型形式会变得非常灵活，可以拼命贴合训练数据中的细小波动。
    # 问题是：这些细小波动里有一部分其实只是噪声，不是真正规律。
    # 所以训练集上看起来可能更“贴”，但泛化能力往往变差，这就是过拟合。
    X_3 = np.hstack([X,X**2,X**3,X**4,X**5,X**6,X**7,X**8,X**9,X**10])
    estimator.fit(X_3,y)

    y_pred = estimator.predict(X_3)

    # 只看训练误差并不能完全判断模型好坏。
    # 过拟合模型在训练集上的误差可能很低，但在新数据上的表现可能更差。
    my_ret = mean_squared_error(y,y_pred)
    print(f'均方误差：{my_ret}')

    plt.scatter(x,y)
    plt.plot(np.sort(x),y_pred[np.argsort(x)],color='r')
    plt.show()

def dm04_overfitting_regularization_L1():
    # 仍然生成同样风格的数据，方便和前面的过拟合例子直接比较。
    np.random.seed(42)
    x = np.random.uniform(-3, 3, size=100)
    y = 0.5 * x ** 2 + x + 2 + np.random.normal(0, 1, size=100)

    # Lasso 是带 L1 正则化的线性模型。
    # alpha 表示正则化强度：
    # alpha 越大，对参数的约束越强，模型越不容易“乱摆动”。
    # max_iter 表示最大迭代次数。
    # 因为 Lasso 需要通过迭代优化参数，所以有时要增大这个值才能更容易收敛。
    estimator = Lasso(alpha=5,max_iter=10000)

    X = x.reshape(-1, 1)

    # 这里依然故意构造高次多项式特征。
    # 目的是说明：即使特征很多，也可以通过正则化抑制过拟合。
    X_3 = np.hstack([X, X ** 2, X ** 3, X ** 4, X ** 5, X ** 6, X ** 7, X ** 8, X ** 9, X ** 10])

    # fit(...)：训练模型，让模型根据输入特征 X_3 和目标值 y 学习参数。
    estimator.fit(X_3, y)

    # coef_：训练完成后学到的权重系数。
    # 对于 Lasso 来说，一部分系数可能被直接压到 0，
    # 这说明模型认为这些特征不重要。
    print(f'权重',estimator.coef_)

    # predict(...)：用训练好的模型，根据特征去预测结果。
    y_pred = estimator.predict(X_3)

    # 观察训练集误差。
    my_ret = mean_squared_error(y, y_pred)
    print(f'均方误差：{my_ret}')

    plt.scatter(x, y)
    plt.plot(np.sort(x), y_pred[np.argsort(x)], color='r')
    plt.show()

def dm05_overfitting_regularization_L2():
    # 数据构造方式与前面保持一致，便于横向比较。
    np.random.seed(42)
    x = np.random.uniform(-3, 3, size=100)
    y = 0.5 * x ** 2 + x + 2 + np.random.normal(0, 1, size=100)


    # Ridge 是带 L2 正则化的线性模型。
    # alpha 同样表示正则化强度。
    # Ridge 通常会让权重变小，但不一定像 Lasso 那样直接压成 0。
    # max_iter 表示迭代上限，不同求解器下它的作用会略有区别。
    estimator = Ridge(alpha=10,max_iter=10000)
    X = x.reshape(-1, 1)

    X_3 = np.hstack([X, X ** 2, X ** 3, X ** 4, X ** 5, X ** 6, X ** 7, X ** 8, X ** 9, X ** 10])

    # fit(...)：根据特征和标签学习模型参数。
    estimator.fit(X_3, y)

    # coef_：查看各个特征对应的权重。
    # Ridge 一般会保留大多数特征，只是把它们的系数缩小。
    print(f'权重',estimator.coef_)

    # predict(...)：使用训练好的模型进行预测。
    y_pred = estimator.predict(X_3)

    my_ret = mean_squared_error(y, y_pred)
    print(f'均方误差：{my_ret}')

    plt.scatter(x, y)
    plt.plot(np.sort(x), y_pred[np.argsort(x)], color='r')
    plt.show()

if __name__ == '__main__':
    # 欠拟合：模型太简单，抓不住真实规律
    dm01_underfitting()

    # 拟合较合适：模型复杂度与数据规律更匹配
    dm02_justfitting()

    # 过拟合：模型太复杂，连噪声也一起学进去了
    dm03_overfitting()

    # L1 正则化：更容易把不重要的特征系数压缩到 0
    dm04_overfitting_regularization_L1()

    # L2 正则化：更倾向于让全部特征系数整体变小
    dm05_overfitting_regularization_L2()

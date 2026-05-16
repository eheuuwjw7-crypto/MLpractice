import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

def dm01_regression_classifier():
    # 1. 构造一组简单的一维回归数据。
    # reshape(-1, 1) 的作用是把一维数组变成二维特征矩阵，符合 sklearn 的输入要求。
    x = np.array(list(range(1,11))).reshape(-1,1)
    y = np.array([1.3,2.1,4.3,5.2,6.9,7.1,8.5,9.7,10.2,11.5])

    # 2. 创建三个模型进行对比。
    # 决策树回归会把特征空间切成多个区间，每个区间预测一个固定值。
    # max_depth 越大，树越复杂，拟合能力越强，也越容易过拟合。
    mode1 = DecisionTreeRegressor(max_depth=1)
    mode2 = DecisionTreeRegressor(max_depth=3)

    # 线性回归会拟合一条直线，用来和决策树回归的阶梯状预测效果做对比。
    mode3 = LinearRegression ()

    # 3. 训练模型。
    mode1.fit(x,y)
    mode2.fit(x,y)
    mode3.fit(x,y)

    # 4. 生成更密集的测试点，用于画出平滑的预测曲线。
    # np.arange(0.00, 10, 0.01) 会生成 0 到 10 之间步长为 0.01 的测试数据。
    x_test = np.arange(0.00,10,0.01).reshape(-1,1)

    # 5. 分别得到三个模型在测试点上的预测结果。
    y_pred1 = mode1.predict(x_test)
    y_pred2 = mode2.predict(x_test)
    y_pred3 = mode3.predict(x_test)
    print(y_pred1.shape)

    # 6. 绘图对比。
    # 散点是真实样本，三条线分别表示不同模型的预测结果。
    plt.figure(figsize=(10,6),dpi=100)
    plt.scatter(x,y,label='data')

    plt.plot(x_test,y_pred1,label='max_depth = 1')
    plt.plot(x_test,y_pred2,label='max_depth = 3')
    plt.plot(x_test,y_pred3,label='linear')
    
    plt.xlabel('data')
    plt.ylabel('target')
    plt.title('DecisionTreeRegression')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    dm01_regression_classifier()

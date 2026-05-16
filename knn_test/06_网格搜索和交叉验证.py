"""
交叉验证和网格搜索教学案例

一、什么是交叉验证（Cross Validation）？
    交叉验证的核心思想是：
    把训练数据分成 K 份，每次拿其中 1 份做验证集，剩下 K-1 份做训练集，
    重复 K 次，最后把 K 次结果求平均。

    例如 5 折交叉验证：
    第 1 次：第 1 份做验证，其余做训练
    第 2 次：第 2 份做验证，其余做训练
    ...
    第 5 次：第 5 份做验证，其余做训练

    平均准确率 = (第1次准确率 + 第2次准确率 + ... + 第K次准确率) / K

作用：
    1. 结果更稳定，不容易因为一次随机划分而偶然偏高或偏低
    2. 能更可靠地评估模型效果
    3. 常用于模型调参

二、什么是网格搜索（Grid Search）？
    网格搜索就是把我们想尝试的参数全部列出来，
    然后让程序把这些参数组合一个一个试过去。

    例如 KNN 中常见的参数：
    1. n_neighbors：选择几个邻居
    2. weights：距离投票方式
    3. p：距离计算方式，p=1 是曼哈顿距离，p=2 是欧氏距离

三、为什么交叉验证常和网格搜索一起使用？
    因为只看一次训练集/测试集划分不够稳定。
    所以我们通常会让每一组参数都做一次交叉验证，
    最后选择平均表现最好的那一组参数。
"""

from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def dm_01_cross_validation():
    """先单独演示交叉验证。"""
    iris = load_iris()

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('knn', KNeighborsClassifier(n_neighbors=5)),
    ])

    scores = cross_val_score(pipeline, iris.data, iris.target, cv=5)

    print('5折交叉验证每一折的准确率：')
    print(scores)
    print(f'平均准确率：{scores.mean():.4f}')


def dm_02_grid_search():
    """演示网格搜索 + 交叉验证。"""
    iris = load_iris()

    x_train, x_test, y_train, y_test = train_test_split(
        iris.data,
        iris.target,
        test_size=0.2,
        random_state=23,
        stratify=iris.target
    )

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('knn', KNeighborsClassifier()),
    ])

    param_grid = {
        'knn__n_neighbors': [i for i in range(1,11)],
        'knn__weights': ['uniform', 'distance'],
        'knn__p': [1, 2],
    }

    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1
    )

    grid.fit(x_train, y_train)

    print('最优参数：')
    print(grid.best_params_)

    print(f'\n交叉验证中的最佳平均准确率：{grid.best_score_:.4f}')

    best_model = grid.best_estimator_
    y_pred = best_model.predict(x_test)

    print(f'\n测试集准确率：{accuracy_score(y_test, y_pred):.4f}')
    print('\n分类报告：')
    print(classification_report(y_test, y_pred, target_names=iris.target_names))

    my_data = [[6.5, 3.0, 5.2, 2.0]]
    pred = best_model.predict(my_data)
    print(f'新样本预测类别：{iris.target_names[pred[0]]}')


def main():
    print('========== 交叉验证演示 ==========')
    dm_01_cross_validation()

    print('\n========== 网格搜索 + 交叉验证演示 ==========')
    dm_02_grid_search()


if __name__ == '__main__':
    main()

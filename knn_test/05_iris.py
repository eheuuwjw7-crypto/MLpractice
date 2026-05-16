"""
机器学习标准研发流程:
    1,加载数据
    2,数据的预处理
    3,特征工程(提取，预处理)
    4,模型训练
    5,模型评估
    6,模型预测
"""
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def dm_01_load_iris():
    iris = load_iris()
    # print(f'数据集：{iris}')
    # print(f'数据集的类型：{type(iris)}')
    # print(f'数据集所有的键：{iris.keys()}')
    print(f'{iris.data[:5]}')
    print(f'{iris.target_names[:5]}')
    print(f'{iris.feature_names[:5]}')

def dm_02_iris_show():
    iris_data = load_iris()
    iris_df = pd.DataFrame(iris_data.data, columns=iris_data.feature_names)
    iris_df['label'] = iris_data.target

    sns.lmplot(data=iris_df, x='sepal length (cm)', y='sepal width (cm)', hue='label', fit_reg=True)
    plt.title('iris data')
    plt.tight_layout()
    plt.show()

def dm_03_split_train_test():
    iris_data = load_iris()

    x_train, x_test, y_train, y_test = train_test_split(
        iris_data.data,
        iris_data.target,
        test_size=0.2,
        random_state=23,
        stratify=iris_data.target
    )
    print(f'训练集特征{x_train}，个数：{len(x_train)}')
    print(f'训练集标签{y_train}，个数：{len(y_train)}')
    print(f'测试集特征{x_test}，个数：{len(x_test)}')
    print(f'测试集标签{y_test}，个数：{len(y_test)}')

def main():
    """机器学习标准研发流程:
    1, 加载数据
    2, 数据的预处理
    3, 特征工程(提取，预处理)
    4, 模型训练
    5, 模型评估
    6, 模型预测
    """
    iris = load_iris()
    x_train, x_test, y_train, y_test = train_test_split(
        iris.data,
        iris.target,
        test_size=0.2,
        random_state=23,
        stratify=iris.target
    )
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(x_train, y_train)

    y_pred = knn.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'准确率:{knn.score(x_test, y_test)}')
    print(f'准确率: {accuracy:.4f}')
    print(classification_report(y_test, y_pred, target_names=iris.target_names))
    # my_data = [[6.6, 3.1, 4.3, 1.9]]
    # my_data = transfer.transform(my_data)
    # pred = knn.predict(my_data)
    # pred_proba = knn.predict_proba(my_data)
    # print(f'预测值为{pred}')
    # print('预测类别:', iris.target_names[pred])
    # print(f'预测概率为{pred_proba}')

if __name__ == '__main__':
    main()

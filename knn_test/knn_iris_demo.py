from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def main():
    # 1. 加载数据
    iris = load_iris()
    X = iris.data
    y = iris.target
    target_names = iris.target_names

    # 2. 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 3. 标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 4. 创建 KNN 模型
    knn = KNeighborsClassifier(n_neighbors=5)

    # 5. 训练模型
    knn.fit(X_train_scaled, y_train)

    # 6. 预测
    y_pred = knn.predict(X_test_scaled)

    # 7. 评估模型
    accuracy = accuracy_score(y_test, y_pred)
    print('=== Iris KNN 分类案例 ===')
    print(f'准确率: {accuracy:.4f}')
    print('\n分类报告:')
    print(classification_report(y_test, y_pred, target_names=target_names))
    print('混淆矩阵:')
    print(confusion_matrix(y_test, y_pred))

    # 8. 示例预测
    sample = [[5.1, 3.5, 1.4, 0.2]]
    sample_scaled = scaler.transform(sample)
    pred = knn.predict(sample_scaled)[0]
    print('\n示例样本:', sample[0])
    print('预测类别:', target_names[pred])


if __name__ == '__main__':
    main()

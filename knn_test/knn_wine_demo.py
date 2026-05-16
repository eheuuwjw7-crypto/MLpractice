import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def main():
    # 1. 读取外部数据
    df = pd.read_csv('../data/winequality-red.csv', sep=';')

    # 2. 构造二分类目标：quality >= 6 为 good，否则为 bad
    df['label'] = (df['quality'] >= 6).astype(int)

    X = df.drop(columns=['quality', 'label'])
    y = df['label']

    # 3. 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 4. 标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 5. 创建 KNN 模型
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train_scaled, y_train)

    # 6. 预测
    y_pred = knn.predict(X_test_scaled)

    # 7. 输出结果
    accuracy = accuracy_score(y_test, y_pred)
    print('=== Wine Quality KNN 分类案例 ===')
    print('标签说明: 1=good (quality >= 6), 0=bad (quality < 6)')
    print(f'准确率: {accuracy:.4f}')
    print('\n分类报告:')
    print(classification_report(y_test, y_pred, target_names=['bad', 'good']))
    print('混淆矩阵:')
    print(confusion_matrix(y_test, y_pred))


if __name__ == '__main__':
    main()

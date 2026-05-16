from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
TRAIN_PATH = DATA_DIR / "titanic_train.csv"
TEST_PATH = DATA_DIR / "titanic_test.csv"
PREDICT_PATH = DATA_DIR / "titanic_decision_tree_predict.csv"


def prepare_features(df, age_median, fare_median, embarked_mode):
    """
    把 Titanic 原始数据整理成决策树可以直接学习的特征。

    教学重点：
    1. 决策树只能接收数值特征，Sex、Embarked 这类文本特征要先编码。
    2. Age、Fare、Embarked 有缺失值，需要先填补。
    3. Name、Ticket、Cabin 文本信息复杂，本案例先不使用，降低入门难度。
    """
    x = df[["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]].copy()

    x["Age"] = x["Age"].fillna(age_median)
    x["Fare"] = x["Fare"].fillna(fare_median)
    x["Embarked"] = x["Embarked"].fillna(embarked_mode)

    # 把类别特征转换成哑变量，例如 Sex -> Sex_female / Sex_male。
    x = pd.get_dummies(x, columns=["Sex", "Embarked"])

    return x


def dm01_titanic_decision_tree():
    # 1. 读取数据
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    print("训练集前 5 行：")
    print(train_df.head())
    print("\n训练集缺失值统计：")
    print(train_df.isnull().sum())
    print("\n是否存活人数统计：")
    print(train_df["Survived"].value_counts())

    # 2. 准备特征和标签
    # 这些统计量必须从训练集计算，后面验证集、测试集都复用它们。
    age_median = train_df["Age"].median()
    fare_median = train_df["Fare"].median()
    embarked_mode = train_df["Embarked"].mode()[0]

    x = prepare_features(train_df, age_median, fare_median, embarked_mode)
    y = train_df["Survived"]

    print("\n模型使用的特征：")
    print(list(x.columns))

    # 3. 划分训练集和验证集
    x_train, x_valid, y_train, y_valid = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # 4. 创建并训练决策树模型
    # max_depth 控制树的深度，避免模型把训练集记得太死，出现过拟合。
    estimator = DecisionTreeClassifier(
        criterion="gini",
        max_depth=4,
        min_samples_leaf=5,
        random_state=42
    )
    estimator.fit(x_train, y_train)

    # 5. 在验证集上评估模型
    y_pred = estimator.predict(x_valid)
    print("\n验证集预测结果前 20 个：")
    print(y_pred[:20])
    print(f"\n验证集准确率：{accuracy_score(y_valid, y_pred):.4f}")
    print("\n分类报告：")
    print(classification_report(y_valid, y_pred, target_names=["未存活", "存活"]))

    # 6. 查看每个特征的重要性
    feature_importance = pd.DataFrame({
        "feature": x.columns,
        "importance": estimator.feature_importances_
    }).sort_values(by="importance", ascending=False)

    print("\n特征重要性：")
    print(feature_importance)

    # 7. 使用训练好的模型预测官方测试集，并保存结果
    test_x = prepare_features(test_df, age_median, fare_median, embarked_mode)
    test_x = test_x.reindex(columns=x.columns, fill_value=0)
    test_pred = estimator.predict(test_x)

    result_df = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": test_pred
    })
    result_df.to_csv(PREDICT_PATH, index=False)
    print(f"\n测试集预测结果已保存到：{PREDICT_PATH}")
    print(result_df.head())

    # 8. 可视化决策树
    plt.figure(figsize=(18, 10))
    plot_tree(
        estimator,
        feature_names=x.columns,
        class_names=["Died", "Survived"],
        filled=True,
        rounded=True,
        fontsize=9
    )
    plt.title("Titanic Decision Tree")
    plt.show()


if __name__ == "__main__":
    dm01_titanic_decision_tree()

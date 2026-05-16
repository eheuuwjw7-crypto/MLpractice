"""
手写数字识别教学版

这个版本和 07 的目标一样：
    1. 读取 digits.csv
    2. 训练数字识别模型
    3. 保存模型
    4. 加载模型
    5. 识别 demo.png

这个版本重点改进了两件事：
    1. 训练和预测使用完全一致的预处理逻辑
    2. 预测时同时尝试原图和反相图，方便排查“为什么总是识别错”
"""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler


DATA_PATH = Path(r"C:\Users\ZhuanZ\PycharmProjects\ML_Project\data\digits.csv")
IMAGE_PATH = Path(r"C:\Users\ZhuanZ\PycharmProjects\ML_Project\data\demo.png")
MODEL_DIR = Path(r"/ML_Project/model/models")
MODEL_PATH = MODEL_DIR / "digits_knn_pipeline.pkl"


def load_digits_data():
    """读取 CSV 数据，并拆分成特征和标签。"""
    df = pd.read_csv(DATA_PATH)
    x = df.iloc[:, 1:]
    y = df.iloc[:, 0]
    return x, y


def show_dataset_digit(index=0):
    """显示数据集中的某一张图片。"""
    x, y = load_digits_data()

    if index < 0 or index >= len(x):
        print("索引越界")
        return

    image = x.iloc[index].values.reshape(28, 28)
    print(f"数据集中的标签是：{y.iloc[index]}")
    plt.imshow(image, cmap="gray")
    plt.title(f"label = {y.iloc[index]}")
    plt.axis("off")
    plt.show()


def build_model():
    """
    建立一个流水线模型。

    MinMaxScaler:
        把像素从原始范围缩放到 0~1

    KNeighborsClassifier:
        使用 KNN 做分类
    """
    model = Pipeline([
        ("scaler", MinMaxScaler()),
        ("knn", KNeighborsClassifier(n_neighbors=5, weights="distance")),
    ])
    return model


def train_and_save_model():
    """训练模型并保存。"""
    x, y = load_digits_data()

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=23,
        stratify=y,
    )

    model = build_model()
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    print(f"测试集准确率：{accuracy_score(y_test, y_pred):.4f}")
    print("分类报告：")
    print(classification_report(y_test, y_pred))

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"模型已保存到：{MODEL_PATH}")


def load_image_as_dataframe(image_path):
    """
    读取图片并整理成模型可用的 1 行 784 列数据。

    关键点：
        1. 保证是灰度图
        2. 保证形状是 28x28
        3. 保证和训练数据一样是 784 个像素特征
    """
    x_train_columns, _ = load_digits_data()
    feature_columns = x_train_columns.columns

    image = plt.imread(image_path)

    if image.ndim == 3:
        image = image[:, :, 0]

    if image.shape != (28, 28):
        raise ValueError(f"图片尺寸必须是 28x28，当前尺寸是：{image.shape}")

    if image.max() <= 1:
        image = image * 255

    image_df = pd.DataFrame(image.reshape(1, -1), columns=feature_columns)
    return image, image_df


def predict_demo_image():
    """
    识别 demo.png。

    同时尝试：
        1. 原图
        2. 反相图

    这样更容易发现你的图片到底是“白底黑字”还是“黑底白字”。
    """
    if not MODEL_PATH.exists():
        print("模型文件不存在，先自动训练并保存模型...")
        train_and_save_model()

    model = joblib.load(MODEL_PATH)
    image, image_df = load_image_as_dataframe(IMAGE_PATH)

    inverted_df = pd.DataFrame(
        (255 - image).reshape(1, -1),
        columns=image_df.columns,
    )

    pred_raw = model.predict(image_df)[0]
    pred_invert = model.predict(inverted_df)[0]

    print(f"原图预测结果：{pred_raw}")
    print(f"反相图预测结果：{pred_invert}")

    plt.figure(figsize=(6, 3))

    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap="gray")
    plt.title(f"raw -> {pred_raw}")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(255 - image, cmap="gray")
    plt.title(f"invert -> {pred_invert}")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


def main():
    """直接运行即可；如果模型不存在，会先自动训练。"""
    # show_dataset_digit(0)
    # train_and_save_model()
    predict_demo_image()


if __name__ == "__main__":
    main()

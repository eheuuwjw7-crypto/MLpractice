import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import joblib
from collections import Counter

from sympy.polys.matrices import dfm


def show_digit(idx):
    df = pd.read_csv(r'C:\Users\ZhuanZ\PycharmProjects\ML_Project\data\digits.csv')
    # print(df)

    if idx < 0 or idx >= len(df) - 1:
        print('索引越界')
        return

    x = df.iloc[:, 1:]
    y = df.iloc[:, 0]
    print(f'该图片所对应的数字是：{y[idx]}')

    print(x.iloc[idx].shape)
    # print(x.iloc[idx].values)
    x = x.iloc[idx].values.reshape(28, 28)
    # print(x)
    plt.imshow(x, cmap='gray')
    plt.axis('off')
    plt.show()

def train_model():
    df = pd.read_csv(r'C:\Users\ZhuanZ\PycharmProjects\ML_Project\data\digits.csv')
    # print(df)
    x = df.iloc[:, 1:]
    y = df.iloc[:, 0]
    # print(f'x的形状：{x.shape}')
    # print(f'y的形状：{y.shape}')
    print(f'y标签的分布情况：{Counter(y)}')

    x = x/255
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=23,
        stratify=y
    )
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(x_train, y_train)

    print(f'准确率：{knn.score(x_test, y_test):.4f}')
    print(f'准确率：{accuracy_score(y_test, knn.predict(x_test))}')

    joblib.dump(knn, r'/ML_Project/model/model1/digits_model.pkl')

def use_model():
    x = plt.imread(r'C:\Users\ZhuanZ\PycharmProjects\ML_Project\data\demo.png')
    # plt.imshow(x,cmap='gray')
    # plt.axis('off')
    # plt.show()

    estimator = joblib.load(r'/ML_Project/model/model1\digits_model.pkl')
    # print(x.reshape(1,-1).shape)
    x = x.reshape(1,-1)

    y_pred = estimator.predict(x)
    print(f'预测值为：{y_pred}')

if __name__ == '__main__':
    #绘制图形
    # show_digit()
    # train_model()
    use_model()
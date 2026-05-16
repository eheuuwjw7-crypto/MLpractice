from sklearn.linear_model import LinearRegression
from sklearn.utils._repr_html import estimator


def dm01_lr_predict():
    x = [[160],[166],[170],[180],[190]]
    y = [56.3, 60.5, 65.3, 73.5, 84.0]

    lr = LinearRegression()
    lr.fit(x, y)

    print('lr.coef --> ', lr.coef_)
    print('lr.intercept --> ', lr.intercept_)

    pred = lr.predict([[176]])
    print('预测176的体重：', pred)

if __name__ == '__main__':
    dm01_lr_predict()
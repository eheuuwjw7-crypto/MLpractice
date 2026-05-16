import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score


def dm01_gbdt():
    data = pd.read_csv(r'C:\Users\ZhuanZ\PycharmProjects\ML_Project\data\titanic_train.csv')
    x = data[['Pclass', 'Age', 'Sex']].copy()
    y = data['Survived']
    x['Age'] = x['Age'].fillna(x['Age'].mean())

    x = pd.get_dummies(x)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
    clf = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=3, random_state=0)
    clf.fit(x_train, y_train)
    print(f'梯度决策树精确度为：{accuracy_score(y_test, clf.predict(x_test))}')

    param = {'n_estimators': [100, 200, 300, 400, 500], 'learning_rate': [0.1, 0.01, 0.001],'max_depth': [3, 4, 5, 6, 7]}
    gdbt = GridSearchCV(estimator=GradientBoostingClassifier(max_depth=5, random_state=0), param_grid=param, cv=3)
    gdbt.fit(x_train, y_train)
    print(f'网格搜索精确度为：{accuracy_score(y_test, gdbt.predict(x_test))}')
    print(f'最佳参数为：{gdbt.best_params_}')

if __name__ == '__main__':
    dm01_gbdt()
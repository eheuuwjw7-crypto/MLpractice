import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score

def dm01_adaboost():
    df_wine = pd.read_csv(r'C:\Users\ZhuanZ\PycharmProjects\ML_Project\data\wine.data.csv')
    # df_wine.info()
    # print(df_wine['class'].unique())
    df_wine = df_wine[df_wine['class'] != 1]
    # print(df_wine['class'].unique())
    x = df_wine[['alcohol', 'hue']]
    y = df_wine['class']
    le = LabelEncoder()
    y = le.fit_transform(y)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    dtc = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=42)
    dtc.fit(x_train, y_train)
    dtc_y_pred = dtc.predict(x_test)
    print(f'单一决策树准确率:{accuracy_score(y_test, dtc_y_pred):.4f}')

    abc = AdaBoostClassifier(estimator=dtc, n_estimators=500, learning_rate=0.1, random_state=42)
    abc.fit(x_train, y_train)
    abc_y_pred = abc.predict(x_test)
    print(f'预测值为：{abc_y_pred}')
    print(f'AdaBoost准确率:{accuracy_score(y_test, abc_y_pred):.4f}')

if __name__ == '__main__':
    dm01_adaboost()
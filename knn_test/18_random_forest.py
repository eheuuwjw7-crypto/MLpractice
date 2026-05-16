import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

def dm01_random_forest():
    # 1. 读取 Titanic 训练集。
    # Survived 是目标标签，0 表示未存活，1 表示存活。
    titan = pd.read_csv(r'C:\Users\ZhuanZ\PycharmProjects\ML_Project\data\titanic_train.csv')

    # 2. 选择特征和标签。
    # 这里为了方便讲解，只选择 Pclass、Age、Sex 三个特征。
    x = titan[['Pclass','Age','Sex']].copy()
    y = titan['Survived']

    # 3. 缺失值处理。
    # Age 有缺失值，用年龄中位数填充，避免模型训练时报错。
    x['Age'] = x['Age'].fillna(x['Age'].median())
    # x.info()

    # 4. 类别特征编码。
    # Sex 是字符串，不能直接输入 sklearn 模型，所以用 get_dummies 转成数值列。
    x = pd.get_dummies(x,columns=['Sex'])

    # 5. 划分训练集和验证集。
    # random_state 固定随机种子，方便重复实验和课堂演示。
    x_train,x_valid,y_train,y_valid = train_test_split(x,y,test_size=0.2,random_state=42)

    # 6. 先训练单棵决策树，作为随机森林的对比基准。
    # 决策树可解释性强，但容易过拟合。
    dtc = DecisionTreeClassifier(criterion='gini',max_depth=4,min_samples_leaf=5,random_state=42)
    dtc.fit(x_train,y_train)
    dtc_y_pred = dtc.predict(x_valid)
    accuracy = accuracy_score(y_valid,dtc_y_pred)
    print(f'决策树准确率:{accuracy:.4f}')

    # 7. 再训练随机森林。
    # 随机森林由多棵决策树组成，最后通过投票得到分类结果。
    # n_estimators=100 表示森林里有 100 棵树。
    rfc = RandomForestClassifier(n_estimators=100,criterion='gini',max_depth=4,min_samples_leaf=5,random_state=42)
    rfc.fit(x_train,y_train)
    rfc_y_pred = rfc.predict(x_valid)
    accuracy = accuracy_score(y_valid,rfc_y_pred)
    print(f'随机森林准确率:{accuracy:.4f}')

    # 8. 使用网格搜索自动寻找更合适的随机森林参数。
    # GridSearchCV 会尝试 param 中所有参数组合，并用交叉验证选择表现最好的组合。
    estimator = RandomForestClassifier()
    param = {
        # 树的数量。树越多，模型通常越稳定，但训练时间也越长。
        'n_estimators': [100,200,300],

        # 节点划分标准：gini 表示基尼系数，entropy 表示信息熵。
        'criterion': ['gini','entropy'],

        # 树的最大深度，用于控制模型复杂度。
        'max_depth': [4,6,8,10],

        # 叶子节点的最小样本数，用于防止树分得过细。
        'min_samples_leaf': [5,10,20,30],

        # 固定随机种子，保证搜索过程可复现。
        'random_state': [42]
    }

    # cv=2 表示使用 2 折交叉验证。
    grid_search = GridSearchCV(estimator,param_grid=param,cv=2)
    grid_search.fit(x_train,y_train)

    # 9. 用验证集评估网格搜索得到的最优模型，并输出最佳参数。
    accuracy = grid_search.score(x_valid,y_valid)
    print(f'随机森林网格搜索准确率:{accuracy:.4f}')
    print(f'随机森林网格搜索参数:{grid_search.best_params_}')


if __name__ == '__main__':
    dm01_random_forest()

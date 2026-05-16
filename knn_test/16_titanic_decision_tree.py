from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree

# pathlib.Path 用来拼接文件路径，避免把绝对路径写死在 read_csv 里。
# 当前文件在 ML_Project/knn_test 下，parents[1] 表示向上两级定位到 ML_Project。
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
TRAIN_PATH = DATA_DIR / "titanic_train.csv"
TEST_PATH = DATA_DIR / "titanic_test.csv"
PREDICT_PATH = DATA_DIR / "titanic_decision_tree_predict.csv"

# 1. 读取 Titanic 训练集。训练集中 Survived 是标签，表示乘客是否存活。
data = pd.read_csv(TRAIN_PATH)
# data.info()

# 2. 选择用于训练模型的特征。
# Pclass：船舱等级，Sex：性别，Age：年龄。
# 决策树只能直接处理数值特征，后面会把 Sex 转成哑变量。
x = data[['Pclass','Sex','Age']]
y = data['Survived']

# 3. copy() 的作用是复制一份特征数据，避免后续修改 x 时影响原始 data。
x = x.copy()

# 4. Age 存在缺失值，用年龄中位数进行填充。
# 注意：不要写 fillna(..., inplace=True) 再赋值，否则新版 pandas 会报链式赋值错误。
x['Age'] = x['Age'].fillna(x['Age'].median())

# 5. 把 Sex 这种文本类别特征转换成数值特征。
# 例如生成 Sex_female 和 Sex_male 两列，值为 True/False。
x = pd.get_dummies(x,columns=['Sex'])

# 6. 划分训练集和验证集。
# 训练集用于训练模型，验证集用于评估模型在新数据上的表现。
x_train,x_valid,y_train,y_valid = train_test_split(x,y,test_size=0.2,random_state=42)

# 7. 创建决策树分类器。
# criterion='gini'：使用基尼系数选择最优划分。
# max_depth=4：限制树最大深度，防止树过深导致过拟合。
# min_samples_leaf=5：叶子节点至少保留 5 个样本，防止分得过细。
# random_state=42：固定随机种子，保证每次运行结果尽量一致。
estimator = DecisionTreeClassifier(criterion='gini',max_depth=4,min_samples_leaf=5,random_state=42)

# 8. 训练模型。
estimator.fit(x_train,y_train)

# 9. 使用验证集进行预测，并输出分类评估报告。
# classification_report 会输出 precision、recall、f1-score 等指标。
y_pred = estimator.predict(x_valid)
print(f'预测值为：{y_pred[:10]}')
print(f'分类评估报告：\n{classification_report(y_valid,y_pred)}')

# 10. 可视化决策树。
# filled=True 表示用颜色区分类别纯度，rounded=True 表示节点边框圆角。
plt.figure(figsize=(30,20))
plot_tree(estimator,feature_names=x.columns,class_names=['Died','Survived'],filled=True,rounded=True,fontsize=9)
plt.show()

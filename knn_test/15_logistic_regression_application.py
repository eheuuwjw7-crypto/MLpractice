import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,classification_report


def dm01_data_processing():
    churn_df = pd.read_csv(r"C:\Users\ZhuanZ\PycharmProjects\ML_Project\data\churn.csv")
    # churn_df.info()
    churn_df = pd.get_dummies(churn_df, columns=["Churn", "gender"])
    # churn_df.info()
    # print(churn_df.head(5))
    churn_df.drop(['Churn_No','gender_Male'],axis=1,inplace=True)
    churn_df.rename(columns={'Churn_Yes':'Flag'},inplace=True)
    # churn_df.info()
    # print(churn_df.head(5))
    print(churn_df.Flag.value_counts())

def dm02_data_visualization():
    churn_df = pd.read_csv(r"C:\Users\ZhuanZ\PycharmProjects\ML_Project\data\churn.csv")
    churn_df = pd.get_dummies(churn_df, columns=["Churn", "gender"])
    churn_df.drop(['Churn_No','gender_Male'],axis=1,inplace=True)
    churn_df.rename(columns={'Churn_Yes':'Flag'},inplace=True)
    print(churn_df.Flag.value_counts())
    sns.countplot(x="Contract_Month",data=churn_df,hue = 'Flag')
    plt.show()

def dm03_logistic_regression():
    churn_df = pd.read_csv(r"C:\Users\ZhuanZ\PycharmProjects\ML_Project\data\churn.csv")
    churn_df = pd.get_dummies(churn_df, columns=["Churn", "gender"])
    churn_df.drop(['Churn_No','gender_Male'],axis=1,inplace=True)
    churn_df.rename(columns={'Churn_Yes':'Flag'},inplace=True)
    x = churn_df[['Contract_Month','internet_other','PaymentElectronic']]
    y = churn_df['Flag']
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)
    estimator = LogisticRegression()
    estimator.fit(x_train,y_train)
    y_pred = estimator.predict(x_test)
    print(f'预测值为：{y_pred}')
    print(f'准确率：{estimator.score(x_test,y_test)}')
    print(f'准确率：{accuracy_score(y_test,y_pred)}')
    print(f'精确率：{precision_score(y_test,y_pred)}')
    print(f'召回率：{recall_score(y_test,y_pred)}')
    print(f'F1-score：{f1_score(y_test,y_pred)}')
    print(f'分类报告：{classification_report(y_test,y_pred)}')



if __name__ == "__main__":
    # dm01_data_processing()
    # dm02_data_visualization()
    dm03_logistic_regression()
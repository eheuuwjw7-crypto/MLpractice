import os
from asyncio import futures
from operator import index

import pandas as pd
import matplotlib.pyplot as plt
import datetime

from scipy.stats import power

from utils.log import Logger
from utils.common import data_preprocessing
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error, \
    mean_absolute_percentage_error
import joblib
plt.rcParams['font.family'] = 'Songti SC'
plt.rcParams['axes.unicode_minus'] = False

# 模型训练
class PowerLoadModel(object):
    def __init__(self,filename):
        logfile_name = 'train' + datetime.datetime.now().strftime('%Y%m%d%H%M%S') + 'log'
        self.logfile = Logger('../', logfile_name).get_logger()
        self.logfile.info('开始训练模型')
        self.data_source = data_preprocessing(filename)

# 数据可视化
def ana_data(data):
    ana_data = data.copy()
    # ana_data.info()
    fig = plt.figure(figsize=(40, 80))
    ax1 = fig.add_subplot(4, 1, 1)
    ax1.hist(ana_data['power_load'], bins=100)
    ax1.set_title('电力负荷分布直方图')
    ax1.set_xlabel('电力负荷')
    # ax1.set_ylabel('频数')

    ana_data['hour'] = ana_data['time'].str[11:13]
    hour_load_mean = ana_data.groupby('hour',as_index = False)['power_load'].mean()
    # print(hour_load_mean)
    ax2 = fig.add_subplot(4, 1, 2)
    ax2.plot(hour_load_mean['hour'], hour_load_mean['power_load'])
    ax2.set_title('每小时平均电力负荷')
    ax2.set_xlabel('时间')

    ana_data['month'] = ana_data['time'].str[5:7]
    month_load_mean = ana_data.groupby('month',as_index = False)['power_load'].mean()
    ax3 = fig.add_subplot(4, 1, 3)
    ax3.plot(month_load_mean['month'], month_load_mean['power_load'])
    ax3.set_title('每月平均电力负荷')
    ax3.set_xlabel('时间')

    ana_data['weekday'] = ana_data['time'].apply(lambda x:pd.to_datetime(x).weekday())
    ana_data['is_holiday'] = ana_data['weekday'].apply(lambda x:1 if x in [5,6] else 0)
    work_load_mean = ana_data[ana_data['is_holiday']==0].power_load.mean()
    holiday_load_mean = ana_data[ana_data['is_holiday']==1].power_load.mean()
    ax4 = fig.add_subplot(4, 1, 4)
    ax4.bar(['工作日平均电力负荷', '节假日平均电力负荷'],[work_load_mean,holiday_load_mean])
    ax4.set_title('工作日与节假日平均电力负荷')


    # print(ana_data.head(50))

    plt.savefig('../data/fig/电力负荷分布直方图1.png')
    plt.show()

# 特征工程
def feature_engineering(data,logger):
    feature_data = data.copy()
    feature_data['hour'] = feature_data['time'].str[11:13]
    feature_data['month'] = feature_data['time'].str[5:7]
    hour_month_data = pd.get_dummies(feature_data[['hour','month']])
    # print(feature_data.head())
    # print(feature_data.info())
    feature_data = pd.concat([feature_data,hour_month_data],axis=1)
    # print(feature_data.head())
    # print(feature_data.info())
    load_1h_data = feature_data['power_load'].shift(1)
    load_2h_data = feature_data['power_load'].shift(2)
    load_3h_data = feature_data['power_load'].shift(3)
    load_shift_data = pd.concat([load_1h_data,load_2h_data,load_3h_data],axis=1)
    load_shift_data.columns = ['load_1h','load_2h','load_3h']
    feature_data = pd.concat([feature_data,load_shift_data],axis=1)

    feature_data['yesterday'] = feature_data['time'].apply(lambda x:(pd.to_datetime(x)-datetime.timedelta(days=1)).strftime('%Y-%m-%d %H:%M:%S'))
    time_load_dict = feature_data.set_index('time')['power_load'].to_dict()
    feature_data['yesterday_load'] = feature_data['yesterday'].apply(lambda x:time_load_dict.get(x))
    feature_data = feature_data.dropna()
    feature_columns = list(hour_month_data.columns) + list(load_shift_data.columns) + ['yesterday_load']
    return feature_data,feature_columns

def model_train(data,features,logger):
    x = data[features]
    y = data['power_load']
    # print(x.shape,y.shape)
    # print(x.head())
    # print(y.head())
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

    logger.info('-------网格搜索 + 交叉验证 寻找最优超参--------')
    logger.info(f'开始时间：{datetime.datetime.now()}')
    param_grid = {
        'max_depth': [3, 4, 6, 8],
        'learning_rate': [0.01, 0.05, 0.1],
        'n_estimators': [100, 300, 500],
        'gamma': [0, 0.2, 0.4],
        'subsample': [0.7, 0.8, 1.0],
    }

    xgb_model = XGBRegressor()
    gs = GridSearchCV(xgb_model, param_grid, cv=2, scoring='neg_mean_squared_error', n_jobs=-1)
    gs.fit(x_train, y_train)
    logger.info(f'打印最优参数组合{gs.best_params_}')
    logger.info(f'训练结束时间：{datetime.datetime.now()}')

    xgb_model = XGBRegressor(**gs.best_params_)
    xgb_model.fit(x_train, y_train)
    y_pred = xgb_model.predict(x_test)
    logger.info(f'测试集的均方误差为：{mean_squared_error(y_test, y_pred)}')
    logger.info(f'测试集的均方根误差为：{root_mean_squared_error(y_test, y_pred)}')
    logger.info(f'平均绝对误差为：{mean_absolute_error(y_test, y_pred)}')
    logger.info(f'平均绝对百分比误差为：{mean_absolute_percentage_error(y_test, y_pred)}')

    joblib.dump(xgb_model,'../model/xgb_model.pkl')

if __name__ == '__main__':
    # 数据预处理
    model = PowerLoadModel('../data/train.csv')
    # print(model.data_source)

    # 数据可视化
    # ana_data(model.data_source)

    # 特征工程
    feature_data,feature_columns = feature_engineering(model.data_source,model.logfile)

    # 模型训练
    model_train(feature_data,feature_columns,model.logfile)
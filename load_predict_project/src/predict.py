import os
from asyncio import futures
from operator import index

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import datetime

from pandas.core.computation.expressions import evaluate
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

def pred_feature_extract(data_dict, time, logger):
    """
    预测数据解析特征，保持与模型训练时的特征列名一致
    1.解析时间特征
    2.解析时间窗口特征
    3.解析昨日同时刻特征
    :param data_dict:历史数据，字典格式，key：时间，value:负荷
    :param time:预测时间，字符串类型，格式为2024-12-20 09:00:00
    :param logger:日志对象
    :return:
    """
    logger.info(f'=========解析预测时间为：{time}所对应的特征==============')
    # 特征列清单
    feature_names = ['hour_00', 'hour_01', 'hour_02', 'hour_03', 'hour_04', 'hour_05',
                     'hour_06', 'hour_07', 'hour_08', 'hour_09', 'hour_10', 'hour_11',
                     'hour_12', 'hour_13', 'hour_14', 'hour_15', 'hour_16', 'hour_17',
                     'hour_18', 'hour_19', 'hour_20', 'hour_21', 'hour_22', 'hour_23',
                     'month_01', 'month_02', 'month_03', 'month_04', 'month_05', 'month_06',
                     'month_07', 'month_08', 'month_09', 'month_10', 'month_11', 'month_12',
                     'load_1h', 'load_2h', 'load_3h', 'yesterday_load']
    # 小时特征数据，使用列表保存起来
    pre_hour = time[11:13]
    hour_list = []
    for i in range(24):
        if pre_hour == feature_names[i][5:7]:
            hour_list.append(1)
        else:
            hour_list.append(0)
    # print(hour_list)

    pre_month = time[5:7]
    month_list = []
    for i in range(24,36):
        if pre_month == feature_names[i][6:8]:
            month_list.append(1)
        else:
            month_list.append(0)
    # print(month_list)

    last_1h_time = (pd.to_datetime(time) - datetime.timedelta(hours=1)).strftime('%Y-%m-%d %H:%M:%S')
    last_1h_load = data_dict.get(last_1h_time,500)
    # print(last_1h_load)
    last_2h_time = (pd.to_datetime(time) - datetime.timedelta(hours=2)).strftime('%Y-%m-%d %H:%M:%S')
    last_2h_load = data_dict.get(last_2h_time,500)
    last_3h_time = (pd.to_datetime(time) - datetime.timedelta(hours=3)).strftime('%Y-%m-%d %H:%M:%S')
    last_3h_load = data_dict.get(last_3h_time,500)

    yesterday_time = (pd.to_datetime(time) - datetime.timedelta(days=1)).strftime('%Y-%m-%d %H:%M:%S')
    yesterday_load = data_dict.get(yesterday_time,500)

    feature_values = hour_list + month_list + [
        last_1h_load,
        last_2h_load,
        last_3h_load,
        yesterday_load
    ]

    feature_data = pd.DataFrame([feature_values], columns=feature_names)
    # print(feature_data)
    return feature_data

def prediction_plot(data):
    fig = plt.figure(figsize=(20, 20))
    ax = fig.add_subplot()
    ax.plot(data['预测时间'],data['真实负荷'],color = 'b',label='真实负荷')
    ax.plot(data['预测时间'],data['预测值'],color = 'r',label='预测值')
    ax.set_title('电力负荷预测结果',fontsize=20)
    ax.set_xlabel('时间')
    ax.set_ylabel('电力负荷')
    ax.grid(True,linestyle='--',alpha=0.5)
    ax.legend(loc = 'best',fontsize=15)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(50))
    ax.set_xlabel('时间',fontsize = 15)
    ax.tick_params(axis='x', labelrotation=45)
    plt.savefig('../data/fig/最终预测.png')
    plt.show()

class PowerLoadPredict(object):
    def __init__(self,filename):
        logfile_name = 'predict' + datetime.datetime.now().strftime('%Y%m%d%H%M%S')
        self.logger = Logger('../', logfile_name).get_logger()
        self.logger.info('开始预测模型')
        self.data_source = data_preprocessing(filename)
        self.time_load_dict = self.data_source.set_index('time')['power_load'].to_dict()


if __name__ == '__main__':
    model = PowerLoadPredict('../data/test.csv')
    xgb_model = joblib.load('../model/xgb_model.pkl')
    pre_times = model.data_source['time'][model.data_source['time'] >= '2015-08-01 00:00:00']
    evaluate_list = []
    for pre_time in pre_times:
        # print(f'正在预测时间为：{pre_time}的电力负荷')
        time_load_dict_mask = {k:v for k,v in model.time_load_dict.items() if k < pre_time}
        # print(time_load_dict_mask)

        feature_df = pred_feature_extract(time_load_dict_mask,pre_time,model.logger)
        y_pred = xgb_model.predict(feature_df)
        model.logger.info(f'预测时间为：{pre_time}的电力负荷为：{y_pred[0]}')
        true_value = model.time_load_dict.get(pre_time,500)
        evaluate_list.append([pre_time,true_value,y_pred[0]])

    evaluate_df = pd.DataFrame(evaluate_list,columns=['预测时间','真实负荷','预测值'])
    # print(evaluate_df)
    print(f'平均绝对误差为：{mean_absolute_error(evaluate_df["真实负荷"],evaluate_df["预测值"])}')
    prediction_plot(evaluate_df)
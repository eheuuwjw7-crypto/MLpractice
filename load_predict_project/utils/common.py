import numpy as np
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / 'data'
LOG_DIR = BASE_DIR / 'log'
TRAIN_DIR = DATA_DIR / 'train.csv'
TEST_DIR = DATA_DIR / 'test.csv'


def data_preprocessing(file_path=TRAIN_DIR):
    data = pd.read_csv(file_path)
    # data.info()
    data['time'] = pd.to_datetime(data['time']).dt.strftime('%Y-%m-%d %H:%M:%S')
    # print(data.head())
    data.sort_values('time', ascending=True,inplace=True)
    data.drop_duplicates(inplace=True)

    return data


if __name__ == '__main__':
    data_preprocessing()
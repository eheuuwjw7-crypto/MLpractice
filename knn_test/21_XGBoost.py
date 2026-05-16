import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from collections import Counter
from sklearn.model_selection import train_test_split,StratifiedKFold,RandomizedSearchCV
from sklearn.metrics import classification_report
from pathlib import Path
from sklearn.utils import class_weight

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
ORI_DIR = DATA_DIR / "红酒品质分类.csv"
TRAIN_PATH = DATA_DIR / "红酒品质分类_train.csv"
TEST_PATH = DATA_DIR / "红酒品质分类_test.csv"
SAVE_DIR = BASE_DIR / "model"
MODEL_DIR = SAVE_DIR / "xgb_model.pkl"


def dm01_prepare_data():
    data = pd.read_csv(ORI_DIR)
    x = data.iloc[:, :-1]
    y = data.iloc[:, -1] - 3
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0,stratify=y)

    # print(f'标签情况：{Counter(y)}')

    pd.concat([x_train, y_train], axis=1).to_csv(TRAIN_PATH, index=False)
    pd.concat([x_test, y_test], axis=1).to_csv(TEST_PATH, index=False)

def dm02_train_model():
    train_data = pd.read_csv(TRAIN_PATH)
    test_data = pd.read_csv(TEST_PATH)

    x_train = train_data.iloc[:, :-1]
    y_train = train_data.iloc[:, -1]
    x_test = test_data.iloc[:, :-1]
    y_test = test_data.iloc[:, -1]
    # print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

    estimator = xgb.XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1,
                              objective='multi:softmax', n_jobs=-1, random_state=42)
    estimator.fit(x_train, y_train)
    y_pred = estimator.predict(x_test)
    print(classification_report(y_test, y_pred))
    joblib.dump(estimator, '../model/xgb_model.pkl')

def dm03_train_model():
    train_data = pd.read_csv(TRAIN_PATH)
    test_data = pd.read_csv(TEST_PATH)

    x_train = train_data.iloc[:, :-1]
    y_train = train_data.iloc[:, -1]
    x_test = test_data.iloc[:, :-1]
    y_test = test_data.iloc[:, -1]

    classes = np.unique(y_train)

    class_weights = class_weight.compute_class_weight(class_weight='balanced',
                                                      classes=classes,
                                                      y = y_train,
                                                      )

    class_weight_dict = dict(zip(classes, class_weights))
    sample_weights = y_train.map(class_weight_dict)

    estimator = xgb.XGBClassifier(n_estimators=300,
                                  max_depth=4,
                                  learning_rate=0.01,
                                  objective='multi:softprob',
                                  n_jobs=-1,
                                  random_state=42,
                                  min_child_weight=1
                                  )


    estimator.fit(x_train, y_train, sample_weight=sample_weights)
    y_pred = estimator.predict(x_test)
    print(classification_report(y_true=y_test, y_pred=y_pred))

    joblib.dump(estimator, MODEL_DIR)

def dm04_grid_search_xgb():
    train_data = pd.read_csv(TRAIN_PATH)
    test_data = pd.read_csv(TEST_PATH)

    x_train = train_data.iloc[:, :-1]
    y_train = train_data.iloc[:, -1]
    x_test = test_data.iloc[:, :-1]
    y_test = test_data.iloc[:, -1]

    spliter = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    param_grid = {
        'max_depth': [2, 3, 4, 5],
        'learning_rate': [0.01, 0.03, 0.05, 0.1],
        'n_estimators': [100, 200, 300, 500],
        'min_child_weight': [0.5, 1, 2, 3],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0],
        'gamma': [0, 0.1, 0.2],
        'reg_alpha': [0, 0.05, 0.1, 0.2],
        'reg_lambda': [0.8, 1, 1.2, 1.5],
    }

    classes = np.unique(y_train)

    class_weights = class_weight.compute_class_weight(class_weight='balanced',
                                                      classes=classes,
                                                      y=y_train,
                                                      )

    class_weight_dict = dict(zip(classes, class_weights))
    sample_weights = y_train.map(class_weight_dict)

    estimator = xgb.XGBClassifier(objective='multi:softmax',
                                  n_jobs=1,
                                  random_state=42,
                                  eval_metric='mlogloss',
                                  )
    estimator = RandomizedSearchCV(estimator,
                                   param_distributions=param_grid,
                                   n_iter=20,
                                   cv=spliter,
                                   scoring='f1_macro',
                                   n_jobs=-1,
                                   random_state=42)
    estimator.fit(x_train, y_train, sample_weight=sample_weights)
    y_pred = estimator.predict(x_test)
    print(classification_report(y_true=y_test, y_pred=y_pred, zero_division=0))
    print(estimator.best_score_)
    print(estimator.best_params_)
    print(estimator.best_estimator_)

    SAVE_DIR.mkdir(exist_ok=True)
    joblib.dump(estimator.best_estimator_, MODEL_DIR)

if __name__ == '__main__':
    # dm01_prepare_data()
    # dm02_train_model()
    # dm03_train_model()
    dm04_grid_search_xgb()

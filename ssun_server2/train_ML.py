import os, sys, time

import pandas as pd

sys.path.append("..")
from setproctitle import *

import torch
import numpy as np
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error

import pandas as pd
from config import Config
from tqdm import tqdm

def dataload_ML(opt):

    train = pd.read_csv(opt.dataset_path + 'train_ssun.csv').iloc[:, 1:]
    test = pd.read_csv(opt.dataset_path + 'train_ssun.csv').iloc[:, 1:]
    train_X, train_Y, test_X, test_Y = train.iloc[:, 4:], train.iloc[:, 0:4], test.iloc[:, 4:], test.iloc[:, 0:4]

    return train_X, train_Y, test_X, test_Y

def get_model(model_name):
    if model_name == 'LR':
        from sklearn.linear_model import LinearRegression
        from sklearn.multioutput import MultiOutputRegressor
        model = MultiOutputRegressor(LinearRegression())

    elif model_name == 'xgb':
        from xgboost import XGBRegressor
        from sklearn.multioutput import MultiOutputRegressor
        model = MultiOutputRegressor(XGBRegressor(n_estimators=1100, learning_rate=0.0085, gamma=0, subsample=0.75, colsample_bytree=1, max_depth=8, tree_method='gpu_hist', gpu_id=0))

    elif model_name == 'ransac':
        from sklearn.linear_model import LinearRegression, RANSACRegressor
        model = RANSACRegressor(LinearRegression(), max_trials=4500, min_samples=1000, loss='absolute_error', residual_threshold=4.0, random_state=0)

    elif model_name == 'lasso':
        from sklearn import linear_model
        from sklearn.multioutput import MultiOutputRegressor
        model = MultiOutputRegressor(linear_model.Lasso(alpha=0.1))

    elif model_name == 'Ridge':
        from sklearn import linear_model
        from sklearn.multioutput import MultiOutputRegressor
        model = MultiOutputRegressor(linear_model.Ridge(alpha=0.5))

    elif model_name == 'svr':
        from sklearn.svm import SVR
        from sklearn.multioutput import MultiOutputRegressor
        model = MultiOutputRegressor(SVR(kernel='poly', C=1.0, epsilon=0.2))

    elif model_name == 'krr':
        from sklearn.kernel_ridge import KernelRidge
        from sklearn.multioutput import MultiOutputRegressor
        model = MultiOutputRegressor(KernelRidge(kernel='linear', alpha=0.1, gamma=1.0))

    return model

def train_model(train_X, train_Y, test_X, test_Y):

    model_start_time = time.time()
    model_name = 'lasso' # LR | xgb | ransac | lasso | Ridge | svr | krr
    model = get_model(model_name)
    model.fit(train_X, train_Y)

    model_end_time = time.time()

    print("Model training time : %ds" %(model_end_time - model_start_time))

    predictions = model.predict(test_X)

    score = r2_score(test_Y, predictions)
    print("r2_score :", score)


if __name__ == '__main__':
    config = Config()

    setproctitle(config.opt.network_name)
    torch.cuda.set_device(int(config.opt.gpu_id))

    train_X, train_Y, test_X, test_Y = dataload_ML(config.opt)

    train_model(train_X, train_Y, test_X, test_Y)

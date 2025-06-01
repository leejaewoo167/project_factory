import os
import json
import random
import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score, f1_score
from sklearn.base import BaseEstimator, TransformerMixin
from loguru import logger

class IQRClipper(BaseEstimator, TransformerMixin):
    """_summary_
    Options:
        fit_transform(X): X에 대해 fitting하고 변환
        fit(X): X를 fitting
        transform():  fitting한 객체를 변환
    Args:
        BaseEstimator (_type_): _description_
        TransformerMixin (_type_): _description_
    """
    def __init__(self):
        self.bounds = {}

    def fit(self, X, y=None):
        self.bounds = []
        for i in range(X.shape[1]):
            q1 = np.percentile(X[:, i], 25)
            q3 = np.percentile(X[:, i], 75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            self.bounds.append((lower, upper))
        return self

    def transform(self, X):
        X_clipped = X.copy()
        for i, (lower, upper) in enumerate(self.bounds):
            X_clipped[:, i] = np.clip(X[:, i], lower, upper)
        return X_clipped


def load_model_config(file_path) -> dict:
    """_summary_

    Args:
        file_path (str): config 파일 path

    Raises:
        FileNotFoundError: config 파일이 없으면 error 발생

    Returns:
        param(dict): dictionary 형태로 parameter 반환 
    """
    if os.path.isfile(file_path):
        with open(file_path, 'r', encoding='utf-8') as json_config:
            param = json.load(json_config)
        return param
    else:
        raise FileNotFoundError("No config file")
    
def set_random_seed(seed):
    """_summary_
    - random seed를 전역적으로 고정합니다.
    - sklearn의 random state도 자동적으로 고정됩니다.
    
    Args:
        seed (int): seed number
    """
    
    random.seed(seed)
    np.random.seed(seed)

def log_param(param):
    """_summary_
    terminal 상에서 parameter의 정보를 log로 나타내어줍니다.
    
    Args:
        param (dict): parameters
    """
    for key, value in param.items():
        if type(value) is dict:
            for in_key, in_value in value.items():
                logger.info('{:20}:{:>50}'.format(
                    in_key, '{}'.format(in_value)))
        else:
            logger.info('{:20}:{:>50}'.format(key, '{}'.format(value)))

def set_parameters(param:list) -> list:
    """_summary_

    Args:
        param (list): 각 모델의 hyper-parameter 값 또는 범위

    Raises:
        ValueError: list길이가 0 또는 4 이상이면 에러 발생

    Returns:
        list: parameter tunning을 위한 candidate parameters
    """
    # floating point,,,,,,,,,,
    if len(param) == 1:
        return param
    
    elif len(param) == 2:
        p_min = param[0]
        p_max = param[1]
        return [p for p in np.arange(p_min, p_max+0.00001, dtype=type(p_max))]
    
    elif len(param) == 3:
        p_min = param[0]
        p_max = param[1]
        p_increment = param[2]
        return [p for p in np.arange(p_min, p_max+0.00001, p_increment, dtype=type(p_max))]
    
    else:
        raise ValueError("Incorrect length of input")
    
def metric(y_truth, y_hat) -> dict:
    """_summary_

    Args:
        y_truth (np.array): _description_
        y_hat (np.array): _description_

    Returns:
        dict: _description_
    """
    
    metric_dict = dict()
    f1_averages = ['binary', 'macro', 'micro']
    
    metric_dict['auc'] = roc_auc_score(y_true=y_truth,
                                       y_score=y_hat)
    
    for ave in f1_averages:
        metric_dict[f'{ave}_f1'] = f1_score(y_true=y_truth,
                                            y_pred=y_hat,
                                            average=ave)
    return metric_dict
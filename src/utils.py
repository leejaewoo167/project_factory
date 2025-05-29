import os
import json
import random
import copy
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score
from loguru import logger

def load_model_config(file_path):
    if os.path.isfile(file_path):
        with open(file_path, 'r', encoding='utf-8') as json_config:
            param = json.load(json_config)
        return param
    else:
        raise Exception("No config file")
    
def set_random_seed(seed):
    
    random.seed(seed)
    np.random.seed(seed)

def log_param(param):
    for key, value in param.items():
        if type(value) is dict:
            for in_key, in_value in value.items():
                logger.info('{:20}:{:>50}'.format(
                    in_key, '{}'.format(in_value)))
        else:
            logger.info('{:20}:{:>50}'.format(key, '{}'.format(value)))

def set_parameters(param:list) -> list:
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
        p_criterion = param[2]
        return [p for p in np.arange(p_min, p_max+0.00001, p_criterion, dtype=type(p_max))]
    
    else:
        raise "Invalid length of parameter"
    
def metric(y_truth, y_hat):
    
    metric_dict = dict()
    f1_averages = ['binary', 'macro', 'micro']
    
    metric_dict['auc'] = roc_auc_score(y_true=y_truth,
                                       y_score=y_hat)
    
    for ave in f1_averages:
        metric_dict[f'{ave}_f1'] = f1_score(y_true=y_truth,
                                            y_pred=y_hat,
                                            average=ave)
    return metric_dict
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV

from utils import logger, set_parameters

def Xgboost(params):
    
    param_set = {"max_depth" : set_parameters(params['max_depth']),
                 "n_estimators": set_parameters(params['n_estimators']), 
                 "learning_rate": set_parameters(params['learning_rate']),
                 "subsample": set_parameters(params['subsample'])}
    classifier = XGBClassifier()
    
    logger.info(f'load xgboost model')
    return {"parameters": param_set,
            "estimator": classifier}
# GridSearchCV(estimator=classifier, 
#                         param_grid=param_set)
from lightgbm import LGBMClassifier
from sklearn.model_selection import GridSearchCV
from utils import set_parameters, logger

def LightGBM(params):
    
    param_set = {
        "max_depth": set_parameters(params['depth']),
        "n_estimators": set_parameters(params['n_estimators']),
        "learning_rate": set_parameters(params['learning_rate']), 
        "num_leaves": set_parameters(params['num_leaves'])
    }
    classifier = LGBMClassifier()
    
    logger.info(f'load LightGBM model')
    return {"parameters": param_set, 
            "estimator": classifier}
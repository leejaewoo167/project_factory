from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.model_selection import GridSearchCV

from utils import logger, set_parameters

def Bagging(params):
    
    param_set = {
        
    }
    
    classifier = BaggingClassifier(
        estimator=LGBMClassifier(), 
        bootstrap=True,
        n_jobs=5) 

    logger.info(f'load Bagging model')

    return {"parameters": param_set, 
            "estimator": classifier}
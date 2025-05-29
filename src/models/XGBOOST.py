from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, BaggingClassifier

from utils import logger

def Xgboost(params):
    
    param_set = {}
    classifier = VotingClassifier(
        estimators=[
            ('randomforest', RandomForestClassifier()),
            ('xgboost', XGBClassifier(use_label_encoder=False, eval_metric='logloss')),
            ('bagged_lgbm', bagged_lgbm)],
        voting='soft'
        )
    
    logger.info(f'load Voting model')
    return {"parameters": param_set,
            "estimator": classifier}
# GridSearchCV(estimator=classifier, 
#                         param_grid=param_set)
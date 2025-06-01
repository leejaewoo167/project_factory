from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, BaggingClassifier
from utils import logger, set_parameters

def Voting(params):
    
    cand_model = ['rf','xgb', 'lgbm']
    
    param_set = {
        f'{model}__{name}': set_parameters(r_param)\
            for model in cand_model \
                for name, r_param in params[model].items() 
    }
    
    classifier = VotingClassifier(
        estimators=[
            ('rf', RandomForestClassifier()),
            ('xgb', XGBClassifier(use_label_encoder=False, eval_metric='logloss')),
            ('lgbm', LGBMClassifier())],
        voting='soft'
        )
    
    logger.info(f'load Voting model')
    return  {"parameters": param_set, 
            "estimator": classifier}
